import os
import argparse
import json
import logging
import datetime
import torch
import numpy as np
import bitsandbytes as bnb

from tqdm import tqdm
from datasets import load_dataset, concatenate_datasets

from model_analysis import make_learning_curves, make_plots, make_table
from utils.dpo_utils import dpo_finetuning, preprocess_squad_dpo_samples
from utils.sft_utils import supervised_finetuning, preprocess_squad_train_samples, group_squad_text_for_train
from utils.validation_utils import compute_metrics_squad, save_prediction_to_file, preprocess_squad_val_samples, process_outputs
from utils.model_utils import load_model_for_training, load_model_for_inference


def main():
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/config.json')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))  
    print(config)

    #Gather all hyperparameters
    experiment_name = config['experiment_name']
    dataset_id = config['dataset_id']
    model_id = config['model_id']
    tokenizer_id = config['tokenizer_id']
    device = config['device']
    max_steps = config['max_steps']
    epochs = config['epochs']
    Qlora = config['Qlora']
    train = config['train']
    val = config['val']
    few_shot = config['few_shot']
    K = config['K']
    lr = config['lr']
    beta = config['beta']
    stride = config['stride']
    max_length = config['max_length']
    block_size = config['block_size']
    batch_size = config['batch_size']
    max_answer_length = config['max_answer_length']
    seed = config['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)

    #Set up output dir and logger
    output_dir = os.path.join("..", 'outputs', experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        #Clean output dir before starting
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            #if file path is a directory, remove the directory
            if os.path.isdir(file_path):
                for sub_filename in os.listdir(file_path):
                    sub_file_path = os.path.join(file_path, sub_filename)
                    os.remove(sub_file_path)
                os.rmdir(file_path)
            else:
                os.remove(file_path)
    
    #Set up logger
    log_file = os.path.join(output_dir, "train_log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.info("Starting evaluation for experiment: " + experiment_name)
    logger.info("Dataset: " + dataset_id)
    logger.info("Model: " + model_id)
    logger.info("Tokenizer: " + tokenizer_id)
    logger.info("Device: " + device)
    logger.info("Epochs: " + str(epochs))
    logger.info("Max steps: " + str(max_steps))
    logger.info("Learning rate: " + str(lr))
    logger.info("Beta: " + str(beta))
    logger.info("Stride: " + str(stride))
    logger.info("Max length: " + str(max_length))
    logger.info("Batch size: " + str(batch_size))
    logger.info("Block size: " + str(block_size))
    logger.info("Max answer length: " + str(max_answer_length))
    logger.info("Qlora: " + str(Qlora))
    logger.info("Train: " + str(train))
    logger.info("Val: " + str(val))
    logger.info("Few shot: " + str(few_shot))
    logger.info("K: " + str(K))
    logger.info("Seed: " + str(seed))
    
    #Load dataset
    raw_dataset = load_dataset(dataset_id)

    if train == "SFT":
        logger.info("SFT Training")
        #Load model
        model, tokenizer = load_model_for_training(model_id, tokenizer_id, output_dir, Qlora, device)
        #Preprocessing
        if few_shot == "True":
            lm_train_dataset_begin = raw_dataset["train"].select(range(int(K * 0.7)))
            lm_train_dataset_end = raw_dataset["train"].select(range(len(raw_dataset["train"]) - int(K * 0.3), len(raw_dataset["train"])))
            lm_train_dataset = concatenate_datasets([lm_train_dataset_begin, lm_train_dataset_end])
        else:
            lm_train_dataset = raw_dataset["train"]

        lm_train_dataset = lm_train_dataset.map(
            lambda x: preprocess_squad_train_samples(x, tokenizer, max_length, stride), 
            batched=True, 
            remove_columns=["id", "title", "context", "question", "answers"],
            load_from_cache_file=False
        )

        lm_train_dataset = lm_train_dataset.map(
            lambda examples: group_squad_text_for_train(examples, block_size=block_size, stride=512),
            batched=True,
            load_from_cache_file=False,
        )

        lm_val_dataset = raw_dataset["validation"].select(range(1024, 2048)).map(
            lambda x: preprocess_squad_train_samples(x, tokenizer, max_length, stride), 
            batched=True, 
            remove_columns=["id", "title", "context", "question", "answers"],
            load_from_cache_file= False
        )

        lm_val_dataset = lm_val_dataset.map(
            lambda examples: group_squad_text_for_train(examples, block_size=block_size, stride=512),
            batched=True,
            load_from_cache_file=False,
        )

        lm_train_dataset.set_format("torch")
        lm_val_dataset.set_format("torch")

        #Train
        logger.info("Memory usage: " + str(torch.cuda.memory_allocated(device=device) / 1024 / 1024 / 1024) + " GB")
        logger.info("Starting training")

        if Qlora == "True":
            supervised_finetuning(lm_train_dataset, lm_val_dataset, model, tokenizer, output_dir, max_steps, epochs, lr, batch_size, logger, fp16=True, optim="paged_adamw_8bit")
        else:
            supervised_finetuning(lm_train_dataset, lm_val_dataset, model, tokenizer, output_dir, max_steps, epochs, lr, batch_size, logger)
        make_learning_curves(output_dir)
    
    if train == "DPO":
        logger.info("DPO Training")
        #Load model
        model, tokenizer = load_model_for_inference(model_id, tokenizer_id, output_dir, Qlora, device, logger)
        #DPO dataset creation
        logger.info("DPO dataset creation")
        if few_shot == "True":
            lm_train_dataset_begin = raw_dataset["train"].select(range(int(K * 0.5)))
            lm_train_dataset_end = raw_dataset["train"].select(range(len(raw_dataset["train"]) - int(K * 0.5), len(raw_dataset["train"])))
            lm_train_dataset = concatenate_datasets([lm_train_dataset_begin, lm_train_dataset_end])
        else:
            lm_train_dataset = raw_dataset["train"]
        
        dpo_train_dataset = lm_train_dataset.map(
            lambda x: preprocess_squad_dpo_samples(x, model, tokenizer, device), 
            batched=True, 
            remove_columns=["id", "title", "context", "question", "answers"],
            load_from_cache_file=False
        )
    
        dpo_train_dataset = dpo_train_dataset.shuffle(seed=seed)
        dpo_train_dataset.set_format("torch")

        model, tokenizer = load_model_for_training(model_id, tokenizer_id, output_dir, Qlora, device)

        #Train
        logger.info("Training started")
        if Qlora == "True":
            dpo_finetuning(model, tokenizer, dpo_train_dataset, output_dir, max_steps, epochs, lr, beta, batch_size, logger, optim="paged_adamw_8bit")
        else:
            dpo_finetuning(model, tokenizer, dpo_train_dataset, output_dir, max_steps, epochs, lr, beta, batch_size, logger)

    if val == "True":   
        logger.info("Validation")
        #Load model
        model, tokenizer = load_model_for_inference(model_id, tokenizer_id, output_dir, Qlora, device, logger)
        #Preprocessing
        lm_test_dataset = raw_dataset["validation"].select(range(1024)).map(
            lambda x: preprocess_squad_val_samples(x, tokenizer, max_length, stride), 
            batched=True, 
            remove_columns=["id", "title", "context", "question", "answers"],
            load_from_cache_file= False
        )
        test_example_ids = lm_test_dataset["example_id"]
        lm_test_dataset =lm_test_dataset.remove_columns(["example_id", "offset_mapping"])
        lm_test_dataset.set_format("torch")

        test_dataloader = torch.utils.data.DataLoader(
            lm_test_dataset, batch_size=1, shuffle=False #Batch size=1 because padding is impossible since generate don't use attention mask to ignore padded tokens
        )

        #Start inference
        logger.info("Validation started")
        for i, batch in enumerate(tqdm(test_dataloader)):
            batch = {k: v.to(device) for k, v in batch.items()}
            with torch.no_grad():
                outputs = model.generate(**batch, 
                                            do_sample=False,
                                            num_beams=20, 
                                            max_new_tokens=max_answer_length, 
                                            num_return_sequences=10, 
                                            eos_token_id=tokenizer.eos_token_id, 
                                            pad_token_id=tokenizer.pad_token_id, 
                                            return_dict_in_generate=True, 
                                            output_scores=True
                                    )
                selected_sentence, sequence_score, no_answer_logit = process_outputs(outputs, batch, tokenizer, model, device)
                    
            save_prediction_to_file(tokenizer, selected_sentence, sequence_score, no_answer_logit, test_example_ids[i * 1 : (i + 1) * 1], output_dir, batch["input_ids"].shape[1])
            #reset memory
            del batch
            if device == "cuda":
                torch.cuda.empty_cache() 

        #Compute metrics and save results
        metrics = compute_metrics_squad(output_dir, test_example_ids, raw_dataset["validation"].select(range(1024)))
        logger.info("Metrics: " + str(metrics))
        with open(os.path.join(output_dir, "metrics.json"), "w") as f:
            json.dump(metrics, f)
        make_plots(output_dir, raw_dataset["validation"].select(range(1024)))
        
    logger.info("Finished experiment: " + experiment_name)

if __name__ == '__main__':
    main() 


        
    
