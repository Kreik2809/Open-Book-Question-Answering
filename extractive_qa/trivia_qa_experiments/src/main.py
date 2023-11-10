import os
import argparse
import json
import logging
import datetime
import torch
import numpy as np

import utils.dataset_utils
import utils.utils
from triviaqa_evaluation import evaluate_triviaqa

from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForQuestionAnswering
from datasets import load_dataset
from torch.utils.data import DataLoader

from model_utils import compute_predictions, preprocess_validation_data

def main():
    #Get config
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/config.json')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))  

    #Gather all hyperparameters
    experiment_name = config['experiment_name']
    dataset_id = config['dataset_id']
    dataset_config = config['dataset_config']
    dataset_json = config['dataset_json']
    model_id = config['model_id']
    tokenizer_id = config['tokenizer_id']
    device = config['device']
    stride = config['stride']
    max_length = config['max_length']
    batch_size = config['batch_size']
    n_best = config['n_best']
    max_answer_length = config['max_answer_length']
    seed = config['seed']

    torch.manual_seed(seed)
    np.random.seed(seed)

    #Set up output dir and logger
    output_dir = os.path.join("..", 'outputs', experiment_name)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            os.remove(file_path)
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
    logger.info("Dataset config: " + str(dataset_config))
    logger.info("Dataset json: " + dataset_json)
    logger.info("Model: " + model_id)
    logger.info("Tokenizer: " + tokenizer_id)
    logger.info("Device: " + device)
    logger.info("Stride: " + str(stride))
    logger.info("Max length: " + str(max_length))
    logger.info("Batch size: " + str(batch_size))
    logger.info("N best: " + str(n_best))
    logger.info("Max answer length: " + str(max_answer_length))
    logger.info("Seed: " + str(seed))

    #Load dataset, tokenizer and model
    dataset = load_dataset(dataset_id, dataset_config)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForQuestionAnswering.from_pretrained(model_id).to(device)

    #Start inference on validation data
    validation_dataset_raw = dataset["validation"]
    n=100
    predictions = {}
    with open(os.path.join(output_dir, "predictions.json"), 'w') as f:
        json.dump(predictions, f)
    
    for i in range(1, len(validation_dataset_raw) // n + 1):
        #Preprocess batch
        logger.info("Preprocessing batch " + str(i) + " - Remaining samples: " + str(len(validation_dataset_raw) - i*n))
        validation_dataset_batch =  validation_dataset_raw.select(range((i-1)*n, i*n))

        validation_samples = validation_dataset_batch.map(
            lambda x: preprocess_validation_data(x, tokenizer, max_length, stride),
            remove_columns=validation_dataset_raw.column_names,
            batched=True,
        )

        validation_samples.set_format("torch")
        #get examples ids and offset mapping columns
        examples_ids = validation_samples["example_id"]
        offset_mapping = validation_samples["offset_mapping"]
        context = validation_samples["contexts"]
        validation_samples = validation_samples.remove_columns(["example_id", "offset_mapping", "contexts"])

        val_dataloader = DataLoader(validation_samples, batch_size=batch_size, shuffle=False)
        device = torch.device(device)
        model.to(device)

        #Inference on batch
        logger.info("Starting inference for batch " + str(i) + " of validation data")
        model.eval()
        start_logits = []
        end_logits = []
        for batch in tqdm(val_dataloader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                start_logits.append(outputs.start_logits.cpu().numpy())
                end_logits.append(outputs.end_logits.cpu().numpy())
        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)

        logger.info("Computing predictions for batch " + str(i) + " of validation data")
        temp_predictions = compute_predictions(start_logits, end_logits, offset_mapping, context, examples_ids, n_best, max_answer_length)
        logger.info("Saving predictions for batch " + str(i) + " of validation data")

        #Save predictions of batch
        with open(os.path.join(output_dir, "predictions.json"), 'a') as f:
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 1, os.SEEK_SET)
            f.truncate() 
            if i > 1:
                f.write(", ")
            f.write(json.dumps(temp_predictions)[1:-1] + "}")
        
        #Reset memory
        del val_dataloader
        del validation_samples
        del start_logits
        del end_logits
        del temp_predictions
        del context
        del offset_mapping
        del examples_ids
        if device == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    #Evaluate predictions and save results
    logger.info("Starting evaluation on validation set")
    dataset_json = utils.dataset_utils.read_triviaqa_data("web-dev.json")
    key_to_ground_truth = utils.dataset_utils.get_key_to_ground_truth(dataset_json)

    with open(os.path.join(output_dir, "predictions.json"), 'r') as f:
        predictions = json.load(f)

    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)

    with open(os.path.join(output_dir, "eval_dict_full.json"), 'w') as f:
        json.dump(eval_dict, f)

    dataset_json = utils.dataset_utils.read_triviaqa_data("verified-web-dev.json")
    key_to_ground_truth = utils.dataset_utils.get_key_to_ground_truth(dataset_json)

    with open(os.path.join(output_dir, "predictions.json"), 'r') as f:
        predictions = json.load(f)

    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)

    with open(os.path.join(output_dir, "eval_dict_verified.json"), 'w') as f:
        json.dump(eval_dict, f)
    
    logger.info("Evaluation results (full): " + str(eval_dict))

if __name__ == '__main__':
    main()