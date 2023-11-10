import os
import logging
import datetime
import torch

import numpy as np
import pandas as pd

from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForQuestionAnswering, default_data_collator, get_scheduler
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torch.optim import AdamW

from utils import preprocess_training_examples, preprocess_validation_examples, compute_metrics


def train(experiment_name, dataset_id, model_id, device, batch_size, epochs, lr, max_length, stride, max_answer_length, n_best, seed):
    """ This function trains a model on the SQuAD v1 or SQuAD v2 dataset.
    """
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

    # Load the dataset and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForQuestionAnswering.from_pretrained(model_id)
    raw_datasets = load_dataset(dataset_id)

    logger.info("Starting training for experiment: " + experiment_name)
    logger.info("Dataset: " + dataset_id)
    logger.info("Model: " + model_id)
    logger.info("Device: " + device)
    logger.info("Batch size: " + str(batch_size))
    logger.info("Epochs: " + str(epochs))
    logger.info("lr: " + str(lr))
    logger.info("Max length: " + str(max_length))
    logger.info("Stride: " + str(stride))
    logger.info("Max answer length: " + str(max_answer_length))
    logger.info("N best: " + str(n_best))
    logger.info("Seed: " + str(seed))

    # Preprocess the dataset
    train_dataset = raw_datasets["train"].map(
        lambda x: preprocess_training_examples(x, tokenizer, max_length, stride),
        batched=True,
        remove_columns=raw_datasets["train"].column_names,
    )

    validation_dataset = raw_datasets["validation"].map(
        lambda x: preprocess_validation_examples(x, tokenizer, max_length, stride),
        batched=True,
        remove_columns=raw_datasets["validation"].column_names,
    )

    logger.info("Preprocessing done")

    train_dataset.set_format("torch")
    validation_set = validation_dataset.remove_columns(["example_id", "offset_mapping"])
    validation_set.set_format("torch")

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=default_data_collator,
        batch_size=batch_size
    )
    eval_dataloader = DataLoader(
        validation_set, 
        collate_fn=default_data_collator, 
        batch_size=batch_size
    )

    device=torch.device(device)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=lr)
    num_update_steps_per_epoch = len(train_dataloader)
    num_training_steps = epochs * num_update_steps_per_epoch

    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )

    logger.info("Starting training")
    progress_bar = tqdm(range(num_training_steps))

    results = pd.DataFrame(columns=["Epoch", "Train loss", "Validation-EM", "Validation-F1"])

    for epoch in range(epochs):
        # Training
        model.train()
        epoch_losses = []
        for step, batch in enumerate(train_dataloader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            epoch_losses.append(loss.detach().cpu().numpy())

            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            progress_bar.update(1)
        
        # Evaluation
        model.eval()
        start_logits = []
        end_logits = []
        for batch in tqdm(eval_dataloader):
            with torch.no_grad():
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)

            start_logits.append(outputs.start_logits.cpu().detach().numpy())
            end_logits.append(outputs.end_logits.cpu().detach().numpy())

        start_logits = np.concatenate(start_logits)
        end_logits = np.concatenate(end_logits)
        start_logits = start_logits[: len(validation_dataset)] #Remove the predictions for the padded samples
        end_logits = end_logits[: len(validation_dataset)] #Remove the predictions for the padded samples

        metrics = compute_metrics(
            start_logits, end_logits, validation_dataset, raw_datasets["validation"], n_best, max_answer_length
        )

        logger.info("epoch {} training loss:".format(epoch) + str(np.mean(epoch_losses)))
        logger.info(f"epoch {epoch} validation metrics:" + str(metrics))

        results.loc[len(results)] = [
            epoch,
            np.mean(epoch_losses),
            metrics["exact"],
            metrics["f1"]       
        ]

    #Model saving
    logger.info("Training done")    
    logger.info("Saving model and results")
    model.save_pretrained(output_dir)
    results = results.round(3)
    results["Epoch"] = results["Epoch"].astype(int)
    results.to_pickle(os.path.join(output_dir, "results.pkl"))
    results.to_latex(os.path.join(output_dir, "results.tex"), index=False, column_format="|r|r|r|r|r|", float_format=(lambda x: "%.3f" % x))
    logger.info("Model and results saved")