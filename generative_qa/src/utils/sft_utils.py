import os

from transformers import Trainer, TrainingArguments


def preprocess_squad_train_samples(examples, tokenizer, max_length, stride):
    """ This function preprocesses the training samples for squad v1 and v2 datasets. 
    """
    #We want the input ids to be the context followed by the question followed by the answer if any otherwise the no answer token id
    answers = [a["text"][0].strip() if len(a["answer_start"]) > 0 else "" for a in examples["answers"]]
    tokenized_answers = tokenizer(answers)["input_ids"]
    
    no_answer_label = "The answer is not in the context."
    tokenized_no_answer = tokenizer(no_answer_label)["input_ids"]
    tokenizer.no_answer_token_ids = tokenized_no_answer + [tokenizer.eos_token_id]

    questions = [q.strip() for q in examples["question"]]
    questions = [q + "?" if q[-1] != "?" else q for q in questions]

    inputs = tokenizer(
        examples["context"],
        questions,
        max_length=max_length,
        truncation="only_first",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True, 
        padding=False, 
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")

    labels = []
    #For each features, 
    for i, offset in enumerate(offset_mapping):
        #Loop over all the examples that were broken down into chunks
        sample_idx = sample_map[i]
        answer = examples["answers"][sample_idx] #Example : {'text': ['Denver Broncos'], 'answer_start': [403]}
        if len(answer["answer_start"]) == 0:
            #Squad v2 : no answer possible
            labels.append(tokenized_no_answer + [tokenizer.eos_token_id])
            continue
        else:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i) #Array of 0s and 1s, with 1s corresponding to the second sentence in the pair.
            if sequence_ids[0] == None:
                #Depending on tokenizer the first token can be bos
                idx = 1
            else:
                idx = 0
            context_start = idx #Tokens idx that correspond to the start of the context
            while sequence_ids[idx] == 0:
                idx += 1
            context_end = idx - 1 #Tokens idx that correspond to the end of the context

            # If the answer is not fully inside the context, label is (0, 0)
            #offset is a list of tuples (start_char, end_char) for each token in the input sequence        
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                labels.append(tokenized_no_answer + [tokenizer.eos_token_id])
            else:
                labels.append(tokenized_answers[sample_idx] + [tokenizer.eos_token_id])    

    #Concatenation of question and context with answer related to them. Note: Context is truncated if it is too long
    inputs["input_ids"] = [x + y for x, y in zip(inputs["input_ids"], labels)]
    inputs["attention_mask"] = [x + [1] * len(y) for x, y in zip(inputs["attention_mask"], labels)]
    return inputs

def group_squad_text_for_train(examples, block_size, stride):
    """ This functions groups squad samples into blocks for LM training.
    """
    #Concatenate all elements of examples["input_ids"] into one list, same for attention_mask
    concatenated_input_ids = sum(examples["input_ids"], [])
    concatenated_attention_mask = sum(examples["attention_mask"], [])

    #Group the input_ids and attention_mask into blocks of size block_size and apply a stride of size stride. All block have to be of the same size.
    result = {
        "input_ids": [],
        "attention_mask": [],
    }

    for i in range(0, len(concatenated_input_ids), block_size - stride):
        result["input_ids"].append(concatenated_input_ids[i : i + block_size])
        result["attention_mask"].append(concatenated_attention_mask[i : i + block_size])
    
    #Discard all block that are not of size block_size
    result["input_ids"] = [x for x in result["input_ids"] if len(x) == block_size]
    result["attention_mask"] = [x for x in result["attention_mask"] if len(x) == block_size]
    
    #Add the labels
    result["labels"] = result["input_ids"].copy()
    
    return result


def supervised_finetuning(train_dataset, val_dataset, model, tokenizer, output_dir, max_steps, epochs, lr, batch_size, logger, fp16=False, optim="adamw_torch"):
    """ This function is used to perform SFT a model on a dataset.
    """
    trainer = Trainer(
        model = model,
        train_dataset = train_dataset,
        eval_dataset = val_dataset,
        args = TrainingArguments(
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            max_steps=max_steps,
            num_train_epochs=epochs,
            learning_rate=lr,
            fp16=fp16,
            logging_steps=50,
            output_dir=output_dir,
            save_strategy = "steps",
            evaluation_strategy = "steps",
            eval_steps = 50,
            save_steps = 2500,
            do_eval = True,
            optim=optim,
        ),
    )
    model.config.use_cache = False
    train_result = trainer.train()
    metrics = train_result.metrics
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()
    logger.info(metrics)                 
    # Saving model
    logger.info("Saving last checkpoint of the model...")
    trainer.model.save_pretrained(os.path.join(output_dir, "last_checkpoint"))
    tokenizer.save_pretrained(os.path.join(output_dir, "last_checkpoint"))
    
    # Free memory
    del model
    del trainer