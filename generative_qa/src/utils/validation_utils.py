import torch
import json
import os
import evaluate


def preprocess_squad_val_samples(examples, tokenizer, max_length, stride):
    """ This function preprocesses the validation samples for squad v1 and v2 datasets.
    """
    no_answer_label = "The answer is not in the context."
    tokenized_no_answer = tokenizer(no_answer_label)["input_ids"]
    tokenizer.no_answer_token_ids = tokenized_no_answer + [tokenizer.eos_token_id]
    #No labels are generated in this function because we don't want to compute validation loss. We only want to compute the evaluation metrics which rely on the predictions.
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
        padding=False,  #because padding is impossible since generate don't use attention mask to ignore padded tokens. Since then, batch size for val is 1
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])

        sequence_ids = inputs.sequence_ids(i) #Sequence ids are 0 for the first sentence and 1 for the second sentence (here context and question)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 0 else None for k, o in enumerate(offset)
        ] #We want the offset mapping to be None for the tokens that are not part of the context so that it is possible to skip answer that are not in the context completely.
        
    inputs["example_id"] = example_ids #Store the id of the example so that we can use it to evaluate the predictions
    return inputs 

def to_tokens_and_logprobs(model, tokenizer, input_ids):
    """ This function returns the tokens and the logprobs of the tokens for a given input_ids.
    """
    with torch.no_grad():
        outputs = model(input_ids)
    probs = torch.log_softmax(outputs.logits, dim=-1).detach()

    # collect the probability of the generated token -- probability at index 0 corresponds to the token at index 1
    probs = probs[:, :-1, :]
    input_ids = input_ids[:, 1:]
    gen_probs = torch.gather(probs, 2, input_ids[:, :, None]).squeeze(-1)

    batch = []
    for input_sentence, input_probs in zip(input_ids, gen_probs):
        text_sequence = []
        for token, p in zip(input_sentence, input_probs):
            text_sequence.append((tokenizer.decode(token), p.item()))
        batch.append(text_sequence)
    
    #reset memory
    del outputs
    torch.cuda.empty_cache()
    
    return batch

def is_similar(sequence1, sequence2, tokenizer):
    """ This function returns True if sequence1 and sequence2 are similar and False otherwise.
    """
    sequence1 = tokenizer.decode(sequence1, skip_special_tokens=True)
    sequence2 = tokenizer.decode(sequence2, skip_special_tokens=True)
    sequence1 = ''.join([c if c.isalnum() else ' ' for c in sequence1])
    sequence2 = ''.join([c if c.isalnum() else ' ' for c in sequence2])
    sequence1 = sequence1.strip()
    sequence2 = sequence2.strip()
    sequence1 = sequence1.lower()
    sequence2 = sequence2.lower()
    return all(word in sequence1 for word in sequence2.split()) or all(word in sequence2 for word in sequence1.split())

def process_outputs(outputs, batch, tokenizer, model, device): 
    """ This function process the outputs of the generate function to extract the generated sequence, the score of the sequence and the score of the no answer sequence.
    """

    no_answer_token_ids = tokenizer.no_answer_token_ids
    selected_sequence = outputs.sequences[0]
    
    if is_similar(selected_sequence[batch["input_ids"].shape[1]:], no_answer_token_ids, tokenizer):
        for i, sequence in enumerate(outputs.sequences):
            if not is_similar(sequence[batch["input_ids"].shape[1]:], no_answer_token_ids, tokenizer):
                selected_sequence = sequence
                break 
     
    context = selected_sequence[:batch["input_ids"].shape[1]]
    no_answer_sequence = torch.cat([context, torch.tensor(no_answer_token_ids, device=device)])

    selected_sequence_score = to_tokens_and_logprobs(model, tokenizer, selected_sequence.unsqueeze(0))
    no_answer_score = to_tokens_and_logprobs(model, tokenizer, no_answer_sequence.unsqueeze(0))

    for i, (token, p) in enumerate(selected_sequence_score[0][batch["input_ids"].shape[1] -1:]):
        if token == tokenizer.eos_token or token == tokenizer.pad_token or token == tokenizer.bos_token:
            selected_sequence_score = [selected_sequence_score[0][batch["input_ids"].shape[1] -1:batch["input_ids"].shape[1] + i]]
            break
    no_answer_score = [no_answer_score[0][batch["input_ids"].shape[1] -1:]] #Scores are shifted by one

    sentence_score = [sum([p for _, p in selected_sequence_score[0]]) / len(selected_sequence_score[0])]
    no_answer_score = [sum([p for _, p in no_answer_score[0]]) / len(no_answer_score[0])]

    selected_sequence = selected_sequence.unsqueeze(0)

    return selected_sequence, sentence_score, no_answer_score

metric = evaluate.load("squad_v2") #Define the format of the predictions and the true answer

def compute_metrics_squad(output_dir, example_ids, examples):
    """ This function computes the metrics for the squad v1 and v2 datasets from the predictions json file where is sotred the dict of id-predicted answers list.
    """
    predicted_answers = []
    
    #Read the predictions from the predictions.json file
    with open(os.path.join(output_dir, "predictions.json"), 'r') as f:
        predictions = json.load(f) 
    
    for example_ids in predictions.keys():
        predictions[example_ids] = max(predictions[example_ids], key=lambda x: x["logit_score"])
        predicted_answers.append(
            {
                "id": example_ids,
                "prediction_text": predictions[example_ids]["text"],
                "no_answer_probability": predictions[example_ids]["no_answer_probability"],
            }
        )
    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]

    return metric.compute(predictions=predicted_answers, references=theoretical_answers)

def save_prediction_to_file(tokenizer, sequence, sequences_scores, no_answer_logits, example_ids, output_dir, max_length):
    """ This function save the predictions outputed by the generate function to a json file.
    """    
    try:
        with open(os.path.join(output_dir, "predictions.json"), 'r') as f:
            predictions = json.load(f)
    except FileNotFoundError:
        predictions = {}

    for i, example_id in enumerate(example_ids):
        if example_id not in predictions:
            predictions[example_id] = []

        predicted_answer = tokenizer.decode(sequence[i][max_length:], skip_special_tokens=True)
        predicted_answer = predicted_answer.strip()
        predicted_answer = ''.join([c if c.isalnum() else ' ' for c in predicted_answer])
        #if predicted_answer != "": For triviaqa, no answer are handled directly in the compute metrics function: For Squad v2, we need no_answer_score
        predictions[example_id].append({"text" : predicted_answer, "logit_score" : sequences_scores[i], "no_answer_probability": no_answer_logits[i]}) #- sequences_scores[i]})
    
    with open(os.path.join(output_dir, "predictions.json"), 'w') as f:
        json.dump(predictions, f)