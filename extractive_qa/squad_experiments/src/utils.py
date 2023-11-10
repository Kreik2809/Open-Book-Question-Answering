import collections
import evaluate

import numpy as np

from tqdm.auto import tqdm

def preprocess_training_examples(examples, tokenizer, max_length, stride):  
    """ This function preprocesses the training examples to generate the labels.
    """
    questions = [q.strip() for q in examples["question"]] 
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    offset_mapping = inputs.pop("offset_mapping")
    sample_map = inputs.pop("overflow_to_sample_mapping")
    answers = examples["answers"]
    start_positions = []
    end_positions = []

    for i, offset in enumerate(offset_mapping):

        #Loop over all the examples that were broken down into chunks
        sample_idx = sample_map[i]
        answer = answers[sample_idx] #Example : {'text': ['Denver Broncos'], 'answer_start': [403]}
        if len(answer["answer_start"]) == 0:
            #Squad v2 : no answer possible
            start_positions.append(0)
            end_positions.append(0)
            continue
        else:
            start_char = answer["answer_start"][0]
            end_char = answer["answer_start"][0] + len(answer["text"][0])
            sequence_ids = inputs.sequence_ids(i) #Array of 0s and 1s, with 1s corresponding to the second sentence in the pair.

            # Find the start and end of the context
            idx = 0
            while sequence_ids[idx] != 1:
                idx += 1
            context_start = idx #Tokens idx that correspond to the start of the context
            while sequence_ids[idx] == 1 and idx < len(sequence_ids) - 1:
                idx += 1
            context_end = idx - 1 #Tokens idx that correspond to the end of the context

            # If the answer is not fully inside the context, label is (0, 0)
            #offset is a list of tuples (start_char, end_char) for each token in the input sequence        
            if offset[context_start][0] > start_char or offset[context_end][1] < end_char:
                start_positions.append(0)
                end_positions.append(0)
            else:
                # Otherwise it's the start and end token positions
                idx = context_start
                while idx <= context_end and offset[idx][0] <= start_char:
                    idx += 1
                start_positions.append(idx - 1)

                idx = context_end
                while idx >= context_start and offset[idx][1] >= end_char:
                    idx -= 1
                end_positions.append(idx + 1)

    #Labels for training
    inputs["start_positions"] = start_positions 
    inputs["end_positions"] = end_positions
    return inputs


def preprocess_validation_examples(examples, tokenizer, max_length, stride):
    """ This function preprocesses the validation examples to compute the evaluation metrics.
    """
    #No labels are generated in this function because we don't want to compute validation loss. We only want to compute the evaluation metrics which rely on the predictions.
    questions = [q.strip() for q in examples["question"]]
    inputs = tokenizer(
        questions,
        examples["context"],
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    example_ids = []

    for i in range(len(inputs["input_ids"])):
        sample_idx = sample_map[i]
        example_ids.append(examples["id"][sample_idx])
        answer = examples["answers"][sample_idx]

        sequence_ids = inputs.sequence_ids(i) #Sequence ids are 0 for the first sentence and 1 for the second sentence (here context and question)
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ] #We want the offset mapping to be None for the tokens that are not part of the context so that it is possible to skip answer that are not in the context completely.
        

    inputs["example_id"] = example_ids #Store the id of the example so that we can use it to evaluate the predictions
    return inputs 


metric = evaluate.load("squad_v2") #Define the format of the predictions and the true answers

def compute_metrics(start_logits, end_logits, features, examples, n_best, max_answer_length):
    """ This function computes the evaluation metrics.
    """
    #example are the original examples before tokenization
    #features are the examples after tokenization (one example can be broken down into several features)
    example_to_features = collections.defaultdict(list)
    for idx, feature in enumerate(features):
        example_to_features[feature["example_id"]].append(idx)
    #example_to_features is a dictionary that maps each example to the list of features that were generated from that example

    predicted_answers = []
    for example in tqdm(examples): #For each example (I.e. each sample in the validation set)
        example_id = example["id"]
        context = example["context"]
        answers = []

        # Loop through all features associated with that example
        for feature_index in example_to_features[example_id]:
            start_logit = start_logits[feature_index]
            end_logit = end_logits[feature_index]
            offsets = features[feature_index]["offset_mapping"]

            start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
            end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
            for start_index in start_indexes:
                for end_index in end_indexes:
                    # Skip answers that are not fully in the context
                    if offsets[start_index] is None or offsets[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offsets[start_index][0] : offsets[end_index][1]], #The answer is the text between the start and end index
                        "logit_score": start_logit[start_index] + end_logit[end_index], #The score is the sum of the start and end logit
                        "no_answer_probability": (start_logit[0] + end_logit[0]) - (start_logit[start_index] + end_logit[end_index]), #The score of the null answer is the sum of the start and end logit for the first token
                    }
                    answers.append(answer)

        # Select the answer with the best score
        if len(answers) > 0:
            best_answer = max(answers, key=lambda x: x["logit_score"])
            predicted_answers.append(
                {"id": example_id, "prediction_text": best_answer["text"], "no_answer_probability": best_answer["no_answer_probability"]}
            )
        else:
            # In the very rare edge case we have not a single non-null prediction, we create a fake prediction to avoid failure
            predicted_answers.append({"id": example_id, "prediction_text": "", "no_answer_probability": 0.})

    theoretical_answers = [{"id": ex["id"], "answers": ex["answers"]} for ex in examples]
    return metric.compute(predictions=predicted_answers, references=theoretical_answers)
