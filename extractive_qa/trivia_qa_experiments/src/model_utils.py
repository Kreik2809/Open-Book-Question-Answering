import torch
import math
import collections
import numpy as np

def preprocess_validation_data(examples, tokenizer, max_length, stride):
    """ Preprocesses validation data to create the dataset.
    """
    initial_questions  = [q for q in examples["question"]] 
    question_ids = [q_id for q_id in examples["question_id"]]
    #Each question has a list of entity pages and a list of search results
    entity_page_files = [epf["filename"] for epf in examples["entity_pages"]] 
    search_context_files = [scf["filename"] for scf in examples["search_results"]]

    #Create the list of all context so that we can tokenize them all at once    
    contexts = []
    questions = []
    example_ids = []
    for i, (epf, scf) in enumerate(zip(entity_page_files, search_context_files)):
        #For each example, we have a list of entity pages and a list of search results that are contexts for the question
        #Each pair of question/context is identified by a unique id
        for j, file in enumerate(epf):
            questions.append(initial_questions[i])
            context = examples["entity_pages"][i]["wiki_context"][j]
            contexts.append(context)
            example_ids.append(str(question_ids[i]) + "--" + file)
        for j, file in enumerate(scf):
            questions.append(initial_questions[i])
            context = examples["search_results"][i]["search_context"][j]
            contexts.append(context)
            example_ids.append(str(question_ids[i]) + "--" + file)

    inputs = tokenizer(
        questions,
        contexts,
        max_length=max_length,
        truncation="only_second",
        stride=stride,
        return_overflowing_tokens=True,
        return_offsets_mapping=True,
        padding="max_length",
    )

    sample_map = inputs.pop("overflow_to_sample_mapping")
    #We want example_ids to be expanding depending on the number pieces of context split by the tokenizer
    example_ids = [example_ids[sample_idx] for sample_idx in sample_map]
    #We want contexts to be expanding depending on the number pieces of context split by the tokenizer
    contexts = [contexts[sample_idx] for sample_idx in sample_map]

    for i in range(len(inputs["input_ids"])):
        sequence_ids = inputs.sequence_ids(i) #Array of 0s and 1s, with 1s corresponding to the second sentence in the pair.
        offset = inputs["offset_mapping"][i]
        inputs["offset_mapping"][i] = [
            o if sequence_ids[k] == 1 else None for k, o in enumerate(offset)
        ] #We want the offset mapping to be None for the tokens that are not part of the context so that it is possible to skip answer that are not in the context completely.
        
    inputs["example_id"] = example_ids
    inputs["contexts"] = contexts

    return inputs
    

def compute_predictions(start_logits, end_logits, offset_mapping, contexts, examples_ids, n_best=10, max_answer_length=30):
    """ Computes the predictions from the logits.
    """
    predictions = {}
    for i, example_id in enumerate(examples_ids):
        if example_id not in predictions:
            predictions[example_id] = []
        start_logit = start_logits[i]
        end_logit = end_logits[i]
        offset = offset_mapping[i]
        context = contexts[i]

        start_indexes = np.argsort(start_logit)[-1 : -n_best - 1 : -1].tolist()
        end_indexes = np.argsort(end_logit)[-1 : -n_best - 1 : -1].tolist()
        for start_index in start_indexes:
            for end_index in end_indexes:
                # Skip answers that are not fully in the context
                    if offset[start_index] is None or offset[end_index] is None:
                        continue
                    # Skip answers with a length that is either < 0 or > max_answer_length
                    if (
                        end_index < start_index
                        or end_index - start_index + 1 > max_answer_length
                    ):
                        continue

                    answer = {
                        "text": context[offset[start_index][0] : offset[end_index][1]], #The answer is the text between the start and end index
                        "logit_score": start_logit[start_index] + end_logit[end_index], #The score is the sum of the start and end logit
                        "no_answer_probability": (start_logit[0] + end_logit[0]) - (start_logit[start_index] + end_logit[end_index]), #The score of the null answer is the sum of the start and end logit for the first token
                        "question_id" : example_id.split("--")[0]
                    }
                    if answer["no_answer_probability"] < -2:
                        predictions[example_id].append(answer)
        
    for example_id in predictions:
        if len(predictions[example_id]) > 0:
            #Return the answer with the highest logit score
            predictions[example_id] = max(predictions[example_id], key=lambda x: x["logit_score"])
        else:
            predictions[example_id] = {
                "text": "No answer found", 
                "logit_score": -math.inf, 
                "no_answer_probability": 0,
                "question_id" : example_id.split("--")[0]
            }
        
    #In trivia_qa, 79% of contexts does not contain the answer. So we want to replace No answer found by the best one if any in such cases
    #Group the predictions by question id
    predictions_by_question_id = collections.defaultdict(list)
    for example_id, prediction in predictions.items():
        predictions_by_question_id[prediction["question_id"]].append(prediction)
                
    #For each question, select the answer with the highest logit score
    for question_id, preds in predictions_by_question_id.items():
        predictions_by_question_id[question_id] = max(preds, key=lambda x: x["logit_score"])
        
    #For each key in prediction, get the question id and replace the answer by the one in predicitons_by_question_id
    for key in list(predictions.keys()):
        if predictions[key]["text"] == "No answer found":
            predictions[key] = predictions_by_question_id[key.split("--")[0]]["text"]
        else:
            predictions[key] = predictions[key]["text"]
        
    return predictions


