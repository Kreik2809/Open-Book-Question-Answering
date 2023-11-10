import torch
import os
import logging
import datetime
import argparse
import json

import pandas as pd
import numpy as np

from tqdm import tqdm

from generative_utils import load_model, load_context, process_outputs

def inference(model, tokenizer, question, context, no_ans_threshold, ans_threshold, max_length=4096, stride=128, device="cuda", max_answer_length=64):
    """ This function performs inference on a given question and context.
    """
    inputs = tokenizer.encode(context, question, max_length=max_length, truncation="only_first", stride=stride, padding=False, return_overflowing_tokens=True)
    answers = []

    for input in tqdm(inputs):
        input_ids = torch.tensor(input, device=device).unsqueeze(0)

        #generate answer
        with torch.no_grad():
            outputs = model.generate(input_ids, 
                                        do_sample=False,
                                        num_beams=20, 
                                        max_new_tokens=max_answer_length, 
                                        num_return_sequences=10, 
                                        eos_token_id=tokenizer.eos_token_id, 
                                        pad_token_id=tokenizer.pad_token_id, 
                                        return_dict_in_generate=True, 
                                        output_scores=True
                                )
            #Process outputs
            selected_sentence, sequence_score, no_answer_logit = process_outputs(outputs, input_ids, tokenizer, model, device)
            selected_sentence = selected_sentence.squeeze(0)
            
            answers.append((selected_sentence[input_ids.shape[1]:], sequence_score, no_answer_logit))
            del outputs
    
    for i, (answer, score, no_answer_logit) in enumerate(answers):
        answers[i] = (tokenizer.decode(answer.squeeze(0), skip_special_tokens=True), score, no_answer_logit)
    answers = [answer for answer in answers if (answer[2][0]) < no_ans_threshold and answer[1][0] > ans_threshold]
    answers.sort(key=lambda x: x[1][0], reverse=True)
    if len(answers) == 0:
        return {"text": "No answer found", "logit_score": "0", "no_answer_score": "0"}
    return {"text": answers[0][0], "logit_score": answers[0][1][0], "no_answer_score": answers[0][2][0]}

if __name__ == "__main__":
    #Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='../configs/config.json')
    args = parser.parse_args()
    config = json.load(open(args.config, 'r'))  

    #Gather all hyperparameters
    experiment_name = config['experiment_name']
    model_id = config['model_id']
    tokenizer_id = config['tokenizer_id']
    device = config['device']
    no_ans_threshold = config['no_ans_threshold']
    ans_threshold = config['ans_threshold']
    stride = config['stride']
    Qlora = config['Qlora']
    max_length = config['max_length']
    max_answer_length = config['max_answer_length']
    seed = config['seed']

    #Set seed and create output directory
    torch.manual_seed(seed)
    np.random.seed(seed)
    output_dir = "../outputs/" + experiment_name
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        for file in os.listdir(output_dir):
            os.remove(output_dir + "/" + file)
    
    #Set up logging
    log_file = os.path.join(output_dir, "inference_log_" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + ".log")
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)    
    fh = logging.FileHandler(log_file)
    fh.setLevel(logging.INFO) 
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)

    logger.info("Starting evaluation for experiment: " + experiment_name)
    logger.info("Model: " + model_id)
    logger.info("Tokenizer: " + tokenizer_id)
    logger.info("Device: " + device)
    logger.info("No answer threshold: " + str(no_ans_threshold))
    logger.info("Answer threshold: " + str(ans_threshold))
    logger.info("Qlora: " + str(Qlora))
    logger.info("Stride: " + str(stride))
    logger.info("Max length: " + str(max_length))
    logger.info("Max answer length: " + str(max_answer_length))
    logger.info("Seed: " + str(seed))

    #Load contexts
    sinch_node_red = load_context("mmd/sinch_doc_node_red.mmd")
    sinch_webhook = load_context("mmd/sinch_doc_how_to_webhook.mmd")
    sinch_overview = load_context("mmd/sinch_doc_overview.mmd")
    nougat_context = load_context("mmd/nougat.mmd")

    #Load model and hyperparameters
    model, tokenizer = load_model(model_id, Qlora=Qlora,  device=device)

    no_ans_threshold = no_ans_threshold
    ans_threshold = ans_threshold
    max_length = max_length

    #Inference
    df = pd.DataFrame(columns=['question', 'answer', 'logit_score', 'no_answer_probability'])

    question = "What is Node RED ? "
    answer = inference(model, tokenizer, question, sinch_node_red, no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    """question = "In few words, What is Node RED ? "
    answer = inference(model, tokenizer, question, sinch_node_red,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "What are the supported channels of Node RED ? "
    answer = inference(model, tokenizer, question, sinch_node_red,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "In which cases can I use Node RED ? "
    answer = inference(model, tokenizer, question, sinch_node_red,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "What are the differents nodes of Sinch Messaging ? "
    answer = inference(model, tokenizer, question, sinch_node_red,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "When was Node RED released ? "
    answer = inference(model, tokenizer, question, sinch_node_red,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "Give me the different steps to add a webhook to my app ? "
    answer = inference(model, tokenizer, question, sinch_webhook,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "What is the Sinch Conversation API ?"
    answer = inference(model, tokenizer, question, sinch_overview,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "Can I use the Sinch Conversation API with Viber Business ? "
    answer = inference(model, tokenizer, question, sinch_overview,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "Can I use the Sinch Conversation API with Outlook ? "
    answer = inference(model, tokenizer, question, sinch_overview,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "Where are the hosting locations for the Conversation API ? "
    answer = inference(model, tokenizer, question, sinch_overview,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "What are the specific pricing details for using the Sinch Conversation API ? "
    answer = inference(model, tokenizer, question, sinch_overview,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    

    question = "How does the Sinch Conversation API handle multimedia content like images and videos ? "
    answer = inference(model, tokenizer, question, sinch_overview,  no_ans_threshold=no_ans_threshold, ans_threshold=ans_threshold, device=device, stride=stride, max_length=max_length, max_answer_length=max_answer_length)
    logger.info(question + " " + str(answer))
    df = pd.concat([df, pd.DataFrame([[question, answer["text"], answer["logit_score"], answer["no_answer_score"]]], columns=['question', 'answer', 'logit_score', 'no_answer_probability'])])    
"""
    #Save results
    df = df.drop(['logit_score', 'no_answer_probability'], axis=1)
    latex_code = df.to_latex(index=False, column_format="|p{5cm}|p{10cm}|", float_format=(lambda x: "%.3f" % x))
    latex_code = latex_code.replace('\\toprule', '\hline')
    latex_code = latex_code.replace('\\bottomrule', '\hline')
    latex_code = latex_code.replace('\\\n', '\\ \hline\n')

    latex_file_name = "/generative_qa_mistral_dpo.tex"
    with open(output_dir + latex_file_name, 'w') as file:
        file.write(latex_code)