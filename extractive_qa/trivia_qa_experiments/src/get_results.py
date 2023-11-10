import os
import json
import pandas as pd

import utils.dataset_utils

from triviaqa_evaluation import evaluate_triviaqa


def generate_table():
    """ Generates a table with the results of the experiments.
    """
    full_results = pd.DataFrame(columns=['model', 'Validation-EM', 'Validation-F1'])
    verif_results = pd.DataFrame(columns=['model', 'Validation-EM', 'Validation-F1'])

    #Iterate over all subfolders of ../outputs
    for model in os.listdir("../outputs"):
        #Read the eval_dict.json file
        with open(os.path.join("../outputs", model, "eval_dict_full.json"), 'r') as f:
            eval_dict_full = json.load(f)
        with open(os.path.join("../outputs", model, "eval_dict_verified.json"), 'r') as f:
            eval_dict_verified = json.load(f)
        #Add a row to the dataframe
        full_results = pd.concat([full_results, pd.DataFrame({'model': model.split("-")[2], 'Validation-EM': eval_dict_full['exact_match'], 'Validation-F1': eval_dict_full['f1']}, index=[0])], ignore_index=True)
        verif_results = pd.concat([verif_results, pd.DataFrame({'model': model.split("-")[2], 'Validation-EM': eval_dict_verified['exact_match'], 'Validation-F1': eval_dict_verified['f1']}, index=[0])], ignore_index=True)
    
    #Sort the dataframe by Validation-F1
    full_results = full_results.sort_values(by=['Validation-F1'], ascending=False)
    verif_results = verif_results.sort_values(by=['Validation-F1'], ascending=False)
    #Round the values to 2 decimals
    full_results = full_results.round(2)
    verif_results = verif_results.round(2)
    full_results.to_latex(os.path.join("..", "outputs", "full_results.tex"), index=False, column_format="|r|r|r|", float_format=(lambda x: "%.3f" % x))
    verif_results.to_latex(os.path.join("..", "outputs", "verif_results.tex"), index=False, column_format="|r|r|r|", float_format=(lambda x: "%.3f" % x))


def compute_results(model_dir):
    """ Computes the results from predictions. (If needed with another json file)
    """
    with open(os.path.join("../outputs", model_dir, "predictions.json"), 'r') as f:
        predictions = json.load(f)

    file = utils.dataset_utils.read_triviaqa_data("web-dev.json")
    key_to_ground_truth = utils.dataset_utils.get_key_to_ground_truth(file)

    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)

    with open(os.path.join("../outputs", model_dir, "eval_dict_full.json"), 'w') as f:
        json.dump(eval_dict, f)
    
    file = utils.dataset_utils.read_triviaqa_data("verified-web-dev.json")
    key_to_ground_truth = utils.dataset_utils.get_key_to_ground_truth(file)

    eval_dict = evaluate_triviaqa(key_to_ground_truth, predictions)

    with open(os.path.join("../outputs", model_dir, "eval_dict_verified.json"), 'w') as f:
        json.dump(eval_dict, f)



if __name__ == "__main__":
    generate_table()
