import matplotlib.pyplot as plt
import seaborn as sns
import json
import os
import warnings

import numpy as np
import pandas as pd

def make_plots(dir, examples):
    """ This function plot the histogram of the scores of the predictions.
    """
    with open(os.path.join(dir, "predictions.json"), 'r') as f:
        predictions = json.load(f)
    
    logit_score_no_answer = []
    logit_score_answer = []

    no_answer_prob_no_answer = []
    no_answer_prob_answer = []

    for i, example_id in enumerate(predictions.keys()):
        if len(examples[i]["answers"]["answer_start"]) == 0:
            logit_score_no_answer.append(predictions[example_id][0]["logit_score"])
            no_answer_prob_no_answer.append(predictions[example_id][0]["no_answer_probability"])
        else:
            logit_score_answer.append(predictions[example_id][0]["logit_score"])
            no_answer_prob_answer.append(predictions[example_id][0]["no_answer_probability"])
    
    logit_score_no_answer = np.exp(logit_score_no_answer)
    logit_score_answer = np.exp(logit_score_answer)

    logit_score_all = np.concatenate([logit_score_no_answer, logit_score_answer])

    warnings.filterwarnings("ignore")

    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.distplot(logit_score_all, ax=ax, color="blue", label="all", kde=True).set(title="Model output probability distribution")
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(os.path.join(dir, "logit_score_distribution.png"))

    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.distplot(logit_score_answer, ax=ax, color="blue", label="answer", kde=True).set(title="Model output answer score distribution")
    sns.distplot(logit_score_no_answer, ax=ax, color="red", label="no answer", kde=True)
    ax.set_xlabel("Probability")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(os.path.join(dir, "logit_score_distribution_split.png"))

    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.distplot(no_answer_prob_no_answer, ax=ax, color="red", label="no answer", kde=True).set(title="Model output no answer score distribution")
    sns.distplot(no_answer_prob_answer, ax=ax, color="blue", label="answer", kde=True)
    ax.set_xlabel("No answer score")
    ax.set_ylabel("Density")
    ax.legend()
    plt.savefig(os.path.join(dir, "no_answer_prob_distribution_split.png"))


def make_table(dir):
    """This function gnerates the tables for the results related to squad and squad 2
    """

    squad_results = pd.DataFrame(columns=["Model", "Validation-EM", "Validation-F1"])
    squad2_results = pd.DataFrame(columns=["Model", "Best Val-EM", "Best Val-F1", "HasAns-EM", "HasAns-F1"])


    for folder in os.listdir(dir):
        if folder.startswith("causal"):
            dataset_name = folder.split("_")[1]
            model_name = folder.split("_")[2]
            setup = folder.split("_")[3:]
            setup = "_".join(setup)
            if dataset_name == "squad":
                if os.path.exists(os.path.join(dir, folder, "metrics.json")):
                    with open(os.path.join(dir, folder, "metrics.json"), 'r') as f:
                        metrics = json.load(f)
                    squad_results = pd.concat([squad_results, pd.DataFrame({"Model": model_name + "\_" + setup, "Validation-EM": metrics["exact"], "Validation-F1": metrics["f1"]}, index=[0])], ignore_index=True)
            elif dataset_name == "squad2":
                if os.path.exists(os.path.join(dir, folder, "metrics.json")):
                    with open(os.path.join(dir, folder, "metrics.json"), 'r') as f:
                        metrics = json.load(f)
                    squad2_results = pd.concat([squad2_results, pd.DataFrame({"Model": model_name + "\_" + setup, "Best Val-EM": metrics["best_exact"], "Best Val-F1": metrics["best_f1"], "HasAns-EM": metrics["HasAns_exact"], "HasAns-F1": metrics["HasAns_f1"]}, index=[0])], ignore_index=True)
    #Sort and save the tables
    squad_results = squad_results.sort_values(by=['Validation-F1'], ascending=False)
    squad2_results = squad2_results.sort_values(by=['Best Val-F1'], ascending=False)
    squad_results = squad_results.round(2)
    squad2_results = squad2_results.round(2)
    squad_results.to_latex(os.path.join("..", "outputs", "squad_results.tex"), index=False, column_format="|l|r|r|", float_format=(lambda x: "%.3f" % x))
    squad2_results.to_latex(os.path.join("..", "outputs", "squad2_results.tex"), index=False, column_format="|l|r|r|r|r|", float_format=(lambda x: "%.3f" % x))
   

def make_learning_curves(dir):
    """This function plots the learning curves of the training
    """
    with open(os.path.join(dir, "trainer_state.json"), 'r') as f:
        trainer_state = json.load(f)

    history = trainer_state["log_history"]

    train_loss = [(step["step"], step["loss"]) for step in history if ("loss" in step.keys() and "step" in step.keys())]
    val_loss = [(step["step"], step["eval_loss"]) for step in history if ("eval_loss" in step.keys() and "step" in step.keys())]

    sns.set(style="darkgrid")
    sns.set(font_scale=1.5)
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.lineplot(x=[x[0] for x in train_loss], y=[x[1] for x in train_loss], ax=ax, color="blue", label="train")
    sns.lineplot(x=[x[0] for x in val_loss], y=[x[1] for x in val_loss], ax=ax, color="red", label="val").set(title="Learning curves")
    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.legend()
    plt.savefig(os.path.join(dir, "learning_curves.png"))

