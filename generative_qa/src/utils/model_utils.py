import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import get_peft_model, prepare_model_for_kbit_training, AutoPeftModelForCausalLM
from utils.qlora_utils import load_model, create_bnb_config, create_peft_config, find_all_linear_names, print_trainable_parameters


def load_model_for_training(model_id, tokenizer_id, output_dir, Qlora, device):
    """ This function is used to load the model and the tokenizer for the training.
    It allows to load the model from a local checkpoint or from huggingface while using Qlora or not.
    """
    if Qlora == "True":
        if not os.path.isdir(model_id):
            #Load model and tokenizer from huggingface
            bnb_config = create_bnb_config()
            model, tokenizer = load_model(model_id, bnb_config)
            model.gradient_checkpointing_enable()
            model = prepare_model_for_kbit_training(model)
            modules = find_all_linear_names(model)
            peft_config = create_peft_config(modules)
            model = get_peft_model(model, peft_config)
            print_trainable_parameters(model)

        else:
            #Load model and tokenizer from local checkpoint
            model = AutoPeftModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
    else:
        #Same script for either huggingface or local checkpoint
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token   
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.bos_token_id
    model.save_pretrained(os.path.join(output_dir, "initial_checkpoint"))
    tokenizer.save_pretrained(os.path.join(output_dir, "initial_checkpoint"))
    return model, tokenizer

def load_model_for_inference(model_id, tokenizer_id, output_dir, Qlora, device, logger):
    """ This function is used to load the model and the tokenizer for the inference.
    It allows to load the model from a local checkpoint or from huggingface while using Qlora or not.
    """
    if Qlora == "True":
        if os.path.isdir(os.path.join(output_dir, "last_checkpoint")):
            #Load model from local checkpoint
            logger.info("Loading model from local last training checkpoint")
            model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(output_dir, "last_checkpoint"), device_map="auto", torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "last_checkpoint"))
        elif os.path.isdir(os.path.join(model_id)):
            #Load model from local checkpoint
            logger.info("Loading model from local checkpoint")
            model = AutoPeftModelForCausalLM.from_pretrained(os.path.join(model_id), device_map="auto", torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(tokenizer_id))
        else:
            #load model from huggingface
            logger.info("Loading model from HF checkpoint")
            model = AutoPeftModelForCausalLM.from_pretrained(model_id, device_map="auto", torch_dtype=torch.bfloat16)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
        model = model.merge_and_unload()
    else:
        if os.path.isdir(os.path.join(output_dir, "last_checkpoint")):
            #Load model from local checkpoint which corresponds to the last training checkpoint
            logger.info("Loading model from local last training checkpoint")
            model = AutoModelForCausalLM.from_pretrained(os.path.join(output_dir, "last_checkpoint")).to(device)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(output_dir, "last_checkpoint"))
        elif os.path.isdir(os.path.join(model_id)):
            #Load model from local checkpoint which corresponds to the last training checkpoint
            logger.info("Loading model from local checkpoint")
            model = AutoModelForCausalLM.from_pretrained(os.path.join(model_id)).to(device)
            tokenizer = AutoTokenizer.from_pretrained(os.path.join(tokenizer_id))
        else:
            #load model from huggingface
            logger.info("Loading model from HF checkpoint")
            model = AutoModelForCausalLM.from_pretrained(model_id).to(device)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)
            
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token   
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.bos_token_id
    return model, tokenizer 
