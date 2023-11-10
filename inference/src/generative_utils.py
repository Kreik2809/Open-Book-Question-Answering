import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import AutoPeftModelForCausalLM

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

    del outputs
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

    seq1_in_seq2 = []
    for word in sequence1.split():
        seq1_in_seq2.append(False)
        for word2 in sequence2.split():
            if word == word2:
                seq1_in_seq2[-1] = True
                break

    seq2_in_seq1 = []
    for word in sequence2.split():
        seq2_in_seq1.append(False)
        for word2 in sequence1.split():
            if word == word2:
                seq2_in_seq1[-1] = True
                break
    
    return all(seq1_in_seq2) or all(seq2_in_seq1)

def process_outputs(outputs, batch, tokenizer, model, device): 
    """ This function process the outputs of the generate function to extract the generated sequence, the score of the sequence and the score of the no answer sequence.
    """
    no_answer_token_ids = tokenizer.no_answer_token_ids
    selected_sequence = outputs.sequences[0]
    
    if is_similar(selected_sequence[batch.shape[1]:], no_answer_token_ids, tokenizer):
        for i, sequence in enumerate(outputs.sequences):
            if not is_similar(sequence[batch.shape[1]:], no_answer_token_ids, tokenizer):
                selected_sequence = sequence
                break 
         
    context = selected_sequence[:batch.shape[1]]
    no_answer_sequence = torch.cat([context, torch.tensor(no_answer_token_ids, device=device)])

    selected_sequence_score = to_tokens_and_logprobs(model, tokenizer, selected_sequence.unsqueeze(0))
    no_answer_score = to_tokens_and_logprobs(model, tokenizer, no_answer_sequence.unsqueeze(0))
    
    for i, (token, p) in enumerate(selected_sequence_score[0][batch.shape[1] -1:]):
        if token == tokenizer.eos_token or i == len(selected_sequence_score[0][batch.shape[1] -1:]) - 1:
            selected_sequence_score = [selected_sequence_score[0][batch.shape[1] -1:batch.shape[1] + i]]
            break
    no_answer_score = [no_answer_score[0][batch.shape[1] -1:]] #Scores are shifted by one

    sentence_score = [sum([p for _, p in selected_sequence_score[0]]) / len(selected_sequence_score[0])]
    no_answer_score = [sum([p for _, p in no_answer_score[0]]) / len(no_answer_score[0])]

    selected_sequence = selected_sequence.unsqueeze(0)

    return selected_sequence, sentence_score, no_answer_score

def load_context(file_path):
    """ This function loads the context from a file.
    """
    with open(file_path, 'r') as file:
        filedata = file.read()

        context = filedata.split('\n')
        context = ''.join(context)
    return context

def load_model(model_dir, Qlora="False", device="cuda"):
    """ This function loads the model and the tokenizer.
    """
    if Qlora == "True":
        model = AutoPeftModelForCausalLM.from_pretrained(model_dir, device_map="auto", torch_dtype=torch.bfloat16)
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = model.merge_and_unload()
    
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_dir)
        model = AutoModelForCausalLM.from_pretrained(model_dir).to(device)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.bos_token
    if model.config.pad_token_id is None:
        model.config.pad_token_id = model.config.bos_token_id

    no_answer_label = "The answer is not in the context."
    tokenized_no_answer = tokenizer(no_answer_label)["input_ids"]
    tokenizer.no_answer_token_ids = tokenized_no_answer + [tokenizer.eos_token_id]

    return model, tokenizer