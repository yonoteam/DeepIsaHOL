# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utilities for training and loading Hugging Face tokenizers

import os
import logging
import proofs
from transformers import AutoTokenizer

def load_hf_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception:
        logging.error(f"'{model_name}' is not a valid local directory or Hugging Face model.")

def train_tokenizer(model_name, data_dir):
    try:
        tokenizer = load_hf_tokenizer(model_name)        
        training_corpus = proofs.get_tokenizer_corpus(data_dir)
        estimated_vocab_size = proofs.estimate_vocab_size(data_dir, 0.98)
        vocab_size = int(1.5 * estimated_vocab_size)
        tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        return tokenizer
    except Exception as e:
        logging.error(f"training tokenizer for {model_name} using data in {data_dir}: {e}")

# works for both models and tokenizers
def save_hf_data_in(hf_data, saving_dir):
    # Determine the latest saving number
    subdirs = [
        os.path.join(saving_dir, d) for d in os.listdir(saving_dir)
        if os.path.isdir(os.path.join(saving_dir, d)) and d.isdigit()
    ]
    if subdirs:
        latest_number = max(int(os.path.basename(d)) for d in subdirs)
    else:
        latest_number = 0

    # Save the hugging face model or tokenizer
    new_dir = os.path.join(saving_dir, str(latest_number + 1))

    try:
        os.makedirs(new_dir, exist_ok=False)
        hf_data.save_pretrained(new_dir)
        logging.info(f"Saved Hugging Face data in {new_dir}")
    except FileExistsError:
        logging.warning(f"Directory '{new_dir}' already exists. Skipping save.")
    
def get_trained_tokenizer(remote, data_dir, tokenizers_dir, model_name):
    if remote:
        tokenizer = train_tokenizer(model_name, data_dir)
        save_hf_data_in(tokenizer, tokenizers_dir)
        return tokenizer
    else:
        subdirs = [
                os.path.join(tokenizers_dir, d) for d in os.listdir(tokenizers_dir)
                if os.path.isdir(os.path.join(tokenizers_dir, d)) and d.isdigit()
            ]
        latest_dir = max(subdirs, key=lambda d: int(os.path.basename(d)))
        tokenizer = AutoTokenizer.from_pretrained(latest_dir)
        return tokenizer, latest_dir
    
def add_data_from(mode_tok_data, proof_json):
    mode, tokenizer, data = mode_tok_data
    for step in proof_json['proof']['steps'][1:]:
        y = step['step']['action']
        xs = step['step']['user_state']
        if mode == 'state_prems':
            xs = xs + ' ' + ' '.join([proofs.GOAL_SEP, proofs.orig_objective_of(proof_json)])
            xs = xs + ' ' + ' '.join([proofs.DEPS_SEP, ' '])
            for thm in proof_json['proof']['deps']:
                zs = ' '.join([proofs.NAME_SEP, thm['thm']['name']]) + ' ' + ' '.join([proofs.TERM_SEP, thm['thm']['term']])
                xs = xs + ' ' + zs
        elif mode == 'state_prems_consts':
            xs = xs + ' ' + ' '.join([proofs.GOAL_SEP, proofs.orig_objective_of(proof_json)])
            xs = xs + ' ' + ' '.join([proofs.DEPS_SEP, ' '])
            for thm in proof_json['proof']['deps']:
                zs = ' '.join([proofs.NAME_SEP, thm['thm']['name']]) + ' ' + ' '.join([proofs.TERM_SEP, thm['thm']['term']])
                xs = xs + ' ' + zs
            xs = xs + ' ' + ' '.join([proofs.CONSTS_SEP, ' '])
            for const in step['step']['constants']:
                for key in const.keys():
                    zs = ' '.join([proofs.TERM_SEP, const[key]])
                    xs = xs + ' ' + zs

        inputs = tokenizer(
            xs, 
            max_length=512,
            truncation=True, 
            return_overflowing_tokens=True,            
            stride=10,
            padding="max_length", # TODO: delete? so that data_collator handles the padding efficiently?
            return_tensors="pt"
        )
        targets = tokenizer(
            y, 
            max_length=512, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        overflow_mapping = inputs["overflow_to_sample_mapping"]
        for i, input_ids in enumerate(inputs["input_ids"]):
            to_add = {
                "input_ids": input_ids,  # No need for squeeze here
                "attention_mask": inputs["attention_mask"][i],
                "overflow_sample_idx": overflow_mapping[i], # tells origin sample of this to_add
                "labels": targets["input_ids"].squeeze()
            }
            data.append(to_add)
    return data