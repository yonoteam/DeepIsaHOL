# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utilities for training and loading Hugging Face tokenizers

import os
import logging

import proofs

from transformers import AutoTokenizer


ACTION_SEP = 'ACTION_SEP'
GOAL_SEP = 'GOAL_SEP'
TERM_SEP = 'TERM_SEP'
HYPS_SEP = 'HYPS_SEP'
VARS_SEP = 'VARS_SEP'
CONSTS_SEP = 'CONSTS_SEP'
TYPES_SEP = 'TYPES_SEP'
APPLY_KWS_SEP = 'APPLY_KWS_SEP'
ISAR_KWS_SEP = 'ISAR_KWS_SEP'
DEPS_SEP = 'DEPS_SEP'
NAME_SEP = 'NAME_SEP'
METHODS_SEP = 'METHODS_SEP'

def separator(sep, txt):
    return sep + ' ' + txt

def string_from(proof_json):
    str_list = []
    for step in proof_json['proof']['steps'][1:]:
        usr_act_str = " ".join([
            step['step']['user_state'], 
            separator(ACTION_SEP, step['step']['action']), 
            separator(GOAL_SEP, step['step']['term'])
        ])
        str_list.append(usr_act_str)
        
        str_list.append(HYPS_SEP)
        for hyp_dict in step['step'].get('hyps', []):
            for _, hyp in hyp_dict.items():
                str_list.append(separator(TERM_SEP, hyp))

        str_list.append(VARS_SEP)
        for var_dict in step['step'].get('variables', []):
            for _, var in var_dict.items():
                str_list.append(separator(TERM_SEP, var))

        str_list.append(VARS_SEP)
        for var_dict in step['step'].get('variables', []):
            for _, var in var_dict.items():
                str_list.append(separator(TERM_SEP, var))

        str_list.append(CONSTS_SEP)
        for const_dict in step['step'].get('constants', []):
            for _, const in const_dict.items():
                str_list.append(separator(TERM_SEP, const))

        str_list.append(TYPES_SEP)
        for type_var_dict in step['step'].get('type variables', []):
            for _, type_var in type_var_dict.items():
                str_list.append(separator(TERM_SEP, type_var))

    str_list.append(APPLY_KWS_SEP)
    for apply_kw in proof_json['proof'].get('apply_kwrds', []):
        str_list.append(separator(NAME_SEP, apply_kw['name']))

    str_list.append(ISAR_KWS_SEP)
    for isar_kw in proof_json['proof'].get('isar_kwrds', []):
        str_list.append(separator(NAME_SEP, isar_kw['name']))

    str_list.append(DEPS_SEP)
    for dep in proof_json['proof'].get('deps', []):
        str_list.append(separator(NAME_SEP, dep['thm']['name']))
        str_list.append(separator(TERM_SEP, dep['thm']['term']))
        
    str_list.append(METHODS_SEP)
    for method in proof_json['proof'].get('methods', []):
        str_list.append(separator(NAME_SEP, method['name']))

    return " ".join(str_list)

def load_hf_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception:
        logging.error(f"'{model_name}' is not a valid local directory or Hugging Face model.")

def train_tokenizer(model_name, data_dir):
    try:
        tokenizer = load_hf_tokenizer(model_name)
        def get_training_corpus():
            imm_subdirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir()]
            for path in imm_subdirs:
                logging.info(f"Processing directory: {path}")
                for subdir, _, files in os.walk(path):
                    json_files = [file for file in files if file.startswith("proof") and file.endswith(".json")]
                    for file in json_files:
                        json_path = os.path.join(subdir, file)
                        proof_json = proofs.get_proof_json(json_path)
                        yield string_from(proof_json)
        
        training_corpus = get_training_corpus()
        tokenizer = tokenizer.train_new_from_iterator(training_corpus, 52000) # TODO: calculate optimal vocabulary size
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
            xs = xs + ' ' + separator(GOAL_SEP, proofs.orig_objective_of(proof_json))
            xs = xs + ' ' + separator(DEPS_SEP, ' ')
            for thm in proof_json['proof']['deps']:
                zs = separator(NAME_SEP, thm['thm']['name']) + ' ' + separator(TERM_SEP, thm['thm']['term'])
                xs = xs + ' ' + zs
        elif mode == 'state_prems_consts':
            xs = xs + ' ' + separator(GOAL_SEP, proofs.orig_objective_of(proof_json))
            xs = xs + ' ' + separator(DEPS_SEP, ' ')
            for thm in proof_json['proof']['deps']:
                zs = separator(NAME_SEP, thm['thm']['name']) + ' ' + separator(TERM_SEP, thm['thm']['term'])
                xs = xs + ' ' + zs
            xs = xs + ' ' + separator(CONSTS_SEP, ' ')
            for const in step['step']['constants']:
                for key in const.keys():
                    zs = separator(TERM_SEP, const[key])
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