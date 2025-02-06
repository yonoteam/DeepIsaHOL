# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utilities for training and loading Hugging Face tokenizers

import os
import logging
import torch
import proofs

from collections import defaultdict
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
        return tokenizer, tokenizers_dir
    else:
        subdirs = [
                os.path.join(tokenizers_dir, d) for d in os.listdir(tokenizers_dir)
                if os.path.isdir(os.path.join(tokenizers_dir, d)) and d.isdigit()
            ]
        latest_dir = max(subdirs, key=lambda d: int(os.path.basename(d)))
        tokenizer = AutoTokenizer.from_pretrained(latest_dir)
        return tokenizer, latest_dir
    
def tokenize(tokenizer, x, y):
    max_length = tokenizer.model_max_length
    inputs = tokenizer(
        x, 
        max_length=max_length, 
        truncation=True, 
        return_overflowing_tokens=True,            
        stride=10,
        padding="max_length",
        return_tensors="pt"
    )
    targets = tokenizer(
        y,
        max_length=max_length,
        truncation=True, 
        padding="max_length",
        return_tensors="pt"
    )
    overflow_mapping = inputs["overflow_to_sample_mapping"]
    return inputs, targets, overflow_mapping


def add_dir_data_to(tokenizer, json_data_dir, dataset_dict, mode=proofs.STATE_MODE):
    proof_paths = proofs.get_proofs_paths(json_data_dir)
    total_proofs = len(proof_paths)
    train_size = int(0.8 * total_proofs)
    valid_size = int(0.1 * total_proofs)

    for split in ["train", "valid", "test"]:
        if split not in dataset_dict:
            dataset_dict[split] = defaultdict(list)
    
    for i, proof_path in enumerate(proof_paths):
        proof_json = proofs.get_proof_json(proof_path)
        if i < train_size:
            data = dataset_dict["train"]
        elif i < train_size + valid_size:
            data = dataset_dict["valid"]
        else:
            data = dataset_dict["test"]
        
        for x, y in proofs.inputs_targets_from(proof_json, mode):
            inputs, targets, overflow_mapping = tokenize(tokenizer, x, y)
            for j, input_ids in enumerate(inputs["input_ids"]):
                data["input_ids"].append(input_ids)
                data["attention_mask"].append(inputs["attention_mask"][j])
                data["overflow_sample_idx"].append(overflow_mapping[j])
                data["labels"].append(targets["input_ids"].squeeze())
    
    return dataset_dict

def make_datasets(tokenizer, data_dir, mode=proofs.STATE_MODE):
    dataset_dict = {
        "train": defaultdict(list),
        "valid": defaultdict(list),
        "test": defaultdict(list)
    }

    imm_subdirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir()]
    for subdir in imm_subdirs:
        logging.info(f"Adding data from directory: {subdir}")
        dataset_dict = add_dir_data_to(tokenizer, subdir, dataset_dict, mode)
    
    for split in ["train", "valid", "test"]:
        dataset_dict[split] = {
            key: torch.stack(value) if key != "overflow_sample_idx" else torch.tensor(value)
            for key, value in dataset_dict[split].items()
        }
    return dataset_dict

def save_datasets_in(dataset_dict, datasets_dir):
    datasets_path = os.path.join(datasets_dir, 'datasets.pt')
    torch.save(dataset_dict, datasets_path)

def make_new_tokenizer_datasets_in(model_name, data_dir, mode, save_dir):
    tokenizer, latest_dir = get_trained_tokenizer(True, data_dir, save_dir, model_name)
    dataset_dict = make_datasets(tokenizer, data_dir, mode)
    save_datasets_in(dataset_dict, latest_dir)

def data_generator(dataset_path, split):
    """
    Generator function for train, validation, and test data.

    :param dataset_path: path to the saved dataset (datasets.pt)
    :param split: 'train', 'valid', or 'test' to specify which split to load
    :returns: generator for inputs with labels for conditional generation
    :rtype: generator
    """
    datasets = torch.load(dataset_path, map_location='cpu', weights_only=True)
    dataset = datasets[split]

    num_samples = dataset['input_ids'].shape[0] # attention_mask, overflow_sample_idx, and labels have the same first dimension
    for i in range(num_samples):
        yield {
            "input_ids": dataset['input_ids'][i],
            "attention_mask": dataset["attention_mask"][i],
            "overflow_sample_idx": dataset["overflow_sample_idx"][i],
            "labels": dataset["labels"][i]
        }

def data_generator_old(datasets_path, split):
    dataset = torch.load(datasets_path, weights_only=True)[split]
    for item in dataset:
        yield item

def add_data_from_old(mode_tok_data, proof_json):
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