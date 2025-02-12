# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utilities for training and loading Hugging Face tokenizers

import os
import logging
import torch

import ops
import proofs

from transformers import AutoTokenizer

TRAIN = "train"
VALID = "valid"
TEST = "test"
NONE = "none"

# TRAINING FROM SCRATCH

def train_tokenizer(model_name, data_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)      
        training_corpus = proofs.get_tokenizer_corpus(data_dir)
        estimated_vocab_size = proofs.estimate_vocab_size(data_dir, 0.98)
        vocab_size = int(1.5 * estimated_vocab_size)
        tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        return tokenizer
    except Exception as e:
        raise Exception(f"Error training tokenizer for {model_name} using data in {data_dir}: {e}")

def train_and_save_tokenizer(tokenizers_dir, data_dir, model_name):
    tokenizer = train_tokenizer(model_name, data_dir)
    ops.save_hf_data_in(tokenizer, tokenizers_dir)
    logging.info(f"Saved Hugging Face tokenizer.")
    return tokenizer


# LOAD TOKENIZER

def load_latest_tokenizer(tokenizers_dir):
    latest_dir = ops.get_latest_dir_from(tokenizers_dir, adding_one=False)
    tokenizer = AutoTokenizer.from_pretrained(latest_dir)
    logging.info(f"Loaded Hugging Face tokenizer from {latest_dir} of type {type(tokenizer)}.")
    return tokenizer    

def get_trained_tokenizer(config_dict, remote=False):
    data_dir = config_dict["data_dir"]
    model_name = config_dict["model_name"]
    tokenizers_dir, _, _ = ops.get_directory_paths(config_dict)
    if remote:
        tokenizer = train_and_save_tokenizer(tokenizers_dir, data_dir, model_name)
        return tokenizer
    else:
        tokenizer = load_latest_tokenizer(tokenizers_dir)
        return tokenizer


# GENERATE DATASETS

def generate_proof_paths(json_data_dir, split="none"):
    if not proofs.valid_data_dir(json_data_dir):
        raise Exception(f"Error: bad input {json_data_dir} is not an existing directory or does not contain a proofN.json.")

    for subdir, _, files in os.walk(json_data_dir):
        json_files = [file for file in files if file.startswith("proof") and file.endswith(".json")]
        json_files = [os.path.join(subdir, file) for file in sorted(json_files)]

        if split == NONE:
            yield from json_files
        else:
            total_proofs = len(json_files)
            train_size = int(total_proofs * 0.64)
            valid_size = int(total_proofs * 0.16)
            if split == TRAIN:
                train_split = json_files[:train_size]
                yield from train_split
            elif split == VALID:
                valid_split = json_files[train_size:train_size + valid_size]
                yield from valid_split
            elif split == TEST:
                test_split = json_files[train_size + valid_size:]
                yield from test_split
            else:
                raise Exception(f"Error: bad input {split}, expected train, valid, or test.")
        
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
    n_overflowed = inputs["input_ids"].shape[0]
    model_inputs = []
    for i in range(n_overflowed):
        to_add = {
            "input_ids": inputs["input_ids"][i],
            "attention_mask": inputs["attention_mask"][i],
            "overflow_sample_idx": inputs["overflow_to_sample_mapping"][i],
            "labels": targets["input_ids"].squeeze()
        }
        model_inputs.append(to_add)
    return model_inputs

def generate_model_inputs(tokenizer, json_data_dir, split, mode=proofs.STATE_MODE):
    """
    Generator function for train, validation, and test data.

    :param tokenizer: the model's tokenizer
    :param json_data_dir: path to the search directory with proofN.json files
    :param split: 'train', 'valid', 'test', or 'none' to specify which split to load
    :param mode: proof data format (see proofs.print_modes())
    :returns: generator for tokenized inputs with labels for conditional generation
    :rtype: generator
    """
    for path in generate_proof_paths(json_data_dir, split):
        proof = proofs.get_proof_json(path)
        for input_text, target_text in proofs.inputs_targets_from(proof, mode):
            model_inputs = tokenize(tokenizer, input_text, target_text)
            for model_input in model_inputs:
                yield model_input


# SAVE AND LOAD DATASETS

def make_and_save_datasets(tokenizer, data_dir, datasets_dir, mode=proofs.STATE_MODE):
    root_dirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir() and proofs.valid_data_dir(entry)]
    for root_dir in root_dirs:
        for split in [TRAIN, VALID, TEST]:
            dataset = list(generate_model_inputs(tokenizer, root_dir, split=split, mode=mode))
            save_name = os.path.join(os.path.basename(root_dir), split + '.pt')
            save_path = os.path.join(datasets_dir, save_name)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(dataset, save_path)

def generate_data_path_set(datasets_dir, split):
    datasets_paths = [os.path.join(entry.path, split + '.pt') for entry in os.scandir(datasets_dir) if entry.is_dir() and os.path.isfile(os.path.join(entry.path, split + '.pt'))]
    for dataset_path in datasets_paths:
        dataset = torch.load(dataset_path, weights_only=True)
        yield dataset_path, dataset


# MAIN

if __name__ == "__main__":
    ops.configure_logging("tokenizer.log")
    try:
        explanation = "Train a new tokenizer as specified in the input JSON configuration."
        config_dict = ops.get_config_dict(ops.parse_config_path(tool_explanation=explanation))
        data_dir, all_models_dir, model_name, mode, _ = ops.extract_params(config_dict)
        ops.check_params(data_dir, all_models_dir)
    except Exception as e:
        raise Exception(f"Error loading configuration information: {e}")
    
    tokenizers_dir, _, _ = ops.get_directory_paths(config_dict)
    _ = train_and_save_tokenizer(tokenizers_dir, data_dir, model_name)