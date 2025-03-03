# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utilities for training and loading Hugging Face tokenizers

import os
import logging

import torch
from datasets import IterableDataset
from transformers import AutoTokenizer

import isa_data
import ops
import proofs

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


# LOAD TOKENIZER

def load_latest_tokenizer(tokenizers_dir):
    latest_dir = ops.get_latest_dir_from(tokenizers_dir, adding_one=False)
    tokenizer = AutoTokenizer.from_pretrained(latest_dir)
    logging.info(f"Loaded Hugging Face tokenizer from {latest_dir} of type {type(tokenizer)}.")
    return tokenizer    

def get_trained_tokenizer(config_dict, making_dirs=False):
    tokenizers_dir, _, _ = ops.get_directory_paths(config_dict, making_dirs=making_dirs)
    previous_tok = ops.exists_previous("tokenizer", tokenizers_dir)
    if previous_tok:
        tokenizer = load_latest_tokenizer(tokenizers_dir)
    else:
        model_name = config_dict["model_name"]
        finetuning = True if config_dict["data_mode"].startswith("finetune") else False
        if finetuning:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            data_dir = config_dict["data_dir"]
            tokenizer = train_tokenizer(model_name, data_dir)
        ops.save_hf_data_in(tokenizer, tokenizers_dir)
    return tokenizer


# GENERATE DATASETS

def generate_proof_paths(json_data_dir, split=isa_data.SPLITS["NONE"]):
    if not proofs.valid_data_dir(json_data_dir):
        raise Exception(f"Error: bad input {json_data_dir} is not an existing directory or does not contain a proofN.json.")

    for subdir, _, files in os.walk(json_data_dir):
        json_files = [file for file in files if file.startswith("proof") and file.endswith(".json")]
        json_files = [os.path.join(subdir, file) for file in sorted(json_files)]

        if split == isa_data.SPLITS["NONE"]:
            yield from json_files
        else:
            total_proofs = len(json_files)
            train_size = int(total_proofs * 0.64)
            valid_size = int(total_proofs * 0.16)
            if split == isa_data.SPLITS["TRAIN"]:
                train_split = json_files[:train_size]
                yield from train_split
            elif split == isa_data.SPLITS["VALID"]:
                valid_split = json_files[train_size:train_size + valid_size]
                yield from valid_split
            elif split == isa_data.SPLITS["TEST"]:
                test_split = json_files[train_size + valid_size:]
                yield from test_split
            else:
                raise Exception(f"Error: bad input {split}, expected train, valid, or test.")

def accumulate_tokenized_lengths(lengths, proof, tokenizer, data_mode=isa_data.FORMATS["S"]):
    """
    Accumulator to be used with proofs.compute_stats. It computes the exact number of tokens 
    by using the tokenizer to split the proof's strings of inputs (and targets).

    :param lengths: pair of lists (for lengths of input-target pairs) to accumulate
    :param proof: dictionary abstracting a proof
    :param tokenizer: a Hugging Face tokenizer 
    :param data_mode: the data format
    :rtype: tuple(list)
    """
    x_y_pairs = proofs.inputs_targets_from(proof, data_mode=data_mode)
    lengths[0].extend(len(tokenizer(x)["input_ids"]) for x, _ in x_y_pairs)
    lengths[1].extend(len(tokenizer(y)["input_ids"]) for _, y in x_y_pairs)
    return lengths

def tokenize(tokenizer, x, y, finetuning=False):
    max_length = tokenizer.model_max_length
    x = "isabelle next step: " + x if finetuning else x
    inputs = tokenizer(
        x, 
        max_length=max_length, 
        truncation=True, 
        return_overflowing_tokens=True,
        stride=int(max_length * 0.1),
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
    labels = targets["input_ids"].squeeze()
    labels[labels == tokenizer.pad_token_id] = -100
    n_overflowed = inputs["input_ids"].shape[0]
    model_inputs = []
    for i in range(n_overflowed):
        to_add = {
            "input_ids": inputs["input_ids"][i],
            "attention_mask": inputs["attention_mask"][i],
            "overflow_sample_idx": inputs["overflow_to_sample_mapping"][i],
            "labels": labels
        }
        model_inputs.append(to_add)
    return model_inputs

def generate_model_inputs(tokenizer, json_data_dir, split, data_mode=isa_data.FORMATS["S"]):
    """
    Generator function for train, validation, and test data.

    :param tokenizer: the model's tokenizer
    :param json_data_dir: path to the search directory with proofN.json files
    :param split: 'train', 'valid', 'test', or 'none' to specify which split to load
    :param data_mode: proof data format (see isa_data.FORMATS)
    :returns: generator for tokenized inputs with labels for conditional generation
    :rtype: generator
    """
    tok_max_length = isa_data.get_context_length(data_mode)
    finetuning = True if config_dict["data_mode"].startswith("finetune") else False
    tokenizer.model_max_length = tok_max_length
    logging.info(f"Tokenizer's model max length is {tokenizer.model_max_length}")
    for path in generate_proof_paths(json_data_dir, split):
        proof = proofs.get_proof_json(path)
        for input_text, target_text in proofs.inputs_targets_from(proof, data_mode):
            model_inputs = tokenize(tokenizer, input_text, target_text, finetuning=finetuning)
            for model_input in model_inputs:
                yield model_input

# TODO: add support for HF datasets library
def get_dataset(tokenizer, config_dict, split=isa_data.SPLITS["NONE"]):
    data_mode = config_dict["data_mode"]
    dataset = IterableDataset.from_generator(
        generate_model_inputs, 
        gen_kwargs={
            'tokenizer': tokenizer, 
            'json_data_dir': config_dict["data_dir"], 
            'split': split, 
            'data_mode': data_mode
        }
    )
    logging.info(f"Loading dataset from the '{split}' split with data format '{data_mode}'.")
    return dataset

# SAVE AND LOAD DATASETS

def make_and_save_datasets(tokenizer, data_dir, datasets_dir, data_mode=isa_data.FORMATS["S"]):
    root_dirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir() and proofs.valid_data_dir(entry)]
    for root_dir in root_dirs:
        for split in [isa_data.SPLITS["TRAIN"], isa_data.SPLITS["VALID"], isa_data.SPLITS["TEST"]]:
            dataset = list(generate_model_inputs(tokenizer, root_dir, split=split, data_mode=data_mode))
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
        explanation = "Creates a tokenizer as specified in the input JSON configuration."
        config_dict = ops.get_json_dict(ops.parse_config_path(tool_explanation=explanation))
        ops.check_params(config_dict)
    except Exception as e:
        raise Exception(f"Error loading configuration information: {e}")
    
    tokenizer = get_trained_tokenizer(config_dict, making_dirs=True)