# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utilities for training and loading Hugging Face tokenizers

import logging

from transformers import AutoTokenizer

import json
import dicts
import config_ops
import save_ops

import proofs
from proofs.str_ops import FORMATS
from proofs.data_dir import SPLITS

# CONFIGURATION

T5_CTX_LENGTHS = {
    FORMATS["S"]: 1024,
    FORMATS["SP"]: 2048,
    FORMATS["SPK"]: 4096,
    FORMATS["SPKT"]: 8192,
    FORMATS["FS"]: 1024,
    FORMATS["FSP"]: 2048,
    FORMATS["FSPK"]: 4096,
    FORMATS["FSPKT"]: 8192
}

def get_t5_context_length(mode):
    return T5_CTX_LENGTHS.get(mode)

GEMMA_CTX_LENGTHS = {
    FORMATS["S"]: 1024,
    FORMATS["SP"]: 4096,
    FORMATS["SPK"]: 4096,
    FORMATS["SPKT"]: 8192
}

def get_gemma_context_length(mode):
    return GEMMA_CTX_LENGTHS.get(mode)

# TRAINING FROM SCRATCH

def get_tokenizer_corpus(json_data_dir):
    for proof in proofs.data_dir.generate_from(json_data_dir):
        yield json.dumps(proof)

def train_tokenizer(model_name, data_dir):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        training_corpus = get_tokenizer_corpus(data_dir)
        estimated_vocab_size = proofs.data_stats.estimate_vocab_size(data_dir, 0.98)
        vocab_size = int(1.5 * estimated_vocab_size)
        tokenizer = tokenizer.train_new_from_iterator(training_corpus, vocab_size)
        return tokenizer
    except Exception as e:
        raise Exception(f"Error training tokenizer for {model_name} using data in {data_dir}: {e}")


# LOAD TOKENIZER

def load_latest_tokenizer(tokenizers_dir):
    latest_dir = save_ops.get_latest_dir(tokenizers_dir, adding_one=False)
    tokenizer = AutoTokenizer.from_pretrained(latest_dir)
    logging.info(f"Loaded Hugging Face tokenizer from {latest_dir} of type {type(tokenizer)}.")
    return tokenizer

def get_trained_tokenizer(config_dict, making_dirs=False):
    tokenizers_dir, _ = save_ops.get_dirs(config_dict, making_dirs=making_dirs)
    previous_tok = save_ops.exists_previous("tokenizer", tokenizers_dir)
    if previous_tok:
        tokenizer = load_latest_tokenizer(tokenizers_dir)
    else:
        model_name = config_dict["model_name"]
        task = config_dict["task"]
        if task == config_ops.save_hf_tokenizer:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        elif task == config_ops.train_hf_tokenizer:
            data_dir = config_dict["data_dir"]
            tokenizer = train_tokenizer(model_name, data_dir)
        else:
            raise ValueError(
                f"Undefined task '{task}' for retrieving a tokenizer."
                f"Expected one of: {list(config_ops.save_hf_tokenizer, config_ops.train_hf_tokenizer)}"
            )
        save_ops.save_in(tokenizer, tokenizers_dir)
    return tokenizer


# GENERATE DATASETS

def accumulate_tokenized_lengths(lengths, proof, tokenizer, data_format=FORMATS["S"]):
    """
    Accumulator to be used with proofs.compute_stats. It computes the exact number of tokens 
    by using the tokenizer to split the proof's strings of inputs (and targets).

    :param lengths: pair of lists (for lengths of input-target pairs) to accumulate
    :param proof: dictionary abstracting a proof
    :param tokenizer: a Hugging Face tokenizer 
    :param data_format: the data format
    :rtype: tuple(list)
    """
    x_y_pairs = proofs.str_ops.inputs_targets_from(proof, data_format=data_format)
    lengths[0].extend(len(tokenizer(x)["input_ids"]) for x, _ in x_y_pairs)
    lengths[1].extend(len(tokenizer(y)["input_ids"]) for _, y in x_y_pairs)
    return lengths

def t5_tokenize(tokenizer, x, y):
    max_length = tokenizer.model_max_length
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

def generate_model_inputs(json_data_dir, split, data_format=FORMATS["S"]):
    """
    Generator function for train, validation, and test data.

    :param tokenizer: the model's tokenizer
    :param json_data_dir: path to the search directory with proofN.json files
    :param split: 'train', 'valid', 'test', or 'none' to specify which split to load
    :param data_format: proof data format (see proofs.str_ops.FORMATS)
    :returns: generator for tokenized inputs with labels for conditional generation
    :rtype: generator
    """
    for path in proofs.data_dir.generate_dataset_paths(json_data_dir, split):
        proof = dicts.load_json(path)
        for input_text, target_text in proofs.str_ops.inputs_targets_from(proof, data_format):
            x = "isabelle next step: " + input_text if "finetune" in data_format else input_text
            model_inputs = {"input": x, "target": target_text}
            yield model_inputs

def t5_tokked_model_inputs(tokenizer, json_data_dir, split=SPLITS["NONE"], data_format=FORMATS["S"]):
    tok_max_length = get_t5_context_length(data_format)
    tokenizer.model_max_length = tok_max_length
    logging.info(f"Tokenizer's model max length is {tokenizer.model_max_length}")
    for model_inputs in generate_model_inputs(json_data_dir, split, data_format=data_format):
        tokked_model_inputs = t5_tokenize(tokenizer, model_inputs["input"], model_inputs["target"])
        for model_input in tokked_model_inputs:
            yield model_input

llm_prompt = """Recommend the next Isabelle proof step given the context below. Enclose the suggestion in <SUGGESTION>-</SUGGESTION> tags. Only return the suggestion, do not include any other text.
{context}
"""

def to_gemma_format(input_text, target_text):
    return {
        "messages": [{
            "role": "user", 
            "content": [
                {"type": "text", "text": llm_prompt.format(context=input_text)}
            ]
        }, 
        {
            "role": "assistant", 
            "content": [
                {"type": "text", "text": f"<SUGGESTION>{target_text}</SUGGESTION>"}
            ]
        }]
    }

def generate_gemma_inputs(json_data_dir, split, data_format):
    for path in proofs.data_dir.generate_dataset_paths(json_data_dir, split):
        proof = dicts.load_json(path)
        for input_text, target_text in proofs.str_ops.inputs_targets_from(proof, data_format):
            yield to_gemma_format(input_text, target_text)

# MAIN

if __name__ == "__main__":
    explanation = "Creates a tokenizer as specified in the input JSON configuration."
    config_dict = config_ops.parse_path(explanation)
    config_ops.setup_logging("tokenizer.log")
    tokenizer = get_trained_tokenizer(config_dict, making_dirs=True)