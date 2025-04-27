# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for reading JSONs of proofs

import os
import json
import logging
import statistics

from pathlib import Path
from collections import Counter
from copy import deepcopy
from tqdm import tqdm

import isa_data


# DIRECTORY OPERATIONS

def get_proof_json(proof_path):
    """Extract proof data from the input file. 
    Returns {} if failed.
    
    :param proof_path: path to the (proof) JSON file
    :returns: a dictionary abstracting the proof
    :rtype: dict
    """
    try:
        with open(proof_path, 'r') as json_file:
            proof_data = json.load(json_file)
            return proof_data
    except Exception as e:
        logging.error(f"Failed to process {proof_path}: {str(e)}")
    return {}

def valid_data_dir(json_data_dir):
    """
    Checks if input is a directory with at least one JSON 
    file starting with 'proof' and ending with '.json' in 
    a subdirectory .

    :param json_data_dir: path to the directory to validate
    :returns: True if a valid JSON file is found, False otherwise.
    :rtype: bool
    """
    data_path = Path(json_data_dir)
    if not data_path.exists() or not data_path.is_dir():
        return False

    for json_file in data_path.rglob("proof*.json"):
        if json_file.is_file():
            return True
    return False

def generate_paths_from(json_data_dir):
    """
    Generator for all "proofN.json" file-paths in the input directory.

    :param json_data_dir: path to the data directory with 'proofN.json's
    :returns: generator of all "proofN.json" files
    :rtype: generator
    """
    for subdir, _, files in os.walk(json_data_dir):
        for file in sorted(files):
            if file.startswith("proof") and file.endswith(".json"):
                yield os.path.join(subdir, file)

def generate_from(json_data_dir):
    """
    Generator for all "proofN.json" files in the input directory.

    :param json_data_dir: path to the data directory with 'proofN.json's
    :returns: generator of all "proofN.json" files
    :rtype: generator
    """
    for proof_path in generate_paths_from(json_data_dir):
        proof = get_proof_json(proof_path)
        yield proof

def get_proofs_paths(json_data_dir):
    """
    Finds all paths "proofN.json" in the input directory.

    :param json_data_dir: path to the data directory with 'proofN.json's
    :returns: sorted list of paths to JSON files
    :rtype: list(str)
    """
    return sorted(list(generate_paths_from(json_data_dir)))

def find_erroneous(json_data_dir, f=lambda x: x):
    """
    Attempts to load each "proofN.json" in the input directory
    and apply `f` to it. If it fails, it stores the failing
    file path in the returned list.

    :param json_data_dir: path to the data directory with 'proofN.json's
    :param f: function to apply to each JSON file
    :returns: list of paths to files that failed to load or process
    :rtype: list(str)
    """
    erroneous_paths = []
    for file_path in generate_paths_from(json_data_dir):
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
            f(data)
        except Exception as e:
            erroneous_paths.append(file_path)
    return erroneous_paths

def delete_erroneous(json_data_dir):
    """
    Deletes all erroneous "proofN.json" in the input directory.

    :param json_data_dir: path to the data directory with 'proofN.json's
    :returns: tuple of lists, the first representing the 
        deleted files and the second, those that failed to 
        be deleted with their respective exceptions
    :rtype: tuple(list(str), list(tuple(str, Exception)))
    """
    deleted_files = []
    failed_files = []
    for file_path in find_erroneous(json_data_dir):
        try:
            os.remove(file_path)
            deleted_files.append(file_path)
        except Exception as e:
            path_reason = file_path, e
            failed_files.append(path_reason)
    return deleted_files, failed_files

def apply(f, inits, json_data_dir):
    """For accumulating the result of applying f to 
    each proof in the input directory starting from 
    initial store "inits".

    :param f: function of type (S x json_proof) -> S 
        to apply to each proof
    :param inits: initial store S for the results
    :param json_data_dir: path to the directory with 'proofN.json's
    :returns: the accumulated results of applying f to
        each proof in the directory
    :rtype: S
    """
    results = inits
    for proof in tqdm(generate_from(json_data_dir), desc="Processing proofs", unit="proof"):
        results = f(results, proof)
    return results

# PROOF OPERATIONS

def get_keys(d, prefix=""):
    """Recursively finds all key-paths in the dictionary d

    :param d: dictionary to extract keys from
    :param prefix: prefix to prepend to each key
    :returns: list of keys preppended with prefix
    :rtype: list(str)
    """
    keys = []
    for key, value in d.items():
        if isinstance(value, dict):
            keys.extend(get_keys(value, prefix + key + "."))
        elif isinstance(value, list) and value and isinstance(value[0], dict):
            keys.extend(get_keys(value[0], prefix + key + "[]."))
        elif isinstance(value, list):
            keys.append(prefix + key + "[]")
        else:
            keys.append(prefix + key)
    return keys

def full_actions_of(proof_json):
    return [step['step']['action'] for step in proof_json['proof']['steps']]

def orig_objective_of(proof_json):
    return proof_json['proof']['steps'][0]['step']['action']

def actions_of(proof_json):
    return full_actions_of(proof_json)[1:]

def user_states_of(proof_json):
    return [step['step']['user_state'] for step in proof_json['proof']['steps'][1:]]

def constants_of(proof_json):
    return [step['step']['constants'] for step in proof_json['proof']['steps'][1:]]

def print_proof(proof_json):
    for act in full_actions_of(proof_json):
        print(act)

def user_proof_up_to(proof_json, i):
    proof_so_far = [orig_objective_of(proof_json)]
    for step in proof_json['proof']['steps'][1:i]:
        proof_so_far.append(step['step']['action'])
    return '\n'.join(proof_so_far)

def count_steps(proof_json):
    return len(proof_json['proof']['steps'])

# STRING REPRESENTATIONS

# TODO: Delete
ACTION_SEP = 'ACTION_SEP'
USER_STATE_SEP = 'USER_STATE'
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

Separator = {
    "action": "ACTION_SEP",
    "user_state": "USER_STATE",
    "goal": "GOAL_SEP",
    "name": "NAME_SEP",
    "term": "TERM_SEP",
    "hyps": "HYPS_SEP",
    "deps": "DEPS_SEP",
    "methods": "METHODS_SEP",
    "apply_kwrds": "APPLY_KWS_SEP",
    "isar_kwrds": "ISAR_KWS_SEP",
    "variables": "VARS_SEP",
    "constants": "CONSTS_SEP",
    "type variables": "TYPES_SEP",
}

def add_str_deps(proof_json, str_list=None):
    if str_list is None:
        str_list = []
    str_list.append(Separator["deps"])
    for dep in proof_json['proof'].get('deps', []):
        str_list.append(' '.join([Separator["name"], dep['thm']['name']]))
        str_list.append(' '.join([Separator["term"], dep['thm']['term']]))
    return str_list

def add_str_keywords(proof_json, kwrds_type=None, str_list=None):
    valid_kwds_type = kwrds_type in ["methods", "apply_kwrds", "isar_kwrds"]
    if str_list is None or not valid_kwds_type:
        str_list = []
    str_list.append(Separator[kwrds_type])
    for method in proof_json['proof'].get(kwrds_type, []):
        str_list.append(' '.join([Separator["name"], method['name']]))
    return str_list

def add_str_terms(proof_step, terms_type=None, str_list=None):
    valid_term_type = terms_type in ["hyps", "variables", "constants", "type variables"]
    if str_list is None or not valid_term_type:
        str_list = []
    str_list.append(Separator[terms_type])
    for terms_dict in proof_step['step'].get(terms_type, []):
        for _, term in terms_dict.items():
            str_list.append(' '.join([Separator["term"], term]))
    return str_list

def string_from(proof_json, readable=False):
    str_list = [orig_objective_of(proof_json)]
    sep_space = '\n' if readable else ' '

    for step in proof_json['proof']['steps'][1:]:
        usr_act_str = sep_space.join([
            ' '.join([USER_STATE_SEP, step['step']['user_state']]), 
            ' '.join([ACTION_SEP, step['step']['action']]), 
            ' '.join([GOAL_SEP, step['step']['term']])
        ])
        str_list.append(usr_act_str)
        str_list = add_str_terms(step, terms_type="hyps", str_list=str_list)
        str_list = add_str_terms(step, terms_type="variables", str_list=str_list)
        str_list = add_str_terms(step, terms_type="constants", str_list=str_list)
        str_list = add_str_terms(step, terms_type="type variables", str_list=str_list)

    str_list = add_str_keywords(proof_json, kwrds_type="apply_kwrds", str_list=str_list)
    str_list = add_str_keywords(proof_json, kwrds_type="isar_kwrds", str_list=str_list)
    str_list = add_str_deps(proof_json, str_list)
    str_list = add_str_keywords(proof_json, kwrds_type="methods", str_list=str_list)

    return sep_space.join(str_list)

def add_spk_data(proof_json, str_list, data_mode=isa_data.FORMATS["SPK"]):
    if isa_data.FORMATS["SP"] in data_mode:
        str_list = add_str_deps(proof_json, str_list)

    # TODO: add isar/apply keyword retrieval
    if isa_data.FORMATS["SPK"] in data_mode:
        str_list = add_str_keywords(proof_json, kwrds_type="methods", str_list=str_list)
    return str_list

def inputs_targets_from(proof_json, data_mode=isa_data.FORMATS["S"], readable=False):
    data = []
    sep_space = '\n' if readable else ' '
    for i, step in enumerate(proof_json['proof']['steps'][1:], 1):
        y = step['step']['action']
        xs = [
            user_proof_up_to(proof_json, i),
            ' '.join([USER_STATE_SEP, step['step']['user_state']])
        ]

        xs = add_spk_data(proof_json, xs, data_mode=data_mode)
        if isa_data.FORMATS["SPKT"] in data_mode:
            xs = add_str_terms(step, terms_type="variables", str_list=xs)
            xs = add_str_terms(step, terms_type="constants", str_list=xs)
            xs = add_str_terms(step, terms_type="type variables", str_list=xs)

        x = sep_space.join(xs)
        x = "isabelle next step: " + x if "finetune" in data_mode else x
        data.append((x, y))
    return data

# DIRECTORY COMPUTATIONS

def get_approx_tokens(token_counter, proof_json):
    """To be used with proofs.apply. Accumulates the token
    count of each proof in the input directory.
    
    :param token_counter: counter to accumulate token counts
    :param proof_json: dictionary representing a proof
    :returns: updated token counter
    :rtype: collections.Counter
    """
    proof_str = string_from(proof_json)
    proof_tokens = proof_str.split()
    for token in proof_tokens:
        token_counter[token] += 1
    return token_counter

def estimate_vocab_size(json_data_dir, coverage_threshold=0.95):
    """Estimates the optimal vocabulary size for tokenization
    of the proofs in the input directory.
    
    :param json_data_dir: path to the directory with 'proofN.json's
    :returns: the estimated optimal vocabulary size
    :rtype: int
    """
    token_counter = apply(get_approx_tokens, Counter(), json_data_dir)
    token_freqs = token_counter.most_common()
    total_tokens = sum(token_counter.values())

    cumulative_coverage = 0.0
    optimal_vocab_size = 0
    for i, (token, freq) in enumerate(token_freqs):
        cumulative_coverage += freq / total_tokens
        if cumulative_coverage >= coverage_threshold:
            optimal_vocab_size = i + 1
            break
    
    logging.info(f"Estimated vocab size is {optimal_vocab_size}")
    return optimal_vocab_size

# TODO: move to tokenizer_ops.py
def get_tokenizer_corpus(json_data_dir, readable=False):
    """Tokenizer's training corpus generator.
    
    :param json_data_dir: path to the data directory with 'proofN.json's  files
    :returns: generator of proof strings
    :rtype: generator
    """
    for proof in generate_from(json_data_dir):
        yield string_from(proof, readable)

def accumulate_approx_split_lengths(lengths, proof, data_mode=isa_data.FORMATS["S"]):
    """
    Accumulator to be used with proofs.compute_stats. It estimates the number of tokens 
    by splitting the proof's strings of inputs (and targets) with blank spaces.

    :param lengths: pair of lists (for lengths of input-target pairs) to accumulate
    :param proof: dictionary abstracting a proof
    :param data_mode: the data format
    :rtype: tuple(list)
    """
    x_y_pairs = inputs_targets_from(proof, data_mode=data_mode)
    lengths[0].extend(len(x.split()) for x, _ in x_y_pairs)
    lengths[1].extend(len(y.split()) for _, y in x_y_pairs)
    return lengths 

def compute_stats(accumulator, json_data_dir, **kwargs):
    """Computes the average, maximum, minimum, median, mode, and total size
    of the (input and target) tokens in the input directory's dataset using
    the estimation from the accumulator.
    
    :param accumulator: function tuple(list) -> tuple(list) with the lengths to process
    :param data_mode: the data format mode
    :returns: dictionary containing the tokenization statistics
    :rtype: dict
    """
    def get_stats(nums):
        return {
            "avg": sum(nums) / len(nums) if nums else 0,
            "max": max(nums, default=0),
            "min": min(nums, default=0),
            "median": statistics.median(nums) if nums else 0,
            "mode": statistics.mode(nums) if nums else 0
        }
    
    accumulate = lambda acc, proof: accumulator(acc, proof, **kwargs)
    x_lengths, y_lengths = apply(accumulate, ([],[]), json_data_dir)
    x_stats, y_stats = map(get_stats, [x_lengths, y_lengths])
    return {
        "x_avg": x_stats["avg"],
        "y_avg": y_stats["avg"],
        "x_max": x_stats["max"],
        "y_max": y_stats["max"],
        "x_min": x_stats["min"],
        "y_min": y_stats["min"],
        "x_median": x_stats["median"],
        "y_median": y_stats["median"],
        "x_mode": x_stats["mode"],
        "y_mode": y_stats["mode"],
        "total_datapoints": len(x_lengths)
    }


# SKELETON
# assumption: make_branch will be applied to the elements 
# of all_keys = apply(union_keys, [], data_path)

def make_leaf(name):
    result = {}
    if name.endswith("[]"):
        result[name[:-2]] = []
    else:
        result[name] = ""
    return result
    
def add_parent(name, d):
    result = {}
    if name.endswith("[]"):
        result[name[:-2]] = [d]
    else:
        result[name]= d
    return result

def make_branch(keys):
    result = {}
    parts = reversed(keys.split("."))
    for i, part in enumerate(parts):
        if i == 0:
            result = make_leaf(part)
        else:
            result = add_parent(part, result)
    return result

def merge_dicts(d1, d2):
    for key in d2:
        if key not in d1.keys():
            d1[key] = deepcopy(d2[key])
        else:
            if isinstance(d1[key], dict) and isinstance(d2[key], dict):
                d1[key] = merge_dicts(d1[key], d2[key])
            elif isinstance(d2[key], list) and isinstance(d1[key], list):
                if d1[key] and d2[key] and isinstance(d1[key][0], dict) and isinstance(d2[key][0], dict):
                    partial = {}
                    for d in d2[key]:
                        partial = merge_dicts(partial, d)
                    d1[key][0] = merge_dicts(d1[key][0], partial)
                else:
                    d1[key].extend(deepcopy(d2[key]))
            elif isinstance(d2[key], list) and isinstance(d1[key], str):
                d1[key] = deepcopy(d2[key])
            elif isinstance(d2[key], str) and isinstance(d1[key], list):
                d1[key] = d1[key]
            elif isinstance(d1[key], str) and isinstance(d2[key], str):
                d1[key] = ""
            else:
                raise Exception(f"Unconsidered case of {type(d1[key])} and {type(d2[key])} for {d2} and {d1}")
    return d1

def union_keys(keys, proof_json):
    json_keys = get_keys(proof_json)
    final_keys = list(set(json_keys) | set(keys))
    return final_keys

def make_skeleton(json_data_dir):
    all_keys = apply(union_keys, [], json_data_dir)
    dicts = list(map(make_branch, all_keys))
    skeleton = {}
    for d in dicts:
        skeleton = merge_dicts(skeleton, d)
    return skeleton


