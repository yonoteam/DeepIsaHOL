# Mantainers: 
#   Jonathan Julián Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for reading JSONs of proofs

import json
import os
from copy import deepcopy

debug = False

def print_debug(message):
    if debug:
        print(message)

def get_proof_json(proof_path):
    """Loads the data in proof_path as a dictionary. Returns {} if failed.

    Args:
        proof_path: str. Path to the (proof) JSON file.

    Returns:
        proof_data: dict. The dictionary abstracting a proof.
    """
    try:
        with open(proof_path, 'r') as json_file:
            proof_data = json.load(json_file)
            return proof_data
    except Exception as e:
        print(f"Failed to process {proof_path}: {str(e)}")
    return {}

def apply(f, inits, json_data_dir):
    """Apply f to each proof_json in json_data_dir starting with inits

    Args:
        inits: T
        f: (T x json_proof) -> T
        json_data_dir: str. Directory path containing JSON files.

    Returns:
        results: T. Aggregated results of applying f to each proof_json.
    """
    results = inits
    for subdir, _, files in os.walk(json_data_dir):
        json_files = [file for file in files if file.startswith("proof") and file.endswith(".json")]
        for file in json_files:
            json_path = os.path.join(subdir, file)
            proof_json = get_proof_json(json_path)
            results = f(results, proof_json)
    return results

def gen_apply(f, inits, data_dir):
    """Executes apply on each immediate subdirectory of data_dir

    Args:
        inits: T
        f: (T x json_proof) -> T
        data_dir: str. Data path containing subdirectories with JSON files.

    Returns:
        result: T. Aggregated results of applying f to each proof_json across all immediate subdirectories.
    """
    all_results = inits
    imm_subdirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir()]
    for path in imm_subdirs:
        print_debug(f"Processing directory: {path}")
        all_results = apply(f, all_results, path)
    return all_results

def get_keys(d, prefix=""):
    """Recursively finds all key-paths in the dictionary d

    Args:
        d: dict
        prefix: str
        json_data_dir: str. Directory path containing JSON files.

    Returns:
        keys: str list. The keys preppended with prefix
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

######## BUILDING THE SKELETON ########
# assumes make_branch will be applied to the elements of
# all_keys = gen_apply(union_keys, [], data_path)

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

def make_skeleton(data_path):
    all_keys = gen_apply(union_keys, [], data_path)
    dicts = list(map(make_branch, all_keys))
    skeleton = {}
    for d in dicts:
        skeleton = merge_dicts(skeleton, d)
    return skeleton

#######################################

def orig_objective_of(proof_json):
    return proof_json['proof']['steps'][0]['step']['action']

def user_states_of(proof_json):
    return [step['step']['user_state'] for step in proof_json['proof']['steps'][1:]]

def actions_of(proof_json):
    return [step['step']['action'] for step in proof_json['proof']['steps'][1:]]

def proof_step_keys(proof_json):
    return proof_json['proof']['steps'][0]['step'].keys()

def proof_env_keys(proof_json):
    return proof_json['proof'].keys()

def print_proof(proof_json):
    print(orig_objective_of(proof_json))
    for act in actions_of(proof_json):
        print(act)