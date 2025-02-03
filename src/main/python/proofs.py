# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for reading JSONs of proofs

import os
import json
import logging

from copy import deepcopy
from pathlib import Path

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
    Generator for all "proofN.json" file-paths in the 
    input directory.

    :param json_data_dir: path to the directory to search
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

    :param json_data_dir: path to the directory to search
    :returns: generator of all "proofN.json" files
    :rtype: generator
    """
    for proof_path in generate_paths_from(json_data_dir):
        proof = get_proof_json(proof_path)
        yield proof

def get_proofs_paths(json_data_dir):
    """
    Finds all paths "proofN.json" in the input directory.

    :param json_data_dir: path to the directory to search
    :returns: sorted list of paths to JSON files
    :rtype: list(str)
    """
    all_paths = []
    for proof_path in generate_paths_from(json_data_dir):
        all_paths.append(proof_path)
    return all_paths

def find_erroneous(json_data_dir, f=lambda x: x):
    """
    Attempts to load each "proofN.json" in the input directory
    and apply `f` to it. If it fails, it stores the failing
    file path in the returned list.

    :param json_data_dir: path to the directory to search
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

    :param json_data_dir: path to the directory to search
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
    :param json_data_dir: path to the directory holding
        the proof JSON files
    :returns: the accumulated results of applying f to
        each proof in the directory
    :rtype: S
    """
    results = inits
    for proof in generate_from(json_data_dir):
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

def count_steps(proof_json):
    return len(proof_json['proof']['steps'])

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


