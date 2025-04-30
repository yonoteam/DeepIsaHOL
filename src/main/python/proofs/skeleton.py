"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Utility for creating an empty structure (a skeleton) from the 
generated data-directory of "proof*.json" files.
"""

from copy import deepcopy

import dicts
from proofs import data_dir

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
    json_keys = dicts.get_keys(proof_json)
    final_keys = list(set(json_keys) | set(keys))
    return final_keys

def make_skeleton(json_data_dir):
    all_keys = data_dir.apply(union_keys, [], json_data_dir)
    dicts = list(map(make_branch, all_keys))
    skeleton = {}
    for d in dicts:
        skeleton = merge_dicts(skeleton, d)
    return skeleton
