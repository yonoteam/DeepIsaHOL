"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Utilities for turning and manipulating the proof data as strings.
"""

import json
from typing import List, Tuple

from proofs import user_proof_up_to

def fix_missing_quotations(s: str) -> str:
    """
    Fixes strings with an odd number of quotation marks by appending one.

    :param s: input string
    """
    double_quotes = s.count('"') % 2
    if double_quotes:
        s += '"'
    return s

def convert_by_to_apply(by_statement: str) -> str:
    """
    Converts a 'by' statement into one or more 'apply' statements.
    
    :param by_statement: A string starting with 'by'
        e.g. 'by (induct rule: x.induct) (auto simp: x_def)'
    :return: equivalent series of apply statements as a string
        e.g. 'apply (induct rule: x.induct)\napply (auto simp: x_def)'
    """
    
    args_string = by_statement.strip()[2:].strip()
    if not args_string:
        return "apply"
    
    # Process each character to correctly handle nested parentheses
    apply_statements = []
    current_arg = ""
    parenthesis_count = 0
    for char in args_string:
        if char == '(' and current_arg.strip() == "":
            # start of a new parenthesized argument
            parenthesis_count = 1
            current_arg += char
        elif char == '(' and parenthesis_count > 0:
            # nested opening parenthesis
            parenthesis_count += 1
            current_arg += char
        elif char == ')' and parenthesis_count > 0:
            # closing parenthesis
            parenthesis_count -= 1
            current_arg += char
            
            # if all closed, add current argument
            if parenthesis_count == 0 and current_arg.strip():
                apply_statements.append(f"apply {current_arg.strip()}")
                current_arg = ""
        else:
            # regular character
            if parenthesis_count > 0:
                current_arg += char
            elif char.strip():
                # start a new non-parenthesized argument
                current_arg += char
            elif current_arg.strip():
                # whitespace after a non-parenthesized argument
                apply_statements.append(f"apply {current_arg.strip()}")
                current_arg = ""
    
    # remaining argument
    if current_arg.strip():
        apply_statements.append(f"apply {current_arg.strip()}")
    
    return "\n".join(apply_statements)

# FORMATTING 

FORMATS = {
    "S": "state", 
    "SP": "state_prems", 
    "SPK": "state_prems_kwrds", 
    "SPKT": "state_prems_kwrds_terms",
    "FS": "finetune_state",
    "FSP": "finetune_state_prems", 
    "FSPK": "finetune_state_prems_kwrds", 
    "FSPKT": "finetune_state_prems_kwrds_terms",
}

def add_spk_data(
        proof_json: dict,
        x_dict: dict,
        data_format: str = FORMATS["S"]
    ) -> dict:
    if FORMATS["SP"] in data_format:
        x_dict["deps"] = proof_json.get('deps', [])

    # TODO: add isar/apply keyword retrieval?
    if FORMATS["SPK"] in data_format:
        x_dict["methods"] = proof_json.get('methods', [])
    
    return x_dict

def inputs_targets_from(
        proof_json: dict,
        data_format: str = FORMATS["S"]
    ) -> List[Tuple[str, str]]:
    """
    Extracts (input, target) pairs from a proof for training or analysis.

    :param proof_json: dictionary representation of the proof
    :param data_format: one of the values of the FORMATS dictionary
    """
    data = []
    for i, step in enumerate(proof_json['proof'][1:], 1):
        y = step['action']
        x_dict = {
            "proof_so_far": user_proof_up_to(proof_json, i),
            "last_usr_state": step.get('user_state', "")
        }
        x_dict = add_spk_data(proof_json, x_dict, data_format)
        
        if FORMATS["SPKT"] in data_format:
            x_dict["variables"] = step.get('variables', [])
            x_dict["constants"] = step.get('constants', [])
            x_dict["type variables"] = step.get('type variables', [])

        x = json.dumps(x_dict)
        data.append((x, y))
    return data
