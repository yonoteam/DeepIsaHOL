"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Utilities for turning and manipulating the proof data as strings.
"""

from typing import List, Tuple, Optional

from proofs import orig_objective_of, user_proof_up_to

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

# TODO: turn into a constant and make key-retrieving functions
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

def add_deps(
        proof_json: dict,
        str_list: Optional[List[str]] = None
    ) -> List[str]:
    """
    Adds the string representation of the proof's dependencies (theorems) 
    to the input list of strings.

    :param proof_json: dictionary representation of the proof
    :param str_list: list of strings to extend
    """
    if str_list is None:
        str_list = []
    str_list.append(Separator["deps"])
    for dep in proof_json['proof'].get('deps', []):
        str_list.append(' '.join([Separator["name"], dep['thm']['name']]))
        str_list.append(' '.join([Separator["term"], dep['thm']['term']]))
    return str_list

def add_keywords(
        proof_json: dict,
        kwrds_type: Optional[str] = None,
        str_list: Optional[List[str]] = None
    ) -> List[str]:
    """
    Adds the available keywords at proof-time to the input list of strings.

    :param proof_json: dictionary representation of the proof
    :param kwrds_type: one of "methods", "apply_kwrds", or "isar_kwrds"
    :param str_list: list of strings to extend
    """
    valid_kwds_type = kwrds_type in ["methods", "apply_kwrds", "isar_kwrds"]
    if str_list is None or not valid_kwds_type:
        str_list = []
    str_list.append(Separator[kwrds_type])
    for method in proof_json['proof'].get(kwrds_type, []):
        str_list.append(' '.join([Separator["name"], method['name']]))
    return str_list

def add_terms(
        proof_step: dict,
        terms_type: Optional[str] = None,
        str_list: Optional[List[str]] = None
    ) -> List[str]:
    """
    Adds the string representation of the proof's to the input list of strings.

    :param proof_json: dictionary representation of the proof
    :param terms_type: one of "hyps", "variables", "constants", or "type variables"
    :param str_list: list of strings to extend
    """
    valid_term_type = terms_type in ["hyps", "variables", "constants", "type variables"]
    if str_list is None or not valid_term_type:
        str_list = []
    str_list.append(Separator[terms_type])
    for terms_dict in proof_step['step'].get(terms_type, []):
        for _, term in terms_dict.items():
            str_list.append(' '.join([Separator["term"], term]))
    return str_list

def string_from(proof_json: dict, readable: bool = False) -> str:
    """
    Provides a string representation of the proof with optional line spacing.

    :param proof_json: dictionary representation of the proof
    :param readable: whether to format the output with newlines
    """
    str_list = [orig_objective_of(proof_json)]
    sep_space = '\n' if readable else ' '

    for step in proof_json['proof']['steps'][1:]:
        usr_act_str = sep_space.join([
            ' '.join([Separator['user_state'], step['step']['user_state']]), 
            ' '.join([Separator['action'], step['step']['action']]), 
            ' '.join([Separator['goal'], step['step']['term']])
        ])
        str_list.append(usr_act_str)
        str_list = add_terms(step, terms_type="hyps", str_list=str_list)
        str_list = add_terms(step, terms_type="variables", str_list=str_list)
        str_list = add_terms(step, terms_type="constants", str_list=str_list)
        str_list = add_terms(step, terms_type="type variables", str_list=str_list)

    str_list = add_keywords(proof_json, kwrds_type="apply_kwrds", str_list=str_list)
    str_list = add_keywords(proof_json, kwrds_type="isar_kwrds", str_list=str_list)
    str_list = add_deps(proof_json, str_list)
    str_list = add_keywords(proof_json, kwrds_type="methods", str_list=str_list)

    return sep_space.join(str_list)

def add_spk_data(
        proof_json: dict,
        str_list: List[str],
        data_format: str = FORMATS["SPK"]
    ) -> List[str]:
    """
    Conditinally adds keywords and dependencies to the input list of strings
    based on the specified data mode.

    :param proof_json: dictionary representation of the proof
    :param str_list: list of strings to extend
    :param data_format: one of the values of the FORMATS dictionary
    """
    if FORMATS["SP"] in data_format:
        str_list = add_deps(proof_json, str_list)

    # TODO: add isar/apply keyword retrieval
    if FORMATS["SPK"] in data_format:
        str_list = add_keywords(proof_json, kwrds_type="methods", str_list=str_list)
    return str_list

def inputs_targets_from(
        proof_json: dict,
        data_format: str = FORMATS["S"],
        readable: bool = False
    ) -> List[Tuple[str, str]]:
    """
    Extracts (input, target) pairs from a proof for training or analysis.

    :param proof_json: dictionary representation of the proof
    :param data_format: one of the values of the FORMATS dictionary
    :param readable: whether to use newlines between sections
    """
    data = []
    sep_space = '\n' if readable else ' '
    for i, step in enumerate(proof_json['proof']['steps'][1:], 1):
        y = step['step']['action']
        xs = [
            user_proof_up_to(proof_json, i),
            ' '.join([Separator['user_state'], step['step']['user_state']])
        ]

        xs = add_spk_data(proof_json, xs, data_format=data_format)
        if FORMATS["SPKT"] in data_format:
            xs = add_terms(step, terms_type="variables", str_list=xs)
            xs = add_terms(step, terms_type="constants", str_list=xs)
            xs = add_terms(step, terms_type="type variables", str_list=xs)

        x = sep_space.join(xs)
        x = "isabelle next step: " + x if "finetune" in data_format else x
        data.append((x, y))
    return data
