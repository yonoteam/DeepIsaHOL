"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Utilities retrieving proofs' data.
"""

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

def user_proof_up_to(proof_json, i):
    proof_so_far = [orig_objective_of(proof_json)]
    for step in proof_json['proof']['steps'][1:i]:
        proof_so_far.append(step['step']['action'])
    return '\n'.join(proof_so_far)

def count_steps(proof_json):
    return len(proof_json['proof']['steps'])

def print_proof(proof_json):
    for act in full_actions_of(proof_json):
        print(act)

__all__ = [
    "full_actions_of",
    "orig_objective_of",
    "actions_of",
    "user_states_of",
    "constants_of",
    "user_proof_up_to",
    "count_steps",
    "print_proof"
]