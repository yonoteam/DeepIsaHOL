# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Test file for proofs.py

import sys
import os
import logging
from io import StringIO

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import isa_data
import proofs

SKELETON = {
    'proof': {
        'deps': [
            {'thm': {'name': '', 'term': ''}}
        ],
        'steps': [
            {
                'step': {
                    'variables': [{'Type0': ''}],
                    'type variables': [{'Sort0': ''}],
                    'user_state': '',
                    'term': '',
                    'proven': [],
                    'action': '',
                    'constants': [{'Type0': ''}],
                    'hyps': [{'term': ''}]
                }
            }
        ],
        'isar_kwrds': [{'name': ''}],
        'methods': [{'name': ''}],
        'apply_kwrds': [{'name': ''}]
    }
}

BAD_PROOF_PATH = os.path.join(TEST_DIR, "proof_bad.json")

def get_bad_proof():
    try:
        with open(BAD_PROOF_PATH, 'r') as f:
            original_content = f.read()
    except FileNotFoundError:
        print("Tests failed: proof_bad.json does not exist.")
    return original_content

BAD_PROOF = get_bad_proof()

def restore_bad_proof():
    with open(BAD_PROOF_PATH, 'w') as f:
        f.write(BAD_PROOF)

def get_proof_json():
    skeleton_dir = os.path.join(TEST_DIR, "proof_skeleton.json")
    skeleton = proofs.get_proof_json(skeleton_dir)
    if skeleton == SKELETON:
        print(f"get_proof_json() passed")
    else:
        print(f"get_proof_json() failed")

def valid_data_dir():
    if proofs.valid_data_dir(TEST_DIR):
        print(f"valid_data_dir() passed")
    else:
        print(f"valid_data_dir() failed")

def generate_paths_from():
    paths = list(proofs.generate_paths_from(TEST_DIR))
    if os.path.join(TEST_DIR, "proof_skeleton.json") in paths:
        print(f"generate_paths_from() passed")
    else:
        print(f"generate_paths_from() failed")

def generate_from():
    # capture logs
    log_stream = StringIO()
    handler = logging.StreamHandler(log_stream)
    logger = logging.getLogger()
    logger.addHandler(handler)

    # actual test
    proof_dicts = list(proofs.generate_from(TEST_DIR))

    # check error logs
    handler.flush()
    log_contents = log_stream.getvalue()
    error_logged = "Failed to process" in log_contents

    # clean up handler
    logger.removeHandler(handler)
    log_stream.close()

    if SKELETON in proof_dicts and error_logged:
        print(f"generate_from() passed")
    else:
        print(f"generate_from() failed")

def get_proofs_paths():
    suffixes = ['proof.json', 'proof_bad.json', 'proof_skeleton.json']
    real_paths = [os.path.join(TEST_DIR, suffix) for suffix in suffixes]
    test_paths = proofs.get_proofs_paths(TEST_DIR)
    if set(real_paths) == set(test_paths):
        print(f"get_proofs_paths() passed")
    else:
        print(f"get_proofs_paths() failed")

def find_erroneous():
    erroneous_paths = proofs.find_erroneous(TEST_DIR)
    try:
        if erroneous_paths[0] == BAD_PROOF_PATH:
            print(f"find_erroneous() passed")
        else:
            print(f"find_erroneous() failed")
    except IndexError:
        print(f"find_erroneous() failed: index out of range")

def delete_erroneous():  
    deleted_files, _ = proofs.delete_erroneous(TEST_DIR)
    if BAD_PROOF_PATH in deleted_files and not os.path.exists(BAD_PROOF_PATH):
        print("delete_erroneous() passed")
    else:
        print("delete_erroneous() failed")
    restore_bad_proof()

def count_steps():
    proof = proofs.get_proof_json(os.path.join(TEST_DIR, "proof.json"))
    count = proofs.count_steps(proof)
    if count > 1:
        print("count_steps() passed")
    else:
        print("count_steps() failed")

def apply():
    _ = proofs.delete_erroneous(TEST_DIR)
    try:
        counts = proofs.apply(lambda n, proof: n + proofs.count_steps(proof), 0, TEST_DIR)
        if counts > 0:
            print("apply() passed")
        else:
            print("apply() failed")
    except Exception as e:
        print(f"apply() failed: {e}")
    finally:
        restore_bad_proof()

def get_keys():
    skeleton = proofs.get_proof_json(os.path.join(TEST_DIR, "proof_skeleton.json"))
    keys = proofs.get_keys(skeleton)
    skeleton_keys = proofs.get_keys(SKELETON)
    if set(keys) == set(skeleton_keys):
        print("get_keys() passed")
    else:
        print("get_keys() failed")

def make_skeleton():
    _ = proofs.delete_erroneous(TEST_DIR)
    try:
        skeleton = proofs.make_skeleton(TEST_DIR)
        if skeleton == SKELETON:
            print("make_skeleton() passed")
        else:
            print("make_skeleton() failed")
    except Exception as e:
        print(f"make_skeleton() failed: {e}")
    finally:
        restore_bad_proof()

def print_proof():
    proof = proofs.get_proof_json(os.path.join(TEST_DIR, "proof.json"))
    proofs.print_proof(proof)

def inputs_targets_from(test_dir=TEST_DIR):
    try:
        iter_f = lambda data_list, proof: proofs.inputs_targets_from(proof, data_mode=isa_data.FORMATS["SPKT"], readable=False)
        proofs.apply(iter_f, [], test_dir)
        print("input_targets_from() passed")
    except Exception as e:
        print(f"input_targets_from() failed: {e}")

if __name__ == "__main__":
    get_proof_json()
    valid_data_dir()
    generate_paths_from()
    generate_from()
    get_proofs_paths()
    find_erroneous()
    delete_erroneous()
    count_steps()
    apply()
    get_keys()
    make_skeleton()
    print("An Isabelle proof is:")
    print_proof()