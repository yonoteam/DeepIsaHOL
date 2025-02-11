# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Test file for ops.py

import sys
import os

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import ops

def correct_parse_config_path():
    try:
        config_path = ops.parse_config_path(tool_explanation="Tests successful information retrieval from input JSON configuration.")
        print("Test passed: correct_parse_config_path")
        return config_path
    except Exception as e:
        print(f"correct_parse_config_path failed to parse input file: {e}")
        print("using default path instead for remaining tests")
        return os.path.join(MAIN_DIR, "train_config.json")
    
def correct_get_config_dict():
    try:
        config_dict = ops.get_config_dict(correct_parse_config_path())
        print("Test passed: correct_get_config_dict")
        return config_dict
    except Exception as e:
        print(f"correct_get_config_dict failed to extract a dictionary from input path: {e}")
        raise e
    
def correct_extract_params():
    try:
        params = ops.extract_params(correct_get_config_dict())
        print("Test passed: correct_extract_params")
        return params
    except Exception as e:
        print(f"correct_extract_params failed to extract information from input dictionary: {e}")
        raise e
    
def correct_params():
    try:
        data_dir, all_models_dir, model_name, mode, num_epochs = correct_extract_params()
        if data_dir != "/path/to/data":
            raise Exception(f"correct_params failed, 'data_dir' is not '/path/to/data'")
        if all_models_dir != "/path/to/all/models":
            raise Exception(f"correct_params failed, 'all_models_dir' is not '/path/to/all/models'")
        if model_name != "model_name":
            raise Exception(f"correct_params failed, 'model_name' is not the string 'model_name'")
        if mode != "state":
            raise Exception(f"correct_params failed, 'mode' is not the string 'state'")
        if num_epochs != 10:
            raise Exception(f"correct_params failed, 'num_epochs' is not {10}")
        ops.check_params(TEST_DIR, SRC_DIR)
        print("Test passed: correct_params")
    except Exception as e:
        print(f"correct_params failed: {e}")

if __name__ == "__main__":
    correct_params()