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

import isa_data
import ops

def correct_parse_config_path():
    try:
        config_path = ops.parse_config_path(tool_explanation="Tests for successful information retrieval from input JSON configuration.")
        print("Test passed: correct_parse_config_path")
        return config_path
    except Exception as e:
        print(f"correct_parse_config_path failed to parse input file: {e}")
        print("using default path instead for remaining tests")
        return os.path.join(MAIN_DIR, "train_config.json")
    
def correct_get_json_dict():
    try:
        config_dict = ops.get_json_dict(correct_parse_config_path())
        print("Test passed: correct_get_json_dict")
        return config_dict
    except Exception as e:
        print(f"correct_get_json_dict failed to extract training configuration from input path: {e}")
        raise e
    
def correct_params():
    try:
        config_dict = correct_get_json_dict()
        expected_data_dir = "/path/to/data/dir/produced/by/isabelle_rl/proof_data_generation"
        if config_dict["data_dir"] != expected_data_dir:
            raise ValueError(f"'data_dir' is not {expected_data_dir}")
        
        expected_models_dir = "/save/path/for/this/model_name/models/"
        if config_dict["models_dir"] != expected_models_dir:
            raise ValueError(f"'models_dir' is not {expected_models_dir}")
        
        expected_datasets_dir = "/save/path/for/this/model_name/tokenized/datasets"
        if config_dict["datasets_dir"] != expected_datasets_dir:
            raise ValueError(f"'datasets_dir' is not {expected_datasets_dir}")
        
        expected_tokenizers_dir = "/save/path/for/this/model_name/tokenizers/"
        if config_dict["tokenizers_dir"] != expected_tokenizers_dir:
            raise ValueError(f"'tokenizers_dir' is not {expected_tokenizers_dir}")
        
        expected_model_name = "hugging_face_model_name"
        if config_dict["model_name"] != expected_model_name:
            raise ValueError(f"'model_name' is not the string {expected_model_name}")
        
        expected_data_mode = isa_data.FORMATS["S"]
        if config_dict["data_mode"] != expected_data_mode:
            raise ValueError(f"'data_mode' is not the string {expected_data_mode}")
        
        expected_data_split = isa_data.SPLITS["NONE"]
        if config_dict["data_split"] != expected_data_split:
            raise ValueError(f"'data_split' is not the string {expected_data_split}")
        
        if config_dict["batch_size"] != 8:
            raise ValueError(f"'batch_size' is not {8}")
        
        if config_dict["num_epochs"] != 1:
            raise ValueError(f"'num_epochs' is not {1}")
        
        print("Test passed: correct_params")
    except Exception as e:
        print(f"correct_params failed: {e}")

if __name__ == "__main__":
    correct_params()