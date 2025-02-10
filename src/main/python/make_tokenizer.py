
import os
import json
import logging
import argparse

import train_t5
import tokenizer_ops as tokops

if __name__ == "__main__":
    # Parser setup
    parser = argparse.ArgumentParser(description="Train a new tokenizer as specified in the input JSON configuration.")
    parser.add_argument("config_path", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Check if JSON configuration exists
    if not os.path.isfile(args.config_path):
        logging.error(f"The configuration file '{args.config_path}' does not exist.")
        exit(1)
    
    # Load the JSON configuration
    try:
        with open(args.config_path, "r") as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse the JSON file. {e}")
        exit(1)
    
    # Setup from config
    try:
        data_dir, all_models_dir, model_name, mode, num_epochs = train_t5.extract_params(config)
        train_t5.test_params(data_dir, all_models_dir)
    except Exception as e:
        logging.error(f"Could not setup from configuration file: '{e}'.")
        exit(1)
    
    tokenizers_dir, datasets_dir, models_dir = train_t5.make_dir_vars(all_models_dir, model_name, mode)
    _ = tokops.get_trained_tokenizer(True, data_dir, tokenizers_dir, model_name)