# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# A collection of reused methods/operations

import os
import time
import json
import queue
import logging
import argparse
import threading

import proofs

def print_dict(d, max=None):
    for i, (k, v) in enumerate(d.items()):
        print(f"{k}: {v}")
        if i == max:
            break

def save_dict_as_json(data, file_path, indent=4):
    """
    Saves a dictionary as a JSON file.

    :param data: dictionary to save
    :param file_path: path to the output JSON file
    :param indent: indentation level for pretty-printing (default: 4)
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=indent)
        logging.info(f"Dictionary successfully saved to {file_path}")
    except Exception as e:
        message = f"Error saving dictionary to JSON: {e}"
        logging.error(message)
        print(message)

def apply_with_timeout(timeout_in_secs, f, *args, **kwargs):
    """
    Executes `f` with arguments `args` and keyword arguments `kwargs`.
    If `f` does not finish within `timeout_in_secs`, the process times out.

    :param timeout_in_secs: Timeout in seconds.
    :param f: Function to execute.
    :param args: Positional arguments to pass to `f`.
    :param kwargs: Keyword arguments to pass to `f`.
    :return: The result of `f` if it finishes within the timeout, else None.
    """
    result = queue.Queue()

    def wrapped_f():
        result.put(f(*args, **kwargs))

    thread = threading.Thread(target=wrapped_f)
    thread.start()
    start_time = time.time()

    while time.time() - start_time < timeout_in_secs:
        if not result.empty():
            return result.get()
        time.sleep(0.1)  # Avoid busy waiting

    logging.info("Timeout reached. Function did not complete.")
    return None


# CONFIGURATION SETUP

def configure_logging(log_file_name, log_level=logging.DEBUG, save_dir=os.getcwd()):
    """
    Applies 'logging.basicConfig' with inputs guiding the configuration. 
    See 'logging' package documentation.
    
    :param log_file_name: Name of the log file (add '.log' ending)
    :param log_level: Choose among logging.{TRACE, DEBUG, INFO, WARN, ERROR, FATAL}.
    :param save_dir: Path to the directory where one wants the log file to be saved.
    """
    log_file = os.path.join(save_dir, log_file_name)
    
    logging.basicConfig(
        filename=log_file,
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging configured. Writing logs to %s", log_file)

def parse_config_path(tool_explanation="Does something as specified in the config path."):
    """
    Parses the first argument (of type 'str') and returns it.
    
    :param tool_explanation: Description of what the main function calling this method will do.
    """
    parser = argparse.ArgumentParser(description=tool_explanation)
    parser.add_argument("config_path", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()
    return args.config_path

def get_config_dict(json_path):
    """
    Extracts a dictionary from the input path to a JSON file.
    
    :param json_path: Path to the JSON file.
    :rtype: dict
    """
    if not os.path.isfile(json_path):
        raise Exception(f"The configuration file '{json_path}' does not exist.")

    with open(json_path, "r") as file:
        config = json.load(file)
    
    return config

def check_params(config_dict):
    """
    If the input configuration is not correct, it raises an error. Otherwise, it does nothing.
    It tests the existence of the following parameters.
    'data_dir' - (str) path to a directory recursively containing at leasst one proofN.json
    'model_name' - (str) a Hugging Face model name
    'all_models_dir' - (str) the directory where the tokenizers, datasets and models are saved
    'mode' - (str) any of the proofs.MODES
    'num_epochs' - (int) the number of times the training loop is repeated

    :param config_dict: (dict) dictionary containing the training configuration
    """
    try:
        data_dir = config_dict["data_dir"]
        _ = config_dict["model_name"]
        all_models_dir = config_dict["all_models_dir"]
        _ = config_dict["mode"]
        _ = int(config_dict["num_epochs"])
    except KeyError as e:
        raise Exception(f"Error extracting parameters from configuration dictionary: {e}")
    
    if not proofs.valid_data_dir(data_dir):
        message = f"""No subdirectory in '{data_dir}' contains  a 
        JSON file that starts with 'proof' and ends with '.json'.""".format()
        raise Exception(f"Error: {message}.")
    
    if not os.path.isdir(all_models_dir):
        message = f"""Input '{all_models_dir}' is not a directory."""
        raise Exception(f"Error: {message}")


# CONFIGURATION RETRIEVAL

def get_directory_paths(config_dict):
    """
    Returns candidate paths to save tokenizers, datasets, and models from the inputs.

    :param all_models_dir: (str) the directory where the tokenizers, datasets and models are saved
    :param model_name: (str) a Hugging Face model name
    :param mode: (str) any of the proofs.MODES
    :rtype: (str, str, str) tuple
    """
    all_models_dir = config_dict["all_models_dir"]
    model_name = config_dict["model_name"]
    mode = config_dict["mode"]
    local_model_dir = os.path.join(all_models_dir, model_name)
    tokenizers_dir = os.path.join(local_model_dir, "tokenizers")
    datasets_dir = os.path.join(tokenizers_dir, f"datasets/{mode}")
    models_dir = os.path.join(local_model_dir, "models", mode)
    return tokenizers_dir, datasets_dir, models_dir

def get_latest_dir_from(saving_dir, adding_one=False):
    """
    If the input path has subdirectories named with numbers and adding_one is False, 
    it returns the path to the highest of them. If adding_one is True, it returns the 
    path to the highest number +1. If there are no number-named directories, it returns
    0 and adding_one is irrelevant.

    :param saving_dir: (str) the directory with possibly number-named subdirectories
    :param adding_one: (bool) whether to return the highest number-named subdirectory +1
    :rtype: str
    """
    subdirs = [
        os.path.join(saving_dir, d) for d in os.listdir(saving_dir)
        if os.path.isdir(os.path.join(saving_dir, d)) and d.isdigit()
    ]
    if subdirs:
        latest_number = max(int(os.path.basename(d)) for d in subdirs)
        if adding_one:
            latest_number = latest_number + 1
    else:
        latest_number = 0
    
    latest_dir = os.path.join(saving_dir, str(latest_number))
    return latest_dir

def save_hf_data_in(hf_data, saving_dir):
    """
    Saves the input tokenizer or model in the latest number-named subdirectory +1.

    :param hf_data: either a Hugging Face tokenizer or model
    :param saving_dir: (str) the directory with possibly number-named subdirectories
    """
    try:
        os.makedirs(saving_dir, exist_ok=True)
        new_dir = get_latest_dir_from(saving_dir, adding_one=True)
        os.makedirs(new_dir, exist_ok=False)
        hf_data.save_pretrained(new_dir)
        logging.info(f"Saved Hugging Face data in {new_dir}")
    except FileExistsError:
        logging.warning(f"Directory '{new_dir}' already exists. Skipping save.")
    except Exception as e:
        logging.error(f"Failed to save hugging face data: {e}")
