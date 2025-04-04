# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# A collection of reused methods/operations

import os
import re
import time
import json
import queue
import logging
import argparse
import threading

from typing import Union
from datetime import datetime

import torch
import matplotlib.pyplot as plt
from accelerate import Accelerator
from transformers.tokenization_utils_base import BatchEncoding

import isa_data
import proofs
import accelerate_test

def dict_to_string(d, max=None):
    s = ""
    for i, (k, v) in enumerate(d.items()):
        s = s + f"{k}: {v}\n"
        if i == max:
            return s
    return s

def print_dict(d, max=None):
    print(dict_to_string(d, max=max))

def to_plot(read_json_path:str, loop_name:str, start_from:int=0):
    read_dir = os.path.dirname(read_json_path)
    metric_dict = get_json_dict(read_json_path)
    save_plot(metric_dict["steps"][start_from:], 
                metric_dict["loss"][start_from:], 
                save_path=os.path.join(read_dir, f"{loop_name}_loss.png"), 
                title=f"{loop_name} Loss", 
                x_label="Steps", 
                y_label="Loss")
    save_plot(metric_dict["steps"][start_from:], 
                metric_dict["accuracy"][start_from:], 
                save_path=os.path.join(read_dir, f"{loop_name}_accur.png"), 
                title=f"{loop_name} Accuracy", 
                x_label="Steps",
                y_label="Accuracy")
    
def ancester_dir_exists(path):
    """
    Checks if an ancester directory for the input FULL path is an existing directory.

    :param path: the path to check.
    :rtype: bool
    """
    while True:
        directory = os.path.dirname(path)
        if os.path.isdir(directory):
            return True
        if directory == os.path.dirname(directory): # checking if at root directory.
            return False
        path = directory

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

def avg_repeat_time(log_file_path: Union[str, os.PathLike], event_description: str):
    """
    Calculates an event's average repetition time in seconds from a log file.
    Returns None if the event is not found or occurs only once.

    :param log_file_path: Path to the log file.
    :param event_description: The string to search for.
    :rtype: float
    """
    timestamps = []
    try:
        with open(log_file_path, 'r') as log_file:
            for line in log_file:
                if event_description in line:
                    match = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2},\d{3})', line)
                    if match:
                        timestamp_str = match.group(1)
                        timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S,%f')
                        timestamps.append(timestamp)
    except FileNotFoundError:
        print(f"Error: Log file '{log_file_path}' not found.")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

    if len(timestamps) < 2:
        return None
    
    time_diffs = []
    for i in range(1, len(timestamps)):
        time_diff = (timestamps[i] - timestamps[i - 1]).total_seconds()
        time_diffs.append(time_diff)

    average_time = sum(time_diffs) / len(time_diffs)
    return average_time


# SAVING DATA

def save_plot(x_vals, y_vals, save_path: Union[str, os.PathLike]="curve.png", title="Graph of y vs x", x_label="X-axis", y_label="Y-axis") -> None:
    """
    Plots a curve from the inputs and saves the graph.

    :param x_vals: independent variable values.
    :param y_vals: dependent variable values.
    :param save_path: Path to save the image.
    :param title: Title of the graph.
    :param x_label: Label for the x-axis.
    :param y_label: Label for the y-axis.
    """
    plt.figure(figsize=(8, 5))
    plt.plot(x_vals, y_vals, marker='o', linestyle='-', color='b', label="Curve")
    
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.legend()
    plt.grid(True)

    plt.savefig(save_path, dpi=300)
    plt.close()

def save_dict_as_json(data: dict, save_path: Union[str, os.PathLike], indent=4) -> None:
    """
    Saves a dictionary as a JSON file.

    :param data: dictionary to save
    :param save_path: path to the output JSON file
    :param indent: indentation level for pretty-printing (default: 4)
    """
    try:
        with open(save_path, 'w') as json_file:
            json.dump(data, json_file, indent=indent)
        logging.info(f"Dictionary successfully saved to {save_path}")
    except Exception as e:
        message = f"Error saving dictionary to JSON: {e}"
        logging.error(message)
        print(message)

def save_tuple_list_as_txt(list_of_tuples, save_path: Union[str, os.PathLike]) -> None:
    """Writes a list of tuples of strings to a .txt file.

    :param list_of_tuples: A list of tuples of strings.
    :param filename: The name of the .txt file to write to.
    """
    try:
        with open(save_path, 'w', encoding='utf-8') as f:
            for tup in list_of_tuples:
                line = "\n".join(tup)
                f.write(line + "\n\n")
        logging.info(f"List successfully saved to {save_path}")
    except Exception as e:
        message = f"Error saving list of tuples of strings to txt: {e}"
        logging.error(message)
        print(message)

def save_batch(tensors: BatchEncoding, save_path: Union[str, os.PathLike]) -> None:
    """
    Saves a transformers BatchEncoding as a dictionary of tensors to a file.

    :param tensors: batch encoding of tensors.
    :param save_path: Path to the output file.
    """
    try:
        save_dict = {k: v for k, v in tensors.items()}
        with open(save_path, 'wb') as f:
            torch.save(save_dict, f, weights_only=True)
        logging.info(f"List of tensors successfully saved to {save_path}")
    except Exception as e:
        message = f"Error saving tensors to file: {e}"
        logging.error(message)
        print(message)


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

def get_json_dict(json_path):
    """
    A method to extract a dictionary from the input path 
    that returns an empty dictionary on failure

    :param json_path: Path to the JSON file.
    :rtype: dict
    """
    result = {}
    try:
        if not os.path.exists(json_path):
            logging.error(f"Path '{json_path}' does not exist.")
            return result

        if not os.path.isfile(json_path):
            logging.error(f"Path '{json_path}' is not a file.")
            return result
        
        with open(json_path, "r") as file:
            result = json.load(file)
            return result
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{json_path}': {e}")
    except OSError as e:
        logging.error(f"OS error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")

    return result

def check_params(config_dict):
    """
    If the input configuration dictionary is not correct, it raises an error. Otherwise, it does nothing.
    It tests the existence of the following parameters.
    'data_dir' - (str) path to a directory recursively containing at leasst one proofN.json
    'model_name' - (str) a Hugging Face model name
    'models_dir' - (str) path to a directory for saving models
    'tokenizers_dir' - (str) path to a directory for saving tokenizers
    'datasets_dir' - (str) path to a directory for saving tokenized datasets
    'data_mode' - (str) any of the isa_data.FORMATS
    'data_split' - (str) any of the possible dataset splits as in isa_data.SPLITS. It will be ignnored in the non-testing scripts.
    'batch_size' - (int) the number of samples per batch
    'num_epochs' - (int) the number of times the training/validation loop is repeated

    :param config_dict: (dict) dictionary containing the training configuration
    """
    try:
        data_dir = config_dict["data_dir"]
        _ = config_dict["model_name"]
        models_dir = config_dict["models_dir"]
        tokenizers_dir = config_dict["tokenizers_dir"]
        datasets_dir = config_dict["datasets_dir"]
        data_mode = config_dict["data_mode"]
        data_split = config_dict["data_split"]
        _ = int(config_dict["batch_size"])
        _ = int(config_dict["num_epochs"])
    except KeyError as e:
        raise KeyError(f"Error extracting parameter from configuration dictionary: {e}")
    except ValueError as e:
        raise ValueError(f"Error extracting parameter from configuration dictionary: {e}")
    except Exception as e:
        raise Exception(f"Error extracting parameter from configuration dictionary: {e}")
    
    if not proofs.valid_data_dir(data_dir):
        message = f"""No subdirectory in '{data_dir}' contains  a JSON file that starts with 'proof' and ends with '.json'.""".format()
        raise ValueError(f"Error: {message}.")
    
    if not ancester_dir_exists(models_dir):
        message = f"""Input '{models_dir}' does not start with a full path."""
        raise ValueError(f"Error: {message}")
    
    if not ancester_dir_exists(tokenizers_dir):
        message = f"""Input '{tokenizers_dir}' does not start with a full path."""
        raise ValueError(f"Error: {message}")
    
    if not ancester_dir_exists(datasets_dir):
        message = f"""Input '{datasets_dir}' does not start with a full path."""
        raise ValueError(f"Error: {message}")
    
    valid_modes = isa_data.FORMATS.values()
    if not data_mode in valid_modes:
        message = f"""Input '{data_mode}' is not a valid data mode: {valid_modes}"""
        raise ValueError(f"Error: {message}")
    
    valid_splits = isa_data.SPLITS.values()
    if not data_split in valid_splits:
        message = f"""Input '{data_split}' is not a valid dataset split: {valid_splits}"""
        raise ValueError(f"Error: {message}")


# CONFIGURATION RETRIEVAL

def get_directory_paths(config_dict:dict, making_dirs=False):
    """
    Returns candidate paths to save tokenizers, datasets, and models from the input configuration. 
    If the directories do not exist and `making_dirs` is True, it creates them.

    :param config_dict: (dict) dictionary containing the training configuration
    :rtype: (str, str, str) tuple
    """
    tokenizers_dir = config_dict["tokenizers_dir"]
    datasets_dir = config_dict["datasets_dir"]
    models_dir = config_dict["models_dir"]
    if making_dirs:
        os.makedirs(tokenizers_dir, exist_ok=True)
        os.makedirs(datasets_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
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

def exists_previous(case_name, dir_path):
    """
    Decides whether there is already a model, dataset, or tokenizer 
    from a previous save-point in the input directory.

    :param config_dict: dictionary containing the training configuration
    :rtype: bool
    """
    def decide(file_path, case_name):
        if os.path.isfile(file_path):
            previous = True
        else:
            previous = False
        message = f"""
        Is there a previous {case_name} in the directory '{dir_path}'?: {previous}.
        Therefore, {case_name} has to be retrieved remotely?: {not previous}
        """
        logging.info(message)
        return previous
    
    file_path = ""
    if case_name == "model":
        file_path = os.path.join(get_latest_dir_from(dir_path), "model.safetensors")
    elif case_name == "dataset":
        file_path = os.path.join(dir_path, "datasets.pt")
    elif case_name == "tokenizer":
        file_path = os.path.join(get_latest_dir_from(dir_path), "tokenizer.json")
    else:
        raise ValueError(f"Unexpected case name {case_name} for exists_previous.")
    
    return decide(file_path, case_name)
    
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
        logging.info(f"Saved Hugging Face data of type {type(hf_data)} in {new_dir}")
    except FileExistsError:
        logging.warning(f"Directory '{new_dir}' already exists. Skipping save.")
    except Exception as e:
        message = f"Failed to save hugging face data: {e}"
        logging.error(message)
        raise Exception(message)


# OPERATING WITH ACCELERATE

def wrap_w_accelerator(f):
    try:
        accelerator = Accelerator(mixed_precision="bf16")
        if accelerator.is_main_process:
            logging.info(f"Accelerator started on {accelerator.num_processes} processes.")
        accelerate_test.log_cuda_info(accelerator)

        # Main body
        f(accelerator)

    except Exception as e:
        logging.error(f"{e}")
        raise e
    finally:
        accelerator.wait_for_everyone()
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                logging.error(f"Error destroying process group: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()