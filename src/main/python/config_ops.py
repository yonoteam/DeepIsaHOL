
import os
import logging
import argparse

import proofs.data_dir
from proofs.str_ops import FORMATS
from proofs.data_dir import SPLITS

CTX_LENGTHS = {
    FORMATS["S"]: 512,
    FORMATS["SP"]: 1024,
    FORMATS["SPK"]: 1024,
    FORMATS["SPKT"]: 1024,
    FORMATS["FS"]: 512,
    FORMATS["FSP"]: 1024,
    FORMATS["FSPK"]: 1024,
    FORMATS["FSPKT"]: 1024
}

def get_context_length(mode):
    return CTX_LENGTHS.get(mode)

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
    
    if not proofs.data_dir.is_valid(data_dir):
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
    
    valid_modes = FORMATS.values()
    if not data_mode in valid_modes:
        message = f"""Input '{data_mode}' is not a valid data mode: {valid_modes}"""
        raise ValueError(f"Error: {message}")
    
    valid_splits = SPLITS.values()
    if not data_split in valid_splits:
        message = f"""Input '{data_split}' is not a valid dataset split: {valid_splits}"""
        raise ValueError(f"Error: {message}")
    

def setup_logging(log_file_name, log_level=logging.DEBUG, save_dir=os.getcwd()):
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