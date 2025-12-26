"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Utilities for processing the configuration dictionary describing
the tokenizing, training, and evaluation parameters for the models.
"""

import os
import fcntl
import logging
import argparse

import dicts
import proofs.data_dir
from proofs.str_ops import FORMATS
from proofs.data_dir import SPLITS

EXAMPLE_TRAINING_ARGS = {
    # operational
    "doc": "https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments",
    "torch_empty_cache_steps": None,
    "use_cpu": False,
    "fp16": False,                       # computed automatically
    "bf16": True,                        # computed automatically
    # training
    "max_steps": 1000,                   # computed automatically
    "max_grad_norm": 0.3,
    "warmup_ratio": 0.03,
    "gradient_checkpointing": False,
    "gradient_accumulation_steps": 1,
    "per_device_train_batch_size": 8,
    # sft-specific args
    "packing": False,
    "max_seq_length": 2048,
    "optim": "adamw_torch_fused",
    # logging
    "logging_strategy": "steps",
    "log_level": "passive",
    "report_to": "tensorboard",
    "logging_dir": os.getcwd(),
    "log_on_each_node": True,
    "logging_steps": 1000,
    # evaluation
    "eval_strategy": "steps",
    "eval_steps": 1000,
    "load_best_model_at_end": True,
    "metric_for_best_model": "eval_loss",
    "per_device_eval_batch_size": 8,
    # training optimization
    "lr_scheduler_type": "constant",
    "learning_rate": 1e-5,
    "weight_decay": 0.01,
    # saving
    "output_dir": os.getcwd(),           # computed automatically
    "overwrite_output_dir": False,
    "save_strategy": "steps",
    "save_total_limit": 5,
    "save_steps": 10000
}

GENERATION_CONFIG = {
    "gen_length": 64,
    "num_return_sequences": 5,
    "num_beams": 5,
    "allowed_depth": 5,
    "proof_timeout_seconds": 30
}

DFS_CONFIG = {
    "device": -1,
    "saving": False,
    "max_prf_attempts": 5
}

def get_device_str(config_dict):
    device_int = config_dict["dfs_config"]["device"]
    if device_int == -1:
        return "cpu"
    else:
        return f"cuda:{device_int}"

# TASKS
save_hf_tokenizer = "save_hf_tokenizer"
train_hf_tokenizer = "train_hf_tokenizer"
count_dataset = "count_dataset"
pretrain_model = "pretrain_model"
finetune_model = "finetune_model"
eval_model = "eval_model"
dfs_eval = "dfs_eval"
deploy_server = "deploy_server"

EXAMPLE_CONFIG_DICT = {
    "task": pretrain_model,
    "data_dir": os.getcwd(),
    "model_name": "google/flan-t5-small",
    "models_dir": os.getcwd(),
    "tokenizers_dir": os.getcwd(),
    "data_format": "state",
    "data_split": "train",
    "float_type": "bf16",
    "num_epochs": 1,
    "batches_per_epoch": 134740,
    "hf_train_args": EXAMPLE_TRAINING_ARGS,
    "generation_config": GENERATION_CONFIG,
    "dfs_config": DFS_CONFIG
}

def get_torch_float_type(float_type_str):
    import torch
    torch_type_mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,  
        "float16": torch.float16,
        "half": torch.float16,  
        "fp16": torch.float16,  
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16, 
        "tf32": torch.float32
    }
    float_type_str_lower = float_type_str.lower()
    if float_type_str_lower in torch_type_mapping:
        return torch_type_mapping[float_type_str_lower]
    else:
        raise ValueError(
            f"Unknown float type '{float_type_str}'. "
            f"Expected one of: {list(torch_type_mapping.keys())}"
        )
    
def create_progress_file(file_name: str = "progress.txt") -> str:
    progress_file = file_name
    if not os.path.exists(progress_file):
        with open(progress_file, "w", encoding="utf-8"):
            pass
    return progress_file

def progress_item_in(item, progress_file):
    # "progress" used both as adjective and verb
    # made in this way for concurrency safety
    with open(progress_file, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            completed_items = {line.strip() for line in f}
            if item in completed_items:
                return True
            
            f.seek(0, os.SEEK_END) # go to end of file
            f.write(item + "\n")
            f.flush()              # ensure buffer written to OS
            os.fsync(f.fileno())   # ensure OS writes to disk
            return False
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

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

def check_has_parent(path):
    if not ancester_dir_exists(path):
        message = f"""Input '{path}' does not start with a full path."""
        raise ValueError(f"Error: {message}")

def check_valid_proof_data_dir(data_dir):
    if not proofs.data_dir.is_valid(data_dir):
        message = f"""No subdirectory in '{data_dir}' contains a JSON file that starts with 'proof' and ends with '.json'.""".format()
        raise ValueError(f"Error: {message}.")
    
def check_valid_format(data_format):
    valid_formats = FORMATS.values()
    if not data_format in valid_formats:
        message = f"""Input '{data_format}' is not a valid data mode: {valid_formats}"""
        raise ValueError(f"Error: {message}")

def check_valid_split(data_split):
    valid_splits = SPLITS.values()
    if not data_split in valid_splits:
        message = f"""Input '{data_split}' is not a valid dataset split: {valid_splits}"""
        raise ValueError(f"Error: {message}")

def check_params(config_dict):
    """
    Raises an error if it finds an issue with the input configuration dictionary. 
    Otherwise, it does nothing.

    :param config_dict: (dict) dictionary containing the training configuration
    """
    try:
        task = config_dict["task"]
        if not isinstance(task, str):
            raise ValueError(f"Input for task must be a string")

    except KeyError as e:
        raise KeyError(f"Error extracting parameter from configuration dictionary: {e}")
    except Exception as e:
        raise Exception(f"Error extracting parameter from configuration dictionary: {e}")
    except ValueError as e:
        raise ValueError(f"Error extracting parameter from configuration dictionary: {e}")
    
    if task == pretrain_model or task == finetune_model:
        try:
            _ = config_dict["model_name"]
            data_format = config_dict["data_format"]
            data_dir = config_dict["data_dir"]
            models_dir = config_dict["models_dir"]
            tokenizers_dir = config_dict["tokenizers_dir"]
            train_args = config_dict["hf_train_args"]
            if not isinstance(train_args, dict):
                raise ValueError(f"Input for hf_train_args must be a dictionary.")
            _ = int(config_dict["num_epochs"])
            _ = int(train_args["per_device_train_batch_size"])
        except KeyError as e:
            raise KeyError(f"Error extracting parameter from configuration dictionary: {e}")
        except ValueError as e:
            raise ValueError(f"Error extracting parameter from configuration dictionary: {e}")
        except Exception as e:
            raise Exception(f"Error extracting parameter from configuration dictionary: {e}")

        check_valid_proof_data_dir(data_dir)
        check_has_parent(models_dir)
        check_has_parent(tokenizers_dir)
        check_valid_format(data_format)
    if task == eval_model:
        try:
            _ = config_dict["model_name"]
            data_format = config_dict["data_format"]
            data_dir = config_dict["data_dir"]
            models_dir = config_dict["models_dir"]
            tokenizers_dir = config_dict["tokenizers_dir"]
            train_args = config_dict["hf_train_args"]
            if not isinstance(train_args, dict):
                raise ValueError(f"Input for hf_train_args must be a dictionary.")
            _ = int(train_args["per_device_eval_batch_size"])
        except KeyError as e:
            raise KeyError(f"Error extracting parameter from configuration dictionary: {e}")
        except ValueError as e:
            raise ValueError(f"Error extracting parameter from configuration dictionary: {e}")
        except Exception as e:
            raise Exception(f"Error extracting parameter from configuration dictionary: {e}")
        
        check_valid_proof_data_dir(data_dir)
        check_has_parent(models_dir)
        check_has_parent(tokenizers_dir)
        check_valid_format(data_format)

def parse_path(
        tool_explanation: str = "Does something as specified in the config path."
    ):
    """
    Entrypoint method to parse the first stdin-argument to a script. The stdin input should be 
    a path to the configuration dictionary. It offers a customisable explanation argument for 
    describing the purpose of the scripts calling it.
    
    :param tool_explanation: Description of what the main function calling this method will do.
    """
    try:
        parser = argparse.ArgumentParser(description=tool_explanation)
        parser.add_argument("config_path", type=str, help="Path to the JSON configuration file.")
        args = parser.parse_args()
        path = args.config_path
        if not ancester_dir_exists(path):
            raise ValueError(f"Not a path: {path}")
        config_dict = dicts.load_json(path)
        check_params(config_dict)
        return config_dict
    except Exception as e:
        raise Exception(f"Error loading configuration information: {e}")
    

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
