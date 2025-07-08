"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Utilities for processing the configuration dictionary describing
the tokenizing, training, and evaluation parameters for the models.
"""

import os
import logging
import argparse
from typing import Union

import dicts
import proofs.data_dir
from proofs.str_ops import FORMATS
from proofs.data_dir import SPLITS

PathLike = Union[str, os.PathLike]

EXAMPLE_TRAINING_ARGS = {
    # operational
    "doc": "https://huggingface.co/docs/transformers/main_classes/trainer#transformers.TrainingArguments",
    "torch_empty_cache_steps": None,
    "use_cpu": False,
    "fp16": False,
    "bf16": True,
    # training
    "max_steps": 1000,
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
    "output_dir": os.getcwd(),
    "overwrite_output_dir": False,
    "save_strategy": "steps",
    "save_total_limit": 5,
    "save_steps": 10000
}

save_hf_tokenizer = "save_hf_tokenizer"
train_hf_tokenizer = "train_hf_tokenizer"
count_dataset = "count_dataset"
pretrain_model = "pretrain_model"
finetune_model = "finetune_model"
eval_model = "eval_model"
TASKS = [save_hf_tokenizer, train_hf_tokenizer, count_dataset, pretrain_model, finetune_model, eval_model]

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
    "hf_train_args": EXAMPLE_TRAINING_ARGS
}

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
        
        data_dir = config_dict["data_dir"]
        _ = config_dict["model_name"]
        models_dir = config_dict["models_dir"]
        tokenizers_dir = config_dict["tokenizers_dir"]
        data_format = config_dict["data_format"]

        train_args = config_dict["hf_train_args"]
        if not isinstance(train_args, dict):
            raise ValueError(f"Input for hf_train_args must be a dictionary.")
    except KeyError as e:
        raise KeyError(f"Error extracting parameter from configuration dictionary: {e}")
    except Exception as e:
        raise Exception(f"Error extracting parameter from configuration dictionary: {e}")
    except ValueError as e:
        raise ValueError(f"Error extracting parameter from configuration dictionary: {e}")

    check_valid_proof_data_dir(data_dir)
    check_has_parent(models_dir)
    check_has_parent(tokenizers_dir)
    check_valid_format(data_format)
    
    if task == pretrain_model or task == finetune_model:
        try:
            _ = int(config_dict["num_epochs"])
            _ = int(train_args["per_device_train_batch_size"])
        except KeyError as e:
            raise KeyError(f"Error extracting parameter from configuration dictionary: {e}")
        except ValueError as e:
            raise ValueError(f"Error extracting parameter from configuration dictionary: {e}")
        except Exception as e:
            raise Exception(f"Error extracting parameter from configuration dictionary: {e}")

    if task == eval_model:
        try:
            _ = int(train_args["per_device_eval_batch_size"])
        except KeyError as e:
            raise KeyError(f"Error extracting parameter from configuration dictionary: {e}")
        except ValueError as e:
            raise ValueError(f"Error extracting parameter from configuration dictionary: {e}")
        except Exception as e:
            raise Exception(f"Error extracting parameter from configuration dictionary: {e}")

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



