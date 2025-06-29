"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Frequently used methods for saving tokenizers, datasets, and models.
"""

import os
import logging

def get_dirs(config_dict:dict, making_dirs=False):
    """
    Returns candidate paths to save tokenizers, datasets, and models from the input configuration. 
    If the directories do not exist and `making_dirs` is True, it creates them.

    :param config_dict: (dict) dictionary containing the training configuration
    :rtype: (str, str, str) tuple
    """
    tokenizers_dir = config_dict["tokenizers_dir"]
    models_dir = config_dict["models_dir"]
    if making_dirs:
        os.makedirs(tokenizers_dir, exist_ok=True)
        os.makedirs(models_dir, exist_ok=True)
    return tokenizers_dir, models_dir

def get_latest_dir(saving_dir, adding_one=False):
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
        file_path = os.path.join(get_latest_dir(dir_path), "model.safetensors")
    elif case_name == "dataset":
        file_path = os.path.join(dir_path, "datasets.pt")
    elif case_name == "tokenizer":
        file_path = os.path.join(get_latest_dir(dir_path), "tokenizer.json")
    else:
        raise ValueError(f"Unexpected case name {case_name} for exists_previous.")
    
    return decide(file_path, case_name)

def save_in(hf_data, saving_dir):
    """
    Saves the input tokenizer or model in the latest number-named subdirectory +1.

    :param hf_data: either a Hugging Face tokenizer or model
    :param saving_dir: (str) the directory with possibly number-named subdirectories
    """
    try:
        os.makedirs(saving_dir, exist_ok=True)
        new_dir = get_latest_dir(saving_dir, adding_one=True)
        os.makedirs(new_dir, exist_ok=False)
        hf_data.save_pretrained(new_dir)
        logging.info(f"Saved Hugging Face data of type {type(hf_data)} in {new_dir}")
    except FileExistsError:
        logging.warning(f"Directory '{new_dir}' already exists. Skipping save.")
    except Exception as e:
        message = f"Failed to save hugging face data: {e}"
        logging.error(message)
        raise Exception(message)

