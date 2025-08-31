"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Utilities for retrieving information from the generated data-directory
of "proof*.json" files.
"""

import os
import re
import json

from tqdm import tqdm
from pathlib import Path
from typing import Union, List, Iterator, Callable

import dicts

PathLike = Union[str, os.PathLike]

def is_valid(json_data_dir: PathLike) -> bool:
    """
    Checks if the input is a directory with at least one JSON file whose 
    name starts with 'proof' and ends with '.json'.

    :param json_data_dir: path to the directory to validate
    :return: true if a valid JSON file is found, false otherwise.
    """
    data_path = Path(json_data_dir)
    if not data_path.exists() or not data_path.is_dir():
        return False

    for json_file in data_path.rglob("proof*.json"):
        if json_file.is_file():
            return True
    return False

def generate_paths(json_data_dir: PathLike) -> Iterator[str]:
    """
    Generates all file-paths in the input directory that end in "proof*.json"

    :param json_data_dir: path to the data directory with 'proof*.json's
    :yield: full path of each "proof*.json"
    """
    for subdir, _, files in os.walk(json_data_dir):
        for file in sorted(files):
            if file.startswith("proof") and file.endswith(".json"):
                yield os.path.join(subdir, file)

def get_paths(json_data_dir: PathLike) -> List[str]:
    """
    Returns an ordered list of all file-paths in the input directory 
    that end in "proof*.json"

    :param json_data_dir: path to the data directory with 'proofN.json's
    :return: sorted list of paths to JSON files
    """
    return sorted(generate_paths(json_data_dir))

def find_erroneous(
        json_data_dir: PathLike,
        f: Callable[[dict], object] = lambda x: x
    ) -> List[str]:
    """
    Identifies "proof*.json" files in the input directory that fail to load or
    fail during `f(data)`, storing the failing file paths in the returned list.

    :param json_data_dir: path to the data directory with 'proof*.json's
    :param f: function to apply to each JSON file
    """
    erroneous_paths = []
    for file_path in generate_paths(json_data_dir):
        try:
            with open(file_path, "r") as json_file:
                data = json.load(json_file)
            f(data)
        except Exception as e:
            erroneous_paths.append(file_path)
    return erroneous_paths

def fix_erroneous(
        json_data_dir: PathLike
    ) -> tuple[list[str], list[tuple[str,str]]]:
    """
    Attempts to fix all erroneous "proof*.json" in the input directory.

    :param json_data_dir: path to the data directory with 'proof*.json's
    :return: a tuple with:
        - a list of successfully fixed file paths
        - a list of the path and the error for the unsuccessfully fixed
    """
    fixed_files = []
    failed_files = []
    for file_path in find_erroneous(json_data_dir):
        opt_str = dicts.fix_json_line_breaks_at(Path(file_path), backing_up=True)
        if opt_str is None:
            fixed_files.append(file_path)
        else:
            failed_files.append((file_path, opt_str))
    return fixed_files, failed_files

def delete_erroneous(
        json_data_dir: PathLike
    ) -> tuple[list[str], list[tuple[str, Exception]]]:
    """
    Deletes all erroneous "proof*.json" in the input directory.

    :param json_data_dir: path to the data directory with 'proof*.json's
    :return: a tuple with:
        - a list of successfully deleted file paths
        - a list of (file path, exception) pairs for failures
    """
    deleted_files = []
    failed_files = []
    for file_path in find_erroneous(json_data_dir):
        try:
            os.remove(file_path)
            deleted_files.append(file_path)
        except Exception as e:
            path_reason = file_path, e
            failed_files.append(path_reason)
    return deleted_files, failed_files

def generate_from(json_data_dir: PathLike) -> Iterator[dict]:
    """
    Yields a dictionary for each "proof*.json" file in the input directory.
    It assumes that each erroneous file has been deleted.

    :param json_data_dir: path to the data directory with 'proof*.json's
    :yield: parsed dictionary from the 'proof*.json' representing a proof
    """
    for proof_path in generate_paths(json_data_dir):
        proof = dicts.load_json(proof_path)
        yield proof

def apply(
        f: Callable[[object, dict], object],
        acc: object,
        json_data_dir: PathLike
    ) -> object:
    """
    Applies the function f to each "proof*.json" file in the input directory
    while accumulating the partial results in the accumulator "acc".

    :param f: function of type (S x dict) -> S to apply to each proof (dict)
    :param acc: initial state of the accumulator (of type S)
    :param json_data_dir: path to the data directory with 'proof*.json's
    :return: final accumulated results of type S
    """
    results = acc
    for proof in tqdm(generate_from(json_data_dir), desc="Processing proofs", unit="proof"):
        results = f(results, proof)
    return results

SPLITS = {
    "TRAIN": "train", 
    "VALID": "valid", 
    "TEST": "test", 
    "NONE": "none"
}

def generate_dataset_paths(
        json_data_dir: PathLike, 
        split:str = SPLITS["NONE"]
    ) -> Iterator[str]:
    """
    Yields file paths to 'proof*.json' files in the input directory 
    according to a specified dataset split (train/valid/test/none).

    If 'none' is specified (default), yields all proof files. 
    Otherwise, yields a split: 64% train, 16% valid, and 20% test.

    :param json_data_dir: path to the data directory with 'proof*.json's
    :param split: one of the values of the SPLITS dictionary
    :yield: full path to a proof file from the dataset split
    :raises Exception: if the directory is invalid or split is unrecognized
    """
    if not is_valid(json_data_dir):
        raise Exception(f"Error: bad input {json_data_dir} is not an existing directory or does not contain a proofN.json.")

    for subdir, _, files in os.walk(json_data_dir):
        json_files = [file for file in files if file.startswith("proof") and file.endswith(".json")]
        json_files = [os.path.join(subdir, file) for file in sorted(json_files)]

        if split == SPLITS["NONE"]:
            yield from json_files
        else:
            total_proofs = len(json_files)
            train_size = int(total_proofs * 0.64)
            valid_size = int(total_proofs * 0.16)
            if split == SPLITS["TRAIN"]:
                train_split = json_files[:train_size]
                yield from train_split
            elif split == SPLITS["VALID"]:
                valid_split = json_files[train_size:train_size + valid_size]
                yield from valid_split
            elif split == SPLITS["TEST"]:
                test_split = json_files[train_size + valid_size:]
                yield from test_split
            else:
                raise Exception(f"Error: bad input {split}, expected one of {SPLITS.values()}.")
            
def group_paths_by_logic(
        json_data_dir: PathLike, 
        split:str = SPLITS["NONE"]
    ) -> dict[str, dict[str, list[tuple[int, str]]]]:
    """
    Groups proof file paths by logic and theory name in a dictionary. It assumes the
    generated data-directory structure, where each logic corresponds to a top-level 
    subdirectory and each theory groups proofs by parent directory.

    The function yields a nested dictionary of the form:
        { logic_dir: { theory_file: [(proof_num, proof_path), ...] } }

    The proof numbers are extracted from the 'proof*.json' and used to sort each 
    group numerically.

    :param json_data_dir: path to the data directory with 'proof*.json's
    :param split: one of the values of the SPLITS dictionary
    :return: nested dictionary grouping proof paths by logic and theory file
    """
    logics_dict = {}
    proof_pattern = re.compile(r'proof(\d+)\.json$')

    for path in generate_dataset_paths(json_data_dir, split=split):
        logic_path = os.path.relpath(path, json_data_dir)
        path_parts = logic_path.split(os.sep)

        logic = path_parts[0]
        if not proof_pattern.search(path_parts[-1]):  
            continue
        
        thy_name = f"{path_parts[-2]}.thy"  

        proof_num = int(proof_pattern.search(path_parts[-1]).group(1))  

        if logic not in logics_dict:
            logics_dict[logic] = {}

        if thy_name not in logics_dict[logic]:
            logics_dict[logic][thy_name] = []

        logics_dict[logic][thy_name].append((proof_num, path))

    # sort numerically
    for logic in logics_dict:
        for thy_name in logics_dict[logic]:
            logics_dict[logic][thy_name].sort()

    return logics_dict