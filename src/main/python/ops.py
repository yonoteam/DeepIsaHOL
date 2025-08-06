# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# A collection of reused methods/operations

import os
import re
import time
import queue
import psutil
import shutil
import random
import logging
import threading

from pathlib import Path
from datetime import datetime
from typing import Union, Iterator, TypeVar, Optional, List
T = TypeVar("T")

import matplotlib.pyplot as plt

import dicts

def log_memory_usage(message):
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024  # MB
    logging.info(f"Memory usage ({message}): {memory:.2f} MB")

def random_element(iterator: Iterator[T], max: Optional[int] = None) -> T:
    """
    Returns a random element from the iterator.

    - If max is provided, it picks a random index from [0, max-1] and returns the element at that index, or the last element if the iterator ends earlier.
    - If max is None, it selects an element uniformly at random from the entire iterator.

    :param iterator: any iterator (e.g., generator or iterable converted with iter())
    :param max: max iteration-size
    :return: pseudo-randomly selected element
    :raises ValueError: if the iterator is empty
    """
    if max is not None:
        rand_index = random.randint(0, max - 1)
        last = None
        for i, element in enumerate(iterator):
            last = element
            if i == rand_index:
                return element
        if last is not None:
            return last
        raise ValueError("Iterator is empty.")
    else:
        chosen = None
        for i, element in enumerate(iterator):
            if i == 0 or random.random() < 1.0 / (i + 1):
                chosen = element
        if chosen is not None:
            return chosen
        raise ValueError("Iterator is empty.")

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

def to_plot(read_json_path:str, loop_name:str, start_from:int=0):
    read_dir = os.path.dirname(read_json_path)
    metric_dict = dicts.load_json(read_json_path)
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

def save_plot(
        x_vals, 
        y_vals, 
        save_path: Union[str, os.PathLike]="curve.png",
        title="Graph of y vs x", x_label="X-axis", y_label="Y-axis"
    ) -> None:
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

def save_tuple_list_as_txt(
        list_of_tuples,
        save_path: Union[str, os.PathLike]
    ) -> None:
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

# COPY DATA

def copy_files_with_structure(file_paths: List[str], destination: str) -> None:
    """
    Copies files to a destination directory, preserving their relative paths
    from their common root.

    :param file_paths: List of file paths (as strings) to copy.
    :param destination: Root directory where files will be copied.
    """
    path_objs = [Path(p) for p in file_paths]
    
    common_root = Path(os.path.commonpath(path_objs))
    destin_root = Path(destination)
    destin_root.mkdir(parents=True, exist_ok=True)

    for file_path in path_objs:
        if not file_path.is_file():
            logging.info(f"Skipping {file_path}: not a file.")
            continue
        relative_path = file_path.relative_to(common_root)
        destination_path = destin_root / relative_path
        destination_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(file_path, destination_path)

