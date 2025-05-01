# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Dictionary utilities

import os
import json
import logging

from typing import Union, Optional, List, IO, Any

def safe_load_json(file_obj: IO[str]) -> dict:
    """
    Loads a JSON from an open file object.
    Returns an empty dictionary on failure.

    :param file_obj: open file object
    """
    try:
        return json.load(file_obj)
    except json.JSONDecodeError:
        return {}

def load_json(json_path: Union[str, os.PathLike]) -> dict:
    """
    Loads a dictionary from a JSON file.
    Returns an empty dictionary on failure.

    :param json_path: path to the JSON file
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
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON from '{json_path}': {e}")
    except OSError as e:
        logging.error(f"OS error occurred: {e}")
    except Exception as e:
        logging.error(f"An unexpected error occurred: {e}")
    return result

def replace_in_opened(data: dict, file: IO[str], indent: int = 4) -> None:
    """
    Replaces a file's contents with the input dictionary.
    The file must be opened in read/write mode.

    :param data: the dictionary to write
    :param file: open file object
    :param indent: indentation level for pretty-printing (default: 4)
    """
    file.seek(0)
    json.dump(data, file, indent=indent)
    file.truncate()

def save_as_json(data: dict, save_path: Union[str, os.PathLike], indent=4) -> None:
    """
    Saves a dictionary as a JSON file.

    :param data: dictionary to save
    :param save_path: path to the output JSON file
    :param indent: indentation level for pretty-printing (default: 4)
    """
    try:
        with open(save_path, 'w') as json_file:
            json.dump(data, json_file, indent=indent)
    except Exception as e:
        logging.error(f"Error saving dictionary to JSON '{save_path}': {e}")

def get_keys(d: dict, prefix: str="") -> List[str]:
    """Recursively finds all key paths in the dictionary d

    :param d: dictionary to extract keys from
    :param prefix: prefix to prepend to each key
    :returns: list of keys preppended with prefix
    """
    keys = []
    for key, value in d.items():
        new_prefix = f"{prefix}{key}"
        if isinstance(value, dict):
            keys.extend(get_keys(value, prefix + key + "."))
        elif isinstance(value, list):
            if value and isinstance(value[0], dict):
                keys.extend(get_keys(value[0], new_prefix + "[]."))
            else:
                keys.append(new_prefix + "[]")
        else:
            keys.append(new_prefix)
    return keys

def to_string(
        d: dict, 
        max_chars: Optional[int] = None, 
        indent: int = 0, 
        _current_len: int = 0
        ) -> str:
    """
    Converts a dictionary into a readable string format,
    handling nested dictionaries and lists safely and efficiently.
    Stops building output early if max_chars is reached.

    :param d: dictionary to turn into a string
    :param max_chars: maximum characters in the output (None = all)
    :param indent: indentation level (for internal recursion)
    :param _current_len: current accumulated string length (for internal recursion)
    :return: human-readable string representation of the dictionary
    """
    lines = []
    cutoff = False

    for key, value in d.items():
        prefix = "  " * indent + str(key) + ": "
        try:
            if isinstance(value, dict):
                line = prefix
                lines.append(line)
                _current_len += len(line) + 1
                if max_chars is not None and _current_len >= max_chars:
                    cutoff = True
                    break

                nested = to_string(
                    value, 
                    max_chars=max_chars, 
                    indent=indent + 1, 
                    _current_len=_current_len
                    )
                lines.append(nested)
                _current_len += len(nested)
                if max_chars is not None and _current_len >= max_chars:
                    cutoff = True
                    break

            elif isinstance(value, list) and all(isinstance(i, dict) for i in value):
                lines.append(prefix + "[")
                _current_len += len(prefix) + 2
                if max_chars is not None and _current_len >= max_chars:
                    cutoff = True
                    break

                for item in value:
                    nested = to_string(
                        item, 
                        max_chars=max_chars, 
                        indent=indent + 2, 
                        _current_len=_current_len
                        )
                    lines.append(nested)
                    _current_len += len(nested)
                    if max_chars is not None and _current_len >= max_chars:
                        cutoff = True
                        break
                lines.append("  " * (indent + 1) + "]")

            else:
                try:
                    value_repr = repr(value)
                except Exception as e:
                    value_repr = f"<unrepresentable: {e}>"
                line = prefix + value_repr
                lines.append(line)
                _current_len += len(line) + 1
                if max_chars is not None and _current_len >= max_chars:
                    cutoff = True
                    break

        except Exception as e:
            # unknown structure
            line = prefix + f"<error processing value: {e}>"
            lines.append(line)
            _current_len += len(line) + 1
            if max_chars is not None and _current_len >= max_chars:
                cutoff = True
                break

    output = "\n".join(lines)
    if cutoff and max_chars is not None:
        output = output[:max_chars].rstrip() + "\n..."
    return output


def printd(d: dict, max_chars: Optional[int] = None) -> None:
    """
    Pretty-prints a dictionary to the console.

    :param d: dictionary to print
    :param max_chars: maximum number of characters to print (None = all)
    """
    print(to_string(d, max_chars=max_chars))



