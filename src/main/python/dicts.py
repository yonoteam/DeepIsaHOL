# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Dictionary utilities

import os
import re
import json
import fcntl
import logging

from pathlib import Path
from typing import Union, Optional, List, IO, Any
from collections.abc import MutableMapping, MutableSequence

PathLike = Union[str, os.PathLike]

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

def clean_linebreaks_in_quoted_strings(text: str) -> str:
    def replace_new_line(regex_match):
        s = regex_match.group(0)
        return s.replace('\n', '\\n')

    return re.sub(r'"(?:[^"\\]|\\.)*?"', replace_new_line, text, flags=re.DOTALL)

def fix_missing_quotations(s: str) -> str:
    """
    Fixes strings with an odd number of quotation marks by appending one.

    :param s: input string
    """
    double_quotes = s.count('"') % 2
    if double_quotes:
        s += '"'
    return s

def _recursively_fix_string_quotations(data: Any) -> Any:
    """
    Recursively applies fix_missing_quotations to all string values 
    in a dictionary or list.
    """
    if isinstance(data, str):
        return fix_missing_quotations(data)
    elif isinstance(data, MutableMapping): # Dictionaries
        return {k: _recursively_fix_string_quotations(v) for k, v in data.items()}
    elif isinstance(data, MutableSequence): # Lists/Arrays
        return [_recursively_fix_string_quotations(item) for item in data]
    else: # int, float, bool, None
        return data

def fix_json_line_breaks(text: str) -> Optional[str]:
    cleaned = clean_linebreaks_in_quoted_strings(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as e:
        logging.error(f"Error fixing JSONs line breaks: {e}")
        return None
    
    fixed_json = _recursively_fix_string_quotations(parsed)
    try:
        json.dumps(fixed_json)
    except Exception as e:
        logging.error(f"Error fixing JSONs quotes: {e}")
        return None
    
    return fixed_json

def fix_json_line_breaks_at(json_path: Path, backing_up: bool = True) -> Optional[str]:
    """
    Attempts to fix improper line breaks, i.e. `\n`, in a JSON file.
    If successful, it saves the corrected version.

    :param json_path: Path to the JSON file.
    :param backing_up: Whether to save a backup of the original file.
    :returns: Error message if fix fails, or None if successful.
    """
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = f.read()
        json.loads(raw)
        return None # already valid
    except json.JSONDecodeError:
        pass # proceed to attempt fixing

    opt_fixed = fix_json_line_breaks(raw)

    if opt_fixed:
        if backing_up:
            json_path.with_suffix(json_path.suffix + ".bak").write_text(raw, encoding="utf-8")
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(opt_fixed, f, indent=2, ensure_ascii=False)
        return None
    return f"Failed to fix {json_path}."

def replace_in_opened(data: dict, file: IO[str], indent: int = 4) -> None:
    """
    Replaces a file's contents with the input dictionary.
    The file must be opened in read/write mode.

    :param data: the dictionary to write
    :param file: open file object
    :param indent: indentation level for pretty-printing (default: 4)
    """
    file.seek(0)
    file.truncate()
    json.dump(data, file, indent=indent)

def save_as_json(
        data: dict, 
        save_path: PathLike,
        indent=4
    ) -> None:
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

def synch_save(
        data: dict, 
        save_path: PathLike,
        indent=4
    ) -> None:
    """
    Synchronously replaces the content of the JSON file with the 
    provided metrics dictionary using an exclusive lock.
    """
    # open in 'w' mode would truncate immediately, 
    # so we open in 'r+' or 'a+' to lock before clearing.
    mode = "r+" if os.path.exists(save_path) else "w+"
    with open(save_path, mode) as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            f.seek(0)
            json.dump(data, f, indent=indent)
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def update_records(
        new_metrics: dict, 
        filename: PathLike="records.json"
    ) -> None:
    """
    Updates a JSON file with new metrics by aggregating numerical values
    and extending lists. Non-numeric values are overwritten.

    :param new_metrics: dictionary with new metrics to update
    :param filename: path to the JSON file to update
    """
    if not os.path.exists(filename):
        save_as_json({}, filename)
    with open(filename, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX) # exclusive lock on file
        try:
            curr_data = safe_load_json(f)
            for key, value in new_metrics.items():
                if isinstance(value, float):
                    curr_data[key] = curr_data.get(key, 0) + value
                if isinstance(value, int):
                    curr_data[key] = curr_data.get(key, 0) + value
                elif isinstance(value, list):
                    curr_data.setdefault(key, []).extend(value)
                else:
                    # TODO: decide other non-numerics handling if added
                    curr_data[key] = value 
            replace_in_opened(curr_data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

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



