# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# A collection of useful methods/operations

import os
import queue
import threading
import time

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

    # Timeout reached
    print("Timeout reached. Function did not complete.")
    return None

def is_potential_full_dir(path):
    """
    Determines if `path` the first two elements of `path` represent an existing directory.

    :param path: a full path.
    """
    if isinstance(path, str):
        splat = [d for d in path.split('/') if d != '']
        _, ending = os.path.splitext(path)
        return len(splat) > 1 and os.path.isdir('/' + '/'.join(splat[:1])) and ending == ''
    return False