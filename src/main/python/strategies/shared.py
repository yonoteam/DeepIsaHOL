# Maintainers:
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
#
# Shared utilities for evaluation strategies.

import os
import logging


def save_proof(repl, prf):
    """Save a completed proof to a .thy file."""
    thy_name = prf["thy_name"]
    prf_num = prf["num"]
    logic = prf["logic"]

    base_name, _ = os.path.splitext(thy_name)
    filename = os.path.join(logic, f"{base_name}{prf_num}.thy")
    header = f"theory {base_name}{prf_num}\n imports {logic}.{base_name}\n begin\n\n"
    body = repl.last_proof()
    end = "\n\nend"
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + body + end)
        logging.info(f"Saved proof to {filename}")
