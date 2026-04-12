# Maintainers:
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
#
# Hammer strategy for the unified evaluation loop (eval.py).
# Extracted from hammer_eval.py.

import time
import logging

import proofs
from strategies.shared import save_proof


# STRATEGY INTERFACE

def configure(config_dict):
    """Return an eval_config dict for hammer evaluation."""
    # Py4J needs Python tuples to map to Scala Tuple2;
    # JSON only has arrays, so convert [key, value] -> (key, value).
    raw_params = config_dict.get("hammer_params", [])
    hammer_params = [tuple(p) for p in raw_params]
    return {
        "hammer_params": hammer_params,
        "saving": config_dict.get("saving", False),
    }


def init_metrics():
    """Return hammer-specific metric counters."""
    return {
        "successful_hammers": 0,
        "failed_hammers": 0,
    }


def load_prf_info(proof_json, prf_num, prf_path, thy_name, logic, eval_config):
    """Build the per-proof info dict from the raw JSON proof."""
    prf_start = proofs.orig_objective_of(proof_json)
    prf_start = proofs.str_ops.fix_missing_quotations(prf_start)
    return {
        "thy_name": thy_name,
        "path": prf_path,
        "start_line": prf_start,
        "num": prf_num,
        "logic": logic
    }


def attempt_proof(repl, prf_info, eval_config, metrics):
    """Run one hammer proof attempt. REPL is already positioned at the proof start."""
    try:
        start_time = time.time()
        result_msg = repl.call_hammer(eval_config["hammer_params"])
        duration = time.time() - start_time
        metrics["durations"].append(duration)
        progress = result_msg.startswith("Used:")
        finished_proof = progress and not repl.is_at_proof()

        if finished_proof:
            logging.info(f"Hammer FINISHED for {prf_info['path']}: '{result_msg}'")
            metrics["finished_proofs"] += 1
            metrics["successful_hammers"] += 1
            if eval_config.get("saving", False):
                try:
                    save_proof(repl, prf_info)
                except Exception as e:
                    logging.warning(f"Failed to save proof {prf_info['path']}: {e}")
        elif progress:
            logging.info(f"Hammer PROGRESSED for {prf_info['path']}: '{result_msg}'")
            metrics["successful_hammers"] += 1
        else:
            logging.info(f"Hammer FAILED for {prf_info['path']}.")
            metrics["failed_hammers"] += 1
    except Exception as e:
        logging.error(f"Error during hammer attempt: {e}")
    finally:
        repl.reset()
    return metrics
