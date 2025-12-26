# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# A loop evaluating sledgehammer on proofs in the dataset

import os
import time
import logging

import dicts
import proofs
import config_ops
from repl import REPL

# PROCESS ONE PROOF

def measure_hammer(prf_info, loop_state):
    repl = loop_state["repl"]
    metrics = loop_state["counters"]
    metrics["attempted_proofs"] += 1
    try:
        repl.go_to(prf_info["thy_name"], prf_info["start_line"])
        start_time = time.time()
        result_msg = repl.call_hammer(loop_state["hammer_params"])
        duration = time.time() - start_time
        metrics["durations"].append(duration)
        progress = result_msg.startswith("Used:")
        finished_proof = progress and not repl.is_at_proof()
        
        if finished_proof:
            logging.info(f"Hammer FINISHED for {prf_info['path']}: '{result_msg}'")
            metrics["finished_proofs"] += 1
            metrics["successful_hammers"] += 1
        elif progress:
            logging.info(f"Hammer PROGRESSED for {prf_info['path']}: '{result_msg}'")
            metrics["successful_hammers"] += 1
        else:
            logging.info(f"Hammer FAILED for {prf_info['path']}.")
            metrics["failed_hammers"] += 1
            
    except Exception as e:
        logging.error(f"Error during hammer attempt: {e}")
        return loop_state
    finally:
        repl.reset()
    
    return loop_state


# PROCESSS ONE LOGIC

def setup_logic_logging(logic_name):
    """
    Setup logging for a specific logic with a dedicated 
    log file inside its own directory.
    """
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{logic_name}_hammer.log")
    
    # remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        filemode="w",  # overwrite each time the script runs
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def should_skip(proof_dict: dict) -> bool:
    """
    User-editable predicate to skip proofs.
    Example: Skip if proof has more than 5 steps
    """
    if proofs.count_steps(proof_dict) > 5:
        return True
    return False

def reset_counters_for_thy(counters):
    counters["skipped_proofs"] = 0
    counters["attempted_proofs"] = 0
    counters["successful_hammers"] = 0
    counters["failed_hammers"] = 0
    counters["finished_proofs"] = 0
    counters["durations"] = []

def mk_prf_info(proof, prf_path, thy_name):
    prf_start = proofs.orig_objective_of(proof)
    prf_start = proofs.str_ops.fix_missing_quotations(prf_start)
    return {
        "thy_name": thy_name,
        "path": prf_path,
        "start_line": prf_start
    }

def max_attempts_reached(loop_state):
    max_attempts = loop_state.get("max_prf_attempts")
    if max_attempts is None:
        return False

    return loop_state["counters"]["attempted_proofs"] >= max_attempts

def process_logic(logic, thys, loop_state):
    # logic proof counter
    prf_count = 0
    len_proofs = sum(len(thy_proofs) for thy_proofs in thys.values())

    # logic configuration
    setup_logic_logging(logic)
    logging.info(f"Processing logic {logic}")
    if loop_state["repl"]:
        loop_state["repl"].switch_to(logic)
    else:
        loop_state["repl"] = REPL(logic)

    # main loop for logic
    for thy_name, thy_proofs in thys.items():
        if loop_state["max_attempts_reached"]:
            break

        reset_counters_for_thy(loop_state["counters"])
        logging.info(f"Processing theory {thy_name}")
        for _, prf_path in thy_proofs:
            proof = dicts.load_json(prf_path)
            if should_skip(proof):
                loop_state["counters"]["skipped_proofs"] += 1
                logging.info(f"Skipping {prf_path} (predicate returned True)")
                continue
            prf_info = mk_prf_info(
                proof, 
                prf_path, 
                thy_name
            )
            loop_state = measure_hammer(
                prf_info,
                loop_state
            )
            prf_count += 1
            logging.info(f"Processed proof {prf_count} of {len_proofs} for logic '{logic}': {prf_path}.\n\n")
            if max_attempts_reached(loop_state):
                loop_state["max_attempts_reached"] = True
                dicts.update_records(loop_state["counters"], "hammer_metrics.json")
                break
        dicts.update_records(loop_state["counters"], "hammer_metrics.json")
    mssg = f"Processed all theories in {logic} or processed {loop_state['counters']['attempted_proofs']} proofs out of {loop_state['max_prf_attempts']}."
    print(mssg)
    logging.info(mssg)
    return loop_state

# PROCESS ALL LOGICS

def init_loop_state(config_dict):
    return {
        "max_prf_attempts": config_dict.get("max_prf_attempts", None),
        "max_attempts_reached": False,
        "repl": None,
        "hammer_params": config_dict.get("hammer_params", []),
        "counters": {
            "skipped_proofs": 0,
            "attempted_proofs": 0,
            "finished_proofs": 0,
            "successful_hammers": 0,
            "failed_hammers": 0,
            "durations": []
        }
    }

def eval_hammer_on_logics(config_dict):
    loop_state = init_loop_state(config_dict)
    progress_file = config_ops.create_progress_file(file_name="hammer_progress.txt")
    logics_dict = proofs.data_dir.group_paths_by_logic(
        config_dict["data_dir"], 
        config_dict["data_split"]
    )
    try:
        for logic in logics_dict.keys():
            if loop_state["max_attempts_reached"]:
                print("Max attempts reached, stopping process.")
                break
            if config_ops.progress_item_in(logic, progress_file):
                print(f"Skipping already processed logic: {logic}")
                continue
            thys = logics_dict[logic]
            loop_state = process_logic(logic, thys, loop_state)
    except Exception as e:
        logging.warning(f"Error processing {logic}: {e}")
    finally:
        if loop_state["repl"]:
            loop_state["repl"].shutdown_gateway()

if __name__ == "__main__":
    info = "Evaluates Isabelle's Sledgehammer on proofs in the dataset."
    config = config_ops.parse_path(tool_explanation=info)
    eval_hammer_on_logics(config)