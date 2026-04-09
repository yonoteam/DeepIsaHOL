# Maintainers:
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
#
# Unified evaluation loop for DFS and Hammer strategies.
# Replaces dfs.py and hammer_eval.py.

import os
import sys
import signal
import logging

import dicts
import proofs
import config_ops
from strategies import STRATEGIES


# SHARED SKIP PREDICATE

def should_skip(proof_json, eval_config):
    """Shared skip predicate applied before any strategy-specific work.

    Configurable via "max_proof_steps" in the config. If absent, no skipping.
    """
    max_steps = eval_config.get("max_proof_steps")
    if max_steps is not None and proofs.count_steps(proof_json) > max_steps:
        return True
    return False


# COMMON BASE METRICS

def init_base_metrics():
    """Counters that every strategy shares, enabling direct comparison."""
    return {
        "skipped_proofs": 0,
        "attempted_proofs": 0,
        "trivially_solved": 0,
        "finished_proofs": 0,
        "durations": []
    }


# SHARED FUNCTIONS

def setup_logic_logging(logic_name, task_label):
    log_dir = os.path.join(os.getcwd(), "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f"{logic_name}_{task_label}.log")
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    logging.basicConfig(
        filename=log_file,
        filemode="w",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )


def max_attempts_reached(loop_state):
    max_attempts = loop_state.get("max_prf_attempts")
    if max_attempts is None:
        return False
    return loop_state["prf_attempts_count"] >= max_attempts


# PROCESS ONE LOGIC

def process_logic(logic, thys, eval_config, loop_state, strategy):
    prf_count = 0
    len_proofs = sum(len(thy_proofs) for thy_proofs in thys.values())
    task_label = eval_config.get("task_label", "eval")
    records_file = eval_config.get("records_file", "eval_records.json")

    setup_logic_logging(logic, task_label)
    logging.info(f"Processing logic {logic}")

    from repl import REPL

    repl = loop_state["repl"]
    if repl:
        repl.switch_to(logic)
    else:
        repl = REPL(logic)
    loop_state["repl"] = repl

    for thy_name, thy_proofs in thys.items():
        if loop_state["max_attempts_reached"]:
            break

        # per-theory metrics: base + strategy-specific
        metrics = init_base_metrics()
        metrics.update(strategy.init_metrics())
        logging.info(f"Processing theory {thy_name}")

        for prf_num, prf_path in thy_proofs:
            proof_json = dicts.load_json(prf_path)

            # shared skip predicate
            if should_skip(proof_json, eval_config):
                metrics["skipped_proofs"] += 1
                prf_count += 1
                logging.info(f"Skipping {prf_path} (skip predicate)")
                continue

            prf_info = strategy.load_prf_info(
                proof_json, prf_num, prf_path, thy_name, logic, eval_config
            )
            logging.info(f"Loaded proof at {prf_path}")

            # shared navigation
            try:
                repl.go_to(prf_info["thy_name"], prf_info["start_line"])
            except Exception as e:
                logging.warning(f"Could not navigate to {prf_path}: {e}")
                prf_count += 1
                continue

            # shared trivially-solved check
            if repl.is_at_proof() and repl.without_subgoals():
                metrics["trivially_solved"] += 1
                logging.info(f"Trivially solved: {prf_path}")
                repl.reset()
                prf_count += 1
                continue

            # actual attempt (strategy-specific)
            metrics["attempted_proofs"] += 1
            metrics = strategy.attempt_proof(repl, prf_info, eval_config, metrics)

            prf_count += 1
            loop_state["prf_attempts_count"] += 1
            logging.info(
                f"Processed proof {prf_count} of {len_proofs} "
                f"for logic '{logic}': {prf_path}\n"
            )

            if max_attempts_reached(loop_state):
                loop_state["max_attempts_reached"] = True
                break

        # per-theory records
        dicts.update_records(metrics, records_file)

    print(
        f"Processed all theories in {logic} or processed "
        f"{loop_state['prf_attempts_count']} proofs out of "
        f"{loop_state['max_prf_attempts']}."
    )
    return loop_state


# PROCESS ALL LOGICS

def init_loop_state(eval_config):
    return {
        "prf_attempts_count": 0,
        "max_prf_attempts": eval_config.get("max_prf_attempts", None),
        "max_attempts_reached": False,
        "repl": None
    }


def process_logics(config_dict):
    task = config_dict.get("task", "dfs_eval")
    strategy = STRATEGIES.get(task)
    if strategy is None:
        raise ValueError(
            f"Unknown task '{task}'. Supported: {list(STRATEGIES.keys())}"
        )

    eval_config = strategy.configure(config_dict)
    eval_config["task_label"] = task.replace("_eval", "")
    eval_config["records_file"] = {
        "dfs_eval": "repling_records.json",
        "hammer_eval": "hammer_metrics.json"
    }.get(task, "eval_records.json")
    # propagate shared config fields
    eval_config["max_prf_attempts"] = config_dict.get(
        "max_prf_attempts",
        config_dict.get("dfs_config", {}).get("max_prf_attempts")
    )
    eval_config["max_proof_steps"] = config_dict.get("max_proof_steps")

    loop_state = init_loop_state(eval_config)
    progress_file = config_ops.create_progress_file(file_name="progress.txt")
    logics_dict = proofs.data_dir.group_paths_by_logic(
        config_dict["data_dir"],
        config_dict["data_split"]
    )

    # signal handler
    def shutdown_all():
        print("\n[!] Shutdown signal received...")
        logging.info("Shutdown signal received.")
        if loop_state["repl"]:
            try:
                loop_state["repl"].shutdown()
            except:
                pass
        sys.exit(0)
    signal.signal(signal.SIGINT, lambda sig, frame: shutdown_all())

    try:
        for logic in logics_dict.keys():
            if loop_state["max_attempts_reached"]:
                print("Max attempts reached, stopping process.")
                break
            if config_ops.progress_item_in(logic, progress_file):
                print(f"Skipping already processed logic: {logic}")
                continue
            try:
                thys = logics_dict[logic]
                loop_state = process_logic(logic, thys, eval_config, loop_state, strategy)
            except Exception as e:
                logging.warning(f"Error processing logic '{logic}': {e}")
                if loop_state["repl"]:
                    loop_state["repl"].disconnect()
                loop_state["repl"] = None
    except KeyboardInterrupt:
        shutdown_all()
    finally:
        if loop_state["repl"]:
            loop_state["repl"].shutdown()


if __name__ == "__main__":
    info = "Evaluates a proof strategy (DFS or Hammer) as specified in the JSON config."
    config_dict = config_ops.parse_path(tool_explanation=info)
    process_logics(config_dict)
