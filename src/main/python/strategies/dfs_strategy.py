# Maintainers:
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
#
# DFS strategy for the unified evaluation loop (eval.py).
# Extracted from dfs.py.

import gc
import time
import logging
from itertools import takewhile

import proofs
from strategies.shared import save_proof


# STRATEGY INTERFACE

def configure(config_dict):
    """Return an eval_config dict with the LLM generator loaded."""
    import generation_ops as genops
    dfs_config = config_dict.get("dfs_config", {}).copy()
    generation_config = genops.configure_generator(config_dict)
    dfs_config.update(generation_config)
    return dfs_config


def init_metrics():
    """Return DFS-specific metric counters."""
    return {
        "total_proofs": 0,
        "progress_counter": 0,
        "no_progress_counter": 0,
        "correct_by": 0,
        "incorrect_by": 0,
        "timed_out_proofs": 0,
        "proof_durations": []
    }


def load_prf_info(proof_json, prf_num, prf_path, thy_name, logic, eval_config):
    """Build the per-proof info dict from the raw JSON proof."""
    prf_start = proofs.orig_objective_of(proof_json)
    prf_start = proofs.str_ops.fix_missing_quotations(prf_start)
    data_format = eval_config.get("data_format", "state")
    prf_data = proofs.str_ops.add_spk_data(proof_json, {}, data_format=data_format)
    return {
        "num": prf_num,
        "path": prf_path,
        "thy_name": thy_name,
        "logic": logic,
        "start_line": prf_start,
        "start_time": None,
        "proof_data": prf_data
    }


def attempt_proof(repl, prf_info, eval_config, metrics):
    """Run one DFS proof attempt. REPL is already positioned at the proof start."""
    try:
        start_time = time.time()
        dfs_metrics = _measure_dfs(repl, prf_info, eval_config)
        duration = time.time() - start_time
        metrics["durations"].append(duration)

        # flow DFS-internal finished_proofs to shared counter
        metrics["finished_proofs"] += dfs_metrics.get("finished_proofs", 0)

        # merge DFS-specific counters
        for k in ("total_proofs", "progress_counter", "no_progress_counter",
                   "correct_by", "incorrect_by", "timed_out_proofs"):
            metrics[k] = metrics.get(k, 0) + dfs_metrics.get(k, 0)
        metrics.setdefault("proof_durations", []).extend(
            dfs_metrics.get("proof_durations", [])
        )
    except Exception as e:
        logging.warning(f"Error processing proof at {prf_info['path']}: {e}")
    finally:
        repl.reset()
        gc.collect()
        _clear_cuda_cache()
    return metrics


# PRIVATE DFS FUNCTIONS

def _clear_cuda_cache():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _carefully_go_back(repl, pos, max_width, curr_width):
    if curr_width >= max_width:
        nonzeros_rl = reversed(list(takewhile(lambda x: x != 0, pos)))
        nsteps_back = len(list(takewhile(lambda x: x == max_width, nonzeros_rl))) + 1
        logging.info(f"Returning {nsteps_back} steps back.\n")
        repl.undoN(nsteps_back)
    else:
        repl.undo()


def _make_prf_record(prf_info, duration):
    return {
        "thy_name": prf_info["thy_name"],
        "logic": prf_info["logic"],
        "num": prf_info["num"],
        "duration": duration
    }


def _init_proof_metrics():
    return {
        "total_proofs": 0,
        "progress_counter": 0,
        "no_progress_counter": 0,
        "correct_by": 0,
        "incorrect_by": 0,
        "finished_proofs": 0,
        "timed_out_proofs": 0,
        "proof_durations": []
    }


def _dfs(repl, metrics, pos, max_depth, prf, dfs_config):
    import generation_ops as genops

    # timeout check
    timeout_seconds = dfs_config.get("proof_timeout_seconds")
    if timeout_seconds is not None:
        elapsed_time = time.time() - prf["start_time"]
        if elapsed_time > timeout_seconds:
            metrics["timed_out_proofs"] += 1
            logging.info(f"Timeout threshold ({timeout_seconds}s) reached during DFS for proof {prf['path']} at pos {pos}. Stopping exploration down this path.")
            return metrics

    prf_info = {
        "proof_so_far": repl.proof_so_far(),
        "last_usr_state": repl.last_usr_state(),
        "proof_data": prf["proof_data"]
    }
    try:
        x, predicts = genops.generate_predicts(prf_info, dfs_config)
    except Exception as e:
        logging.warning(f"Error generating predictions at pos={pos} for proof {prf['path']}: {e}")
        return metrics

    # cleanup after generation
    _clear_cuda_cache()
    gc.collect()

    logging.info(f"Next (trimmed) model input from Isabelle is: {x[:500]}")
    logging.info(f"at pos={pos}.")
    if not predicts:
        logging.warning(f"No predictions returned for proof {prf['path']} at pos={pos}")
        return metrics

    first_generation = predicts[0]
    if first_generation is None:
        logging.warning(f"First prediction is None at pos={pos} for proof {prf['path']}")
    else:
        first_generation = predicts[0][:200]
        logging.info(f"Successful prediction: {first_generation}")

    max_breadth = len(predicts)

    # determine current position in dfs tree
    curr_pos = pos.copy()
    curr_depth = next((i for i, x in enumerate(pos) if x == 0), None)

    # main loop
    for i, predict in enumerate(predicts, start=1):
        curr_pos[curr_depth] = i
        logging.info(f"Attempt at pos={curr_pos}")

        # unlikely safety check
        if predict is None:
            message = f"""
            Invalid prediction found at:
            pos = {curr_pos}
            model input = {x}
            proof = {prf['path']}\n"""
            logging.warning(message)
            continue

        # apply prediction replacing by with apply
        y = predict
        logging.info(f"Model output at pos={curr_pos} is: {y}")
        handling_by = y.strip().startswith("by")
        if handling_by:
            updated_y = proofs.str_ops.convert_by_to_apply(y)
            logging.info(f"Changing to {updated_y}")
        else:
            updated_y = y
        repl.apply(updated_y)

        # if repl replied with an error
        err = repl.last_error()
        if err:
            logging.info(f"Attempt did not work. Backtracking due to error '{err}'.")
            metrics["no_progress_counter"] += 1
            _carefully_go_back(repl, curr_pos, max_breadth, i)
            continue

        # action was successful
        else:
            metrics["progress_counter"] += 1

            # if reached the end of the proof
            if repl.is_at_proof() and repl.without_subgoals():
                logging.info("Without subgoals reached!")
                if handling_by:
                    metrics["correct_by"] += 1
                    repl.undo()
                    repl.apply(y)
                else:
                    repl.complete_step()

            # if proof is finished
            if not repl.is_at_proof() or "Duplicate" in repl.last_error():
                duration = time.time() - prf["start_time"]
                metrics["finished_proofs"] += 1
                metrics["proof_durations"].append(_make_prf_record(prf, duration))
                if dfs_config.get("saving", False):
                    logging.info("trying to save proof")
                    save_proof(repl, prf)
                repl.reset()
                return metrics

            # reached max depth
            elif max_depth == 1:
                if handling_by:
                    metrics["incorrect_by"] += 1
                logging.info("reached max depth. Last proof was:")
                logging.info(f"{repl.last_proof()}")
                _carefully_go_back(repl, curr_pos, max_breadth, i)
                continue
            else:
                if handling_by:
                    metrics["incorrect_by"] += 1

                # recurse
                curr_finished_proofs = metrics["finished_proofs"]
                curr_timed_out_proofs = metrics["timed_out_proofs"]
                metrics = _dfs(
                    repl,
                    metrics,
                    pos=curr_pos,
                    max_depth=max_depth-1,
                    prf=prf,
                    dfs_config=dfs_config
                )

                # if proof found downstream, stop exploration
                if metrics["finished_proofs"] > curr_finished_proofs:
                    return metrics

                # if timeout occurred downstream, stop exploration
                if metrics["timed_out_proofs"] > curr_timed_out_proofs:
                    return metrics

        logging.info(f"Processed prediction at pos={curr_pos}\n")
    return metrics


def _measure_dfs(repl, prf_info, dfs_config):
    metrics = _init_proof_metrics()
    max_depth = dfs_config["allowed_depth"]
    start_pos = list(0 for _ in range(max_depth))
    prf_info["start_time"] = time.time()

    metrics = _dfs(
        repl,
        metrics,
        pos=start_pos,
        max_depth=max_depth,
        prf=prf_info,
        dfs_config=dfs_config
    )
    metrics["total_proofs"] += 1
    return metrics
