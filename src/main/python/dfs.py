# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Draft for a loop evaluating the T5 model interation with the repl

import os
import re
import gc
import time
import fcntl
import logging
from itertools import takewhile

import torch

import dicts
import proofs
import config_ops
import tokenizer_ops as tokops
from repl import REPL

# DFS OPERATIONS

def update_repling_records(
        new_metrics, 
        filename="repling_records.json"
    ):
    if not os.path.exists(filename):
        dicts.save_as_json({}, filename)
    with open(filename, "r+") as f:
        fcntl.flock(f, fcntl.LOCK_EX) # exclusive lock on file
        try:
            curr_data = dicts.safe_load_json(f)
            for key, value in new_metrics.items():
                if isinstance(value, (int, float)):
                    curr_data[key] = curr_data.get(key, 0) + value
                elif isinstance(value, list):
                    curr_data.setdefault(key, []).extend(value)
                else:
                    # TODO: decide other non-numerics handling if added
                    curr_data[key] = value 
            dicts.replace_in_opened(curr_data, f)
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def save_proof(repl, prf):
    thy_name = prf["thy_name"]
    prf_num = prf["num"]
    logic = prf["logic"]

    base_name, _ = os.path.splitext(thy_name) # removes .thy
    filename = f"{logic}/{base_name}{prf_num}.thy" 
    header = f"theory {base_name}{prf_num}\n imports {logic}.{base_name}\n begin\n\n"
    body = repl.last_proof()
    end = "\n\nend"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + body + end)
        logging.info(f"Saved proof to {filename}")


# DEPTH FIRST SEARCH

def extract_gemma_suggestion(text):
    pattern = r"<SUGGESTION>(.*?)</SUGGESTION>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def generate_predicts(repl, prf_info, dfs_config):
    data_format = dfs_config["data_format"]
    model_type = dfs_config["model_type"]
    usr_sep = proofs.str_ops.Separator["user_state"]

    xs = [repl.proof_so_far(), usr_sep, repl.last_usr_state()]
    xs.extend(prf_info["proof_data"])

    x = " ".join(xs)
    logging.info(f"Next (trimmed) model input from Isabelle is: {x[:500]}")

    if model_type == "t5":
        x = "isabelle next step: " + x if "finetune" in data_format else x  
        predicts = dfs_config["generator"](
            x, 
            max_length=dfs_config["gen_length"], 
            num_return_sequences=dfs_config["num_return_sequences"],
            num_beams=dfs_config["num_beams"]
        )
        predicts = [p["generated_text"] for p in predicts]
    elif model_type == "gemma":
        gemma_prompt = "Recommend the next Isabelle proof step given the context below:\n{context}"
        convers = {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": gemma_prompt.format(context=x)}
                    ]
                }
            ]
        }
        predicts = dfs_config["generator"](
            convers["messages"], 
            max_new_tokens=dfs_config["gen_length"], 
            num_return_sequences=dfs_config["num_return_sequences"],
            num_beams=dfs_config["num_beams"],
            temperature = 1.0,
            top_p = 0.95,
            top_k = 64
        )
        predicts = [extract_gemma_suggestion(p["generated_text"]) for p in predicts]
    return predicts

# the pos variable indicates a position in the dfs tree, i.e. [i, j, k, l] corresponds 
# to the i, j, k, and l branches at resp. depths 1, 2, 3, and 4. If the options at a 
# given depth are exhausted, e.g. [i, max_width, max_width, 0], the repl needs to 
# backtrack (the number of occurances of max_width plus one) to keep exploring at an 
# unfinished depth, e.g. explore i+1 at 1st depth.
def carefully_go_back(repl, pos, max_width, curr_width):
    if curr_width >= max_width:
        nonzeros_rl = reversed(list(takewhile(lambda x: x != 0, pos)))
        nsteps_back = len(list(takewhile(lambda x: x == max_width, nonzeros_rl))) + 1
        logging.info(f"Returning {nsteps_back} steps back.\n")
        repl.undoN(nsteps_back)
    else:
        repl.undo()

def make_prf_record(prf_info, duration):
    return {
        "thy_name": prf_info["thy_name"],
        "logic": prf_info["logic"],
        "num": prf_info["num"],
        "duration": duration
    }

# TODO: If text-generation, extract correct predicts
def dfs(
        repl, # mutable parameters
        metrics,
        pos,
        max_depth,
        prf, # immutable parameters
        dfs_config
    ):
    # timeout check
    timeout_seconds = dfs_config.get("proof_timeout_seconds")
    if timeout_seconds is not None:
        elapsed_time = time.time() - prf["start_time"]
        if elapsed_time > timeout_seconds:
            metrics["timed_out_proofs"] += 1
            logging.info(f"Timeout threshold ({timeout_seconds}s) reached during DFS for proof {prf['path']} at pos {pos}. Stopping exploration down this path.")
            return metrics

    predicts = generate_predicts(repl, prf, dfs_config)
    logging.info(f"at pos={pos}.")
    max_breadth = len(predicts)

    # determine current position in dfs tree
    curr_pos = pos.copy()
    curr_depth = next((i for i, x in enumerate(pos) if x == 0), None)

    # main loop
    for i, predict in enumerate(predicts, start=1):
        curr_pos[curr_depth] = i
        logging.info(f"Attempt at pos={curr_pos}")
        
        # unlikely safety check
        if predict is None or "generated_text" not in predict:
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
            carefully_go_back(repl, curr_pos, max_breadth, i)
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
                metrics["proof_durations"].append(make_prf_record(prf, duration))
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
                carefully_go_back(repl, curr_pos, max_breadth, i)
                continue
            else:
                if handling_by:
                    metrics["incorrect_by"] += 1

                # recurse
                curr_finished_proofs = metrics["finished_proofs"]
                curr_timed_out_proofs = metrics["timed_out_proofs"]
                metrics = dfs(
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
                
                # it timeout occurred downstream, stop exploration
                if metrics["timed_out_proofs"] > curr_timed_out_proofs:
                    return metrics
        
        logging.info(f"Processed prediction at pos={curr_pos}\n")
    return metrics

def init_proof_metrics():
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

def measure_dfs(
        repl, 
        prf_info, 
        dfs_config
    ):
    metrics = init_proof_metrics()
    max_depth = dfs_config["allowed_depth"]
    start_pos = list(0 for _ in range(max_depth))
    prf_info["start_time"] = time.time()

    metrics = dfs(
        repl,
        metrics,
        pos=start_pos, 
        max_depth=max_depth, 
        prf=prf_info, 
        dfs_config=dfs_config
        )
    metrics["total_proofs"] += 1
    return metrics

# PROOF PROCESSING

def load_proof(data_format, prf_num, prf_path, thy_name, logic):
    proof = dicts.load_json(prf_path)
    prf_start = proofs.orig_objective_of(proof)
    prf_start = proofs.str_ops.fix_missing_quotations(prf_start)
    prf_data = proofs.str_ops.add_spk_data(proof, [], data_format=data_format)
    return {
        "num": prf_num,          # proof number inside the theory file
        "path": prf_path,        # path to the proof JSON file
        "thy_name": thy_name,    # theory file's name
        "logic": logic,          # isabelle session name
        "start_line": prf_start, # string stating the lemma to prove
        "start_time": None,      # time when the proof started
        "proof_data": prf_data   # data (e.g. deps) from the original proof
    }

def attempt_proof(
        repl,
        prf_count,
        prf_info,
        dfs_config
    ):
    try:
        thy_name = prf_info["thy_name"]
        prf_path = prf_info["path"]
        logic = prf_info["logic"]
        prf_start = prf_info["start_line"]
        # do dfs and measure proof attempt

        repl.go_to(thy_name, prf_start)
        metrics = measure_dfs(
            repl, 
            prf_info,
            dfs_config
        )
        update_repling_records(metrics, "repling_records.json")
    except Exception as e:
        logging.warning(f"Error processing proof at {prf_path} with REPL at {logic}.{thy_name}: {e}")
    finally:
        repl.reset()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return prf_count + 1, repl


# PROGRESS FILE AND LOGIC PROCESSING

def create_progress_file():
    progress_file = "progress.txt"
    if not os.path.exists(progress_file):
        with open(progress_file, "w", encoding="utf-8"):
            pass
    return progress_file

def progress_logic_in(logic, progress_file):
    # "progress" used both as adjective and verb
    # made in this way for concurrency safety
    with open(progress_file, "r+", encoding="utf-8") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            completed_logics = {line.strip() for line in f}
            if logic in completed_logics:
                return True
            
            f.seek(0, os.SEEK_END) # go to end of file
            f.write(logic + "\n")
            f.flush()              # ensure buffer written to OS
            os.fsync(f.fileno())   # ensure OS writes to disk
            return False
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def setup_logic_logging(logic_name):
    """
    Setup logging for a specific logic with a dedicated 
    log file inside its own directory.
    """
    log_dir = os.path.join(os.getcwd(), logic_name)
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"{logic_name}_repl.log")
    
    # remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        filemode="w",  # overwrite each time the script runs
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def max_attempts_reached(loop_state):
    max_attempts = loop_state.get("max_prf_attempts")
    if max_attempts is None:
        return False
    
    return loop_state["prf_attempts_count"] >= max_attempts

def process_logic(logic, thys, dfs_config, loop_state):
    repl = loop_state["repl"]
    prf_count = 0 # counter with respect to len_proofs
    len_proofs = sum(len(thy_proofs) for thy_proofs in thys.values())
    if repl:
        repl.switch_to(logic)
    else:
        repl = REPL(logic)
    setup_logic_logging(logic)
    logging.info(f"Processing logic {logic}")
    for thy_name in thys:
        if loop_state["max_attempts_reached"]:
            break
        logging.info(f"Processing theory {thy_name}")
        for prf_num, prf_path in thys[thy_name]:
            prf_info = load_proof(
                dfs_config["data_format"],
                prf_num,
                prf_path, 
                thy_name,
                logic
            )
            logging.info(f"Loaded proof at {prf_path}")
            prf_count, loop_state["repl"] = attempt_proof(
                repl, 
                prf_count, 
                prf_info, 
                dfs_config
            )
            logging.info(f"Processed proof {prf_count} of {len_proofs}: {prf_path}\n\n")
            loop_state["prf_attempts_count"] += 1
            if max_attempts_reached(loop_state):
                loop_state["max_attempts_reached"] = True
                break
    print(f"Processed all theories in {logic} or max attempts reached.")
    return loop_state


# MAIN LOOP

def get_model_type(config_dict):
    model_name = config_dict["model_name"]
    lower_case_model = model_name.lower()
    if "t5" in lower_case_model:
        return "t5"
    elif "gemma" in lower_case_model:
        return "gemma"
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported: T5, Gemma.")

def load_tok_model(config_dict):
    model_type = get_model_type(config_dict)
    data_format = config_dict["data_format"]
    device = config_dict["dfs_config"]["device"]

    if model_type == "t5":
        import eval_t5
        model, tokenizer, _ = eval_t5.load_model_tok_data(config_dict)
        tok_max_length = tokops.get_t5_context_length(data_format)
        tokenizer.model_max_length = tok_max_length
        print(f"Model context length = {model.config.n_positions}")
        print(f"Tokenizer context length = {tokenizer.model_max_length}")
    elif model_type == "gemma":
        import train_unsloth_gemma
        model, tokenizer = train_unsloth_gemma.load_tuned_objs(config_dict)
    
    if device:
        model.to(config_ops.get_device_str(config_dict))
    return tokenizer, model

def configure(config_dict):
    model_type = get_model_type(config_dict)
    data_format = config_dict["data_format"]
    tokenizer, model = load_tok_model(config_dict)
    from transformers import pipeline # after importing unsloth in load_tok_model

    if model_type == "t5":
        generation_task = "text2text-generation"
    elif model_type == "gemma":
        generation_task = "text-generation"
    
    # setup configuration
    dfs_config = config_dict["dfs_config"].copy()

    # extensions to the configuration
    dfs_config["repl"] = None
    dfs_config["data_format"] = data_format
    dfs_config["model_type"] = model_type
    dfs_config["generator"] = pipeline(
        generation_task,
        model=model, 
        tokenizer=tokenizer, 
        device=config_dict["dfs_config"]["device"] # -1 for CPU, N for GPU N
    )
    return dfs_config

def init_loop_state(dfs_config):
    return {
        "prf_attempts_count": 0,
        "max_prf_attempts": dfs_config.get("max_prf_attempts", None),
        "max_attempts_reached": False,
        "repl": None
    }

def process_logics(config_dict):
    dfs_config = configure(config_dict) # llm-generator inside dfs_config
    loop_state = init_loop_state(dfs_config)
    progress_file = create_progress_file()
    logics_dict = proofs.data_dir.group_paths_by_logic(
        config_dict["data_dir"], 
        config_dict["data_split"]
    )
    try:
        for logic in logics_dict.keys():
            if loop_state["max_attempts_reached"]:
                print("Max attempts reached, stopping process.")
                break
            if progress_logic_in(logic, progress_file):
                print(f"Skipping already processed logic: {logic}")
                continue
            # else: progress_logic_in(logic, progress_file)

            thys = logics_dict[logic]
            
            loop_state = process_logic(logic, thys, dfs_config, loop_state)
    except Exception as e:
        logging.warning(f"Error processing {logic}: {e}")
    finally:
        if loop_state["repl"]:
            loop_state["repl"].shutdown_gateway()

if __name__ == "__main__":
    try:
        info="Evaluates a model via depth-first search proof exploration as specified in the input JSON configuration."
        config_dict = config_ops.parse_path(tool_explanation=info)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception(f"Error {e}")
    
    process_logics(config_dict)