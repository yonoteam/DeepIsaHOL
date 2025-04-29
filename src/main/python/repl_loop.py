# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Draft for a loop evaluating the T5 model interation with the repl

import os
import gc
import re
import json
import fcntl
import psutil
import logging
from itertools import takewhile

import torch
from transformers import pipeline

import dicts
import ml.eval_t5
import ml.config_ops
import proofs.data_dir
import proofs.str_ops
import ml.tokenizer_ops as tokops
from repl import REPL

# REPLING

def setup_logging(logic_name):
    """Setup logging for a specific logic with a dedicated log file inside its own directory."""
    log_dir = os.path.join(os.getcwd(), logic_name)
    os.makedirs(log_dir, exist_ok=True)  # Ensure directory exists
    
    log_file = os.path.join(log_dir, f"{logic_name}_repl.log")
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)

    logging.basicConfig(
        filename=log_file,
        filemode="w",  # Overwrite each time the script runs
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

def log_memory_usage(message):
    process = psutil.Process(os.getpid())
    memory = process.memory_info().rss / 1024 / 1024  # MB
    logging.info(f"Memory usage ({message}): {memory:.2f} MB")

def update_repling_records(new_metrics, filename="repling_records.json"):
    if not os.path.exists(filename):
        with open(filename, "w", encoding="utf-8") as f:
            json.dump({}, f)
    
    with open(filename, "r+") as f:
        # exclusive lock on file.
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            try:
                current_data = json.load(f)
            except json.JSONDecodeError:
                current_data = {}
            
            for key, value in new_metrics.items():
                if isinstance(value, (int, float)):
                    current_data[key] = current_data.get(key, 0) + value
                else:
                    # TODO: decide non-numerics handling if added
                    current_data[key] = value 
            
            # move file pointer to beginning and truncate file.
            f.seek(0)
            json.dump(current_data, f, indent=4)
            f.truncate()
        finally:
            fcntl.flock(f, fcntl.LOCK_UN)

def fix_missing_quotations(s):
    double_quotes = s.count('"') % 2
    if double_quotes:
        s += '"'
    return s

def convert_by_to_apply(by_statement:str):
    """
    Converts a 'by' statement into one or more 'apply' statements.
    
    Args:
        by_statement (str): A string starting with 'by', e.g., 'by (induct rule: hpMHDecidable.induct) (auto simp: hpMHDecidable_def)'
    
    Returns:
        str: The equivalent apply statement(s)
    """
    
    # Remove the 'by' prefix and strip any leading/trailing whitespace
    args_string = by_statement.strip()[2:].strip()
    
    # If there are no arguments, return an empty string
    if not args_string:
        return "apply"
    
    # Initialize variables
    apply_statements = []
    current_arg = ""
    parenthesis_count = 0
    
    # Process each character to correctly handle nested parentheses
    for char in args_string:
        if char == '(' and current_arg.strip() == "":
            # Start of a new parenthesized argument
            parenthesis_count = 1
            current_arg += char
        elif char == '(' and parenthesis_count > 0:
            # Nested opening parenthesis
            parenthesis_count += 1
            current_arg += char
        elif char == ')' and parenthesis_count > 0:
            # Closing parenthesis
            parenthesis_count -= 1
            current_arg += char
            
            # If we've closed all parentheses, add this argument
            if parenthesis_count == 0 and current_arg.strip():
                apply_statements.append(f"apply {current_arg.strip()}")
                current_arg = ""
        else:
            # Regular character
            if parenthesis_count > 0:
                # Inside parentheses, add to current argument
                current_arg += char
            elif char.strip():
                # Outside parentheses and not whitespace, start a new non-parenthesized argument
                current_arg += char
            elif current_arg.strip():
                # Whitespace after a non-parenthesized argument
                apply_statements.append(f"apply {current_arg.strip()}")
                current_arg = ""
    
    # Handle any remaining argument
    if current_arg.strip():
        apply_statements.append(f"apply {current_arg.strip()}")
    
    # Join the apply statements with newlines
    return "\n".join(apply_statements)

def group_paths_by_logic(config_dict):
    data_dir = config_dict["data_dir"]
    logics_dict = {}
    proof_pattern = re.compile(r'proof(\d+)\.json$')

    for path in tokops.generate_proof_paths(data_dir, split=config_dict["data_split"]):
        logic_path = os.path.relpath(path, data_dir)
        path_parts = logic_path.split(os.sep)

        logic = path_parts[0]
        if not proof_pattern.search(path_parts[-1]):  
            continue
        
        thy_name = f"{path_parts[-2]}.thy"  

        proof_num = int(proof_pattern.search(path_parts[-1]).group(1))  

        if logic not in logics_dict:
            logics_dict[logic] = {}

        if thy_name not in logics_dict[logic]:
            logics_dict[logic][thy_name] = []

        logics_dict[logic][thy_name].append((proof_num, path))

    # sort numerically
    for logic in logics_dict:
        for thy_name in logics_dict[logic]:
            logics_dict[logic][thy_name].sort()

    return logics_dict

def save_proof(repl, proof_info):
    thy_name = proof_info["prf_thy_name"]
    prf_num = proof_info["prf_num"]
    logic = proof_info["logic"]
    filename = f"{logic}/{thy_name[:-4]}{prf_num}.thy"
    header = f"theory {thy_name[:-4]}{prf_num}\n imports {logic}.{thy_name[:-4]}\n begin\n\n"
    body = repl.last_proof()
    end = "\n\nend"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + body + end)
        logging.info(f"Saved proof to {filename}")

def inputs_from(repl, proof, data_mode):
    xs = [repl.proof_so_far(), proofs.str_ops.Separator["user_state"], repl.last_usr_state()]
    x = " ".join(proofs.str_ops.add_spk_data(proof, xs, data_mode=data_mode))
    x = "isabelle next step: " + x if "finetune" in data_mode else x
    return x        

default_generation_config = {
    "gen_length": 50,
    "num_return_sequences": 5,
    "num_beams": 5
}

def carefully_go_back(repl, pos, allowed_breadth, current_breath):
    if current_breath >= allowed_breadth:
        nonzeros_rl = reversed(list(takewhile(lambda x: x != 0, pos)))
        nsteps_back = len(list(takewhile(lambda x: x == allowed_breadth, nonzeros_rl))) + 1
        logging.info(f"Returning {nsteps_back} steps back.\n")
        repl.undoN(nsteps_back)
    else:
        repl.undo()

def attempt_proof(repl, 
                  proof_info, 
                  gen_config, 
                  metrics, 
                  data_mode, 
                  pos,
                  max_depth=5, 
                  saving=False):
    x = inputs_from(repl, proof_info["proof"], data_mode)
    logging.info(f"Trimmed model input from Isabelle at pos {pos}: {x[:500]}")
    predicts = gen_config["generator"](
        x, 
        max_length=gen_config["gen_length"], 
        num_return_sequences=gen_config["num_return_sequences"],
        num_beams=gen_config["num_beams"]
    )
    curr_pos = pos.copy()
    curr_depth = next((i for i, x in enumerate(pos) if x == 0), None)
    for i, predict in enumerate(predicts, start=1):
        pos = curr_pos.copy()
        pos[curr_depth] = i
        logging.info(f"Attempt at pos={pos}")
        if predict is None:
            message = f"""
            None prediction found at:
            pos = {pos}
            model input = {x} 
            proof = {proof_info['prf_path']}\n"""
            logging.warning(message)
            continue

        y = predict["generated_text"]
        logging.info(f"Model output at pos={pos} is: {y}")

        handling_by = y.strip().startswith("by")
        if handling_by:
            updated_y = convert_by_to_apply(y)
            logging.info(f"Changing to {updated_y}")
        else:
            updated_y = y
        
        repl.apply(updated_y)
        allowed_breadth = len(predicts)

        # repl replied with an error
        err = repl.last_error()
        if err:
            logging.info(f"Attempt did not work. Backtracking.")
            metrics["no_progress_counter"] += 1
            carefully_go_back(repl, pos, allowed_breadth, i)
            continue
        # action was successful
        else:
            metrics["progress_counter"] += 1
            # reached the end of the proof
            if repl.is_at_proof():
                if repl.without_subgoals():
                    logging.info("Without subgoals reached!")
                    if handling_by:
                        metrics["correct_by"] += 1
                        repl.undo()
                        repl.apply(y)
                    else:
                        repl.complete_step()
            
            # proof is finished
            if not repl.is_at_proof() or "Duplicate" in repl.last_error():
                metrics["finished_proofs"] += 1
                if saving:
                    logging.info("trying to save proof")
                    save_proof(repl, proof_info)
                repl.reset()
                return metrics
            # reached max depth
            elif max_depth == 1:
                logging.info("reached max depth. Last proof was:")
                logging.info(f"{repl.last_proof()}")
                carefully_go_back(repl, pos, allowed_breadth, i)
                continue
            else:
                if handling_by:
                    metrics["incorrect_by"] += 1
                curr_finished_proofs = metrics["finished_proofs"]
                metrics = attempt_proof(repl, 
                                        proof_info, 
                                        gen_config, 
                                        metrics,
                                        data_mode, 
                                        pos = pos,
                                        max_depth=max_depth-1, 
                                        saving=saving)
                if metrics["finished_proofs"] > curr_finished_proofs:
                    return metrics
        
        logging.info(f"Processed prediction at pos={pos}\n")
    return metrics

def make_repl_metrics():
    return {
        "total_proofs": 0,
        "progress_counter": 0,
        "no_progress_counter": 0,
        "correct_by": 0,
        "incorrect_by": 0,
        "finished_proofs": 0
    }

def do_repling(
        config_dict, 
        model, 
        tokenizer, 
        gen_config=default_generation_config, 
        allowed_depth=5, 
        saving=False):
    logics_dict = group_paths_by_logic(config_dict)
    tok_max_length = ml.get_context_length(config_dict["data_mode"])
    tokenizer.model_max_length = tok_max_length
    print(f"Model context length = {model.config.n_positions}")
    print(f"Tokenizer context length = {tokenizer.model_max_length}")
    gen_config["generator"] = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
    gen_config["allowed_depth"] = allowed_depth

    progress_file = "progress.txt"
    if not os.path.exists(progress_file):
        with open(progress_file, "w", encoding="utf-8"):
            pass
    
    repl = None
    metrics = make_repl_metrics()
    try:
        for logic in logics_dict.keys():
            with open(progress_file, "r", encoding="utf-8") as f:
                completed_logics = {line.strip() for line in f}
            if logic in completed_logics:
                print(f"Skipping already processed logic: {logic}")
                continue
            with open(progress_file, "a", encoding="utf-8") as f:
                f.write(logic + "\n")
            
            len_proofs = sum(len(thy_list) for thy_list in logics_dict[logic].values())
            prf_count = 0
            if repl:
                repl.switch_to(logic)
            else:
                repl = REPL(logic)
            setup_logging(logic)
            logging.info(f"Processing logic {logic}")

            for thy_name in logics_dict[logic]:
                logging.info(f"Processing theory {thy_name}")
                for prf_num, path in logics_dict[logic][thy_name]:
                    try:
                        proof = dicts.load_json(path)
                        logging.info(f"Attempting (successfully loaded) proof {path}")
                        proof_info = {
                            "prf_num": prf_num,
                            "prf_path": path,
                            "prf_thy_name": thy_name,
                            "logic": logic,
                            "lemma_name": fix_missing_quotations(proofs.orig_objective_of(proof)),
                            "proof": proof
                        }
                        # acts = [fix_missing_quotations(a) for a in proofs.full_actions_of(proof)]
                        repl.go_to(thy_name, proof_info["lemma_name"])

                        # log_memory_usage("Before processing proof")
                        metrics = attempt_proof(
                            repl, 
                            proof_info, 
                            gen_config, 
                            make_repl_metrics(), 
                            config_dict["data_mode"], 
                            pos = list(0 for _ in range(allowed_depth)),
                            max_depth=allowed_depth, 
                            saving=saving
                        )
                        # log_memory_usage("After processing proof")

                        metrics["total_proofs"] += 1
                        update_repling_records(metrics, "repling_records.json")
                        prf_count += 1
                        logging.info(f"Processed proof {prf_count} of {len_proofs}: {path}\n\n")
                        gc.collect()
                        torch.cuda.empty_cache()
                    except Exception as e:
                        logging.warning(f"Error processing proof at {path} with REPL at {logic}.{thy_name}: {e}")
                    finally:
                        repl.reset()          
    except Exception as e:
        logging.warning(f"Error processing {logic}: {e}")
    finally:
        if repl:
            repl.shutdown_gateway()
    return metrics

if __name__ == "__main__":
    try:
        info="Evaluates a T5 model via depth-first search proof exploration as specified in the input JSON configuration."
        path = ml.config_ops.parse_config_path(tool_explanation=info)
        config_dict = dicts.load_json(path)
        ml.config_ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception(f"Error {e}")
    
    model, tokenizer, dataset = ml.eval_t5.load_model_tok_data(config_dict)
    model.to("cpu")
    do_repling(config_dict, model, tokenizer, saving=True)