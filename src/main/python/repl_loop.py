# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Draft for a loop evaluating the T5 model interation with the repl

import os
import re
import logging

from transformers import pipeline

import ops
import proofs
import eval_t5
import tokenizer_ops as tokops
from repl import REPL

# REPLING

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
    filename = f"{thy_name[:-4]}{prf_num}.thy"
    header = f"theory {thy_name[:-4]}{prf_num}\n imports {logic}.{thy_name[:-4]}\n begin\n\n"
    body = repl.last_proof()
    end = "\n\nend"
    with open(filename, "w", encoding="utf-8") as f:
        f.write(header + body + end)
        print(f"Saved proof to {filename}")

def inputs_from(repl, proof, data_mode):
    xs = [repl.proof_so_far(), proofs.Separator["user_state"], repl.last_usr_state()]
    x = " ".join(proofs.add_spk_data(proof, xs, data_mode=data_mode))
    x = "isabelle next step: " + x if "finetune" in data_mode else x
    return x        

default_generation_config = {
    "gen_length": 50,
    "num_return_sequences": 5,
    "num_beams": 5
}

def attempt_proof(repl, proof, proof_info, gen_config, metrics, data_mode, curr_depth, recurse_depth=5, saving=False):
    x = inputs_from(repl, proof, data_mode)
    print(f"Model input from Isabelle at depth {curr_depth}: {x}")
    predicts = gen_config["generator"](
        x, 
        max_length=gen_config["gen_length"], 
        num_return_sequences=gen_config["num_return_sequences"],
        num_beams=gen_config["num_beams"]
    )
    for i, predict in enumerate(predicts):
        print(f"Attempt {i+1} at depth={curr_depth}")
        if predict is None:
            message = f"""
            None prediction found at:
            attempt {i+1}
            depth {curr_depth}
            model input {x} 
            proof {proof_info['prf_path']}\n"""
            logging.warning(message)
            continue

        y = predict["generated_text"]
        print(f"Model output at depth={curr_depth} is: {y}")
        handling_by = y.strip().startswith("by")
        if handling_by:
            updated_y = convert_by_to_apply(y)
            print(f"Changing to {updated_y}")
        else:
            updated_y = y
        
        repl.apply(updated_y)
        err = repl.latest_error()
        if err:
            print(f"Attempt did not work. Backtracking.")
            metrics["no_progress_counter"] += 1
            repl.undo()
            continue
        else:
            metrics["progress_counter"] += 1
            if repl.without_subgoals():
                print("Without subgoals reached!")
                if handling_by:
                    metrics["correct_by"] += 1
                    repl.undo()
                    repl.apply(y)
                else:
                    repl.complete_step()
            if not repl.is_at_proof() or "Duplicate" in repl.latest_error():
                metrics["finished_proofs"] += 1
                if saving:
                    print("trying to save proof")
                    save_proof(repl, proof_info)
                repl.reset()
                return metrics
            elif recurse_depth == 0:
                print("reached max depth. Last proof was:")
                print(f"{repl.last_proof()}\n")
                repl.reset()
                continue
            else:
                if handling_by:
                    metrics["incorrect_by"] += 1
                metrics = attempt_proof(repl, proof, proof_info, gen_config, metrics, data_mode, curr_depth=curr_depth+1, recurse_depth=recurse_depth-1, saving=saving)
        print(f"Prediction {i} processed\n")
    return metrics

def make_repl_metrics():
    metrics = {}
    metrics["total_proofs"] = 0
    metrics["progress_counter"] = 0
    metrics["no_progress_counter"] = 0
    metrics["correct_by"] = 0
    metrics["incorrect_by"] = 0
    metrics["finished_proofs"] = 0
    return metrics

def do_repling(config_dict, model, tokenizer, gen_config=default_generation_config, recurse_depth=5, saving=False):
    logics_dict = group_paths_by_logic(config_dict)
    print(f"Model context length = {model.config.n_positions}")
    print(f"Tokenizer context length = {tokenizer.model_max_length}")
    gen_config["generator"] = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    metrics = make_repl_metrics()
    try:
        for logic in logics_dict.keys():
            repl = None
            print(f"Processing logic {logic}")
            for thy_name in logics_dict[logic]:
                try:
                    len_proofs = len(logics_dict[logic][thy_name])
                    print(f"Processing theory {thy_name}")
                    if repl is None:
                        repl = REPL(logic, thy_name)
                    else:
                        repl.go_to_end_of(thy_name)
                    for i, (prf_num, path) in enumerate(logics_dict[logic][thy_name]):
                        try:
                            proof_info = {
                                "prf_num": prf_num,
                                "prf_path": path,
                                "prf_thy_name": thy_name,
                                "logic": logic
                                }
                            proof = proofs.get_proof_json(path)
                            acts = [fix_missing_quotations(a) for a in proofs.full_actions_of(proof)]
                            repl.apply(acts[0])
                            print(f"Attempting (successfully loaded) proof {path}")
                            metrics = attempt_proof(repl, proof, proof_info, gen_config, metrics, config_dict["data_mode"], curr_depth=1, recurse_depth=recurse_depth, saving=saving)
                            metrics["total_proofs"] += 1
                            ops.save_dict_as_json(metrics, "repling_records.json")
                            print(f"Processed proof {i + 1} of {len_proofs}: {path}\n\n")
                        except Exception as e:
                            logging.warning(f"Error processing proof at {path}: {e}")
                        finally:
                            repl.reset()
                except Exception as e:
                    logging.warning(f"Error initializing REPL for {logic}.{thy_name}: {e}")
                finally:
                    if repl:
                        repl.shutdown_isabelle()
        if repl:
            repl.shutdown_isabelle()   
    finally:
        if repl:
            repl.shutdown_gateway()
    return metrics

if __name__ == "__main__":
    # ops.configure_logging("t5_repl_loop.log")
    try:
        config_dict = ops.get_json_dict(ops.parse_config_path(tool_explanation="Evaluate the transformer as specified in the input JSON configuration."))
        ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)

    model, tokenizer, dataset = eval_t5.load_model_tok_data(config_dict)
    do_repling(config_dict, model, tokenizer, saving=True)