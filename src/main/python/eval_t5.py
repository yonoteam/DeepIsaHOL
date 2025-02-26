# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for evaluating a Hugging Face T5 Model

import os
import logging

import torch
from torch.utils.data import DataLoader
from transformers import (
    DataCollatorForSeq2Seq,
    pipeline
)
from accelerate.utils import broadcast_object_list

import ops
import proofs
import train_t5
from repl import REPL
import tokenizer_ops as tokops


# LOADING

def load_model_tok_data1(config_dict):
    toks_dir, _, models_dir = ops.get_directory_paths(config_dict)
    tokenizer = tokops.load_latest_tokenizer(toks_dir)
    dataset = tokops.get_dataset(tokenizer, config_dict, split = config_dict["data_split"])
    model = train_t5.load_latest_model(models_dir)
    return model, tokenizer, dataset

def load_model_tok_dataN(config_dict, accelerator):
    toks_dir, _, models_dir = ops.get_directory_paths(config_dict)
    if accelerator.is_main_process:
        tokenizer = tokops.load_latest_tokenizer(toks_dir)
        dataset = tokops.get_dataset(tokenizer, config_dict, split = config_dict["data_split"])
        model = train_t5.load_latest_model(models_dir)
    else:
        tokenizer = None
        dataset = None
        model = None
    
    accelerator.wait_for_everyone()
    tokenizer = broadcast_object_list([tokenizer])[0]
    dataset = broadcast_object_list([dataset])[0]
    model = broadcast_object_list([model])[0]
    return model, tokenizer, dataset

def load_model_tok_data(config_dict, accelerator=None):
    if accelerator is None or accelerator.num_processes <= 1:
        return load_model_tok_data1(config_dict)
    else:
        return load_model_tok_dataN(config_dict, accelerator)
    
def prepare_model_and_dataloader1(model, tokenizer, dataset, batch_size=8):
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    logging.info(f"Prepared model and dataloader.")
    return model, dataloader

def prepare_model_and_dataloaderN(model, tokenizer, dataset, accelerator, batch_size=8):
    batch_size = batch_size // accelerator.num_processes
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    dataloader, model = accelerator.prepare(dataloader, model)
    logging.info(f"Prepared model and dataloader.")
    return model, dataloader

def prepare_model_and_dataloader(model, tokenizer, dataset, batch_size=8, accelerator=None):
    if accelerator is None or accelerator.num_processes <= 1:
        return prepare_model_and_dataloader1(model, tokenizer, dataset, batch_size=batch_size)
    else:
        return prepare_model_and_dataloaderN(model, tokenizer, dataset, accelerator, batch_size=batch_size)

# VALIDATION

def step_validation(batch_idx, batch, metrics, model=None):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    outputs = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        labels=labels
    )

    loss = outputs.loss.item()
    predictions = outputs.logits.argmax(dim=-1)
    non_pad_mask = labels != -100
    corrects = (predictions == labels)
    non_pad_corrects = (corrects & non_pad_mask)
    non_pad_corrects_sum = non_pad_corrects.sum().item()
    num_non_pad_toks = non_pad_mask.sum().item()

    metrics["step"] = batch_idx + 1
    metrics["loss_sum"] += loss
    metrics["losses"].append(loss)
    metrics["non_pad_corrects_sum"] += non_pad_corrects_sum
    metrics["total_non_pad_toks"] += num_non_pad_toks
    metrics["accuracies"].append(non_pad_corrects_sum / num_non_pad_toks)
    metrics["first_correct_sum"] += corrects[:, 0].sum().item()
    metrics["total_first_toks"] += torch.numel(labels[:, 0])
    return metrics 

def report_validation1(metrics, records):
    records["steps"].append(metrics["step"])
    records["avg_loss"].append(metrics["loss_sum"] / metrics["step"])
    records["non_pad_accuracy"].append(metrics["non_pad_corrects_sum"] / metrics["total_non_pad_toks"])
    records["first_token_accuracy"].append(metrics["first_correct_sum"] / metrics["total_first_toks"])
    ops.save_dict_as_json(records, "validation_records.json")
    return records

def report_validationN(metrics, records, accelerator):
    accelerator.wait_for_everyone()
    local_vals = torch.tensor([
        metrics["step"],
        metrics["loss_sum"],
        metrics["non_pad_corrects_sum"],
        metrics["total_non_pad_toks"],
        metrics["first_correct_sum"],
        metrics["total_first_toks"]
    ], device=accelerator.device)

    global_step, global_loss_sum, global_non_pad_corrects, global_non_pad_toks, global_first_corrects, global_first_toks = accelerator.reduce(local_vals, reduction="sum")

    if accelerator.is_main_process:
        records["steps"].append(metrics["step"])
        records["avg_loss"].append(global_loss_sum.item() / global_step.item())
        records["non_pad_accuracy"].append(global_non_pad_corrects.item() / global_non_pad_toks.item())
        records["first_token_accuracy"].append(global_first_corrects.item() / global_first_toks.item())
        ops.save_dict_as_json(records, "validation_records.json")
    else:
        records = None
    accelerator.wait_for_everyone()
    records = broadcast_object_list([records])[0]
    return records

def report_validation(metrics, records, accelerator=None):
    if accelerator is None or accelerator.num_processes <= 1:
        records = report_validation1(metrics, records)
        accel_prefix = ""
    else:
        records = report_validationN(metrics, records, accelerator)
        accel_prefix = f"{accelerator.process_index}: "
    
    log_message = f"""{accel_prefix}Latest step ({records['steps'][-1]}) completed. 
        Current average loss is {records['avg_loss'][-1]}."""
    logging.info(log_message)
    return records

# MATCHING

def get_matches(metrics, ins_outs_predicts, max_matches=20):
    match_count = 0
    mismatch_count = 0
    coincide_count = 0
    for input, target, predict in ins_outs_predicts:
        metrics["total_count"] += 1
        if predict.strip() == target.strip():
            metrics["exacts_count"] += 1
            if match_count < 2 and len(metrics["exact_matches"]) < max_matches:
                metrics["exact_matches"].append((input, target, predict))
                match_count += 1
        elif mismatch_count < 2 and len(metrics["mismatches"]) < max_matches:
            metrics["mismatches"].append((input, target, predict))
            mismatch_count += 1

        common_prefix = os.path.commonprefix([target, predict])
        if common_prefix:
            metrics["coincide_count"] += 1
            if coincide_count < 2 and len(metrics["coincidences"]) < max_matches:
                metrics["coincidences"].append((input, target, predict))
                coincide_count += 1
    return metrics
    
def step_matching(batch_idx, batch, metrics, tokenizer=None, model=None, max_length=512, max_matches=20):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']
    labels = torch.where(labels == -100, tokenizer.pad_token_id, labels)

    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

    orig_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    orig_output = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    itps = zip(orig_input, orig_output, predictions)

    metrics["step"] = batch_idx + 1
    metrics = get_matches(metrics, itps, max_matches=max_matches)
    return metrics

def report_matching1(metrics, records, max_matches=20):
    records["steps"].append(metrics["step"])
    records["exact_accuracy"].append(metrics["exacts_count"] / metrics["total_count"])
    records["coincide_accuracy"].append(metrics["coincide_count"] / metrics["total_count"])
    if len(records["exact_matches"]) < max_matches:
        records["exact_matches"].extend(metrics["exact_matches"])
    if len(records["mismatches"]) < max_matches:
        records["mismatches"].extend(metrics["mismatches"])
    if len(records["coincidences"]) < max_matches:
        records["coincidences"].extend(metrics["coincidences"])
    ops.save_dict_as_json(records, "matching_records.json")
    return records

def report_matchingN(metrics, records, accelerator, max_matches=20):
    accelerator.wait_for_everyone()
    local_vals = torch.tensor([
        metrics["exacts_count"],
        metrics["total_count"],
        metrics["coincide_count"]
    ], device=accelerator.device)

    global_exact_count, global_total_count, global_coincide_count = accelerator.reduce(local_vals, reduction="sum")

    if accelerator.is_main_process:
        records["steps"].append(metrics["step"])
        records["exact_accuracy"].append(global_exact_count.item() / global_total_count.item())
        records["coincide_accuracy"].append(global_coincide_count.item() / global_total_count.item())
        if len(records["exact_matches"]) < max_matches:
            records["exact_matches"].extend(metrics["exact_matches"])
        if len(records["mismatches"]) < max_matches:
            records["mismatches"].extend(metrics["mismatches"])
        if len(records["coincidences"]) < max_matches:
            records["coincidences"].extend(metrics["coincidences"])
        ops.save_dict_as_json(records, "matching_records.json")
    else:
        records = None
    accelerator.wait_for_everyone()
    records = broadcast_object_list([records])[0]
    return records

def report_matching(metrics, records, accelerator=None, max_matches=20):
    if accelerator is None or accelerator.num_processes <= 1:
        records = report_matching1(metrics, records, max_matches=max_matches)
        accel_prefix = ""
    else:
        records = report_matchingN(metrics, records, accelerator, max_matches=max_matches)
        accel_prefix = f"{accelerator.process_index}: "

    log_message = f"""{accel_prefix}Latest step ({records['steps'][-1]}) completed. 
        Coincidence accuracy is {records['coincide_accuracy'][-1]}."""
    logging.info(log_message)
    return records

# REPLING

def group_paths_by_logic(data_dir, split):
    grouped = {}
    for path in tokops.generate_proof_paths(data_dir, split=split):
        rel_path = os.path.relpath(path, data_dir)
        logic = rel_path.split(os.sep)[0]
        if logic not in grouped:
            grouped[logic] = []
        grouped[logic].append(path)
    return grouped

# TODO: Create necessary repl methods for this function to work
def step_repling(config_dict, model, tokenizer, max_length=512):
    grouped = group_paths_by_logic(config_dict["data_dir"], config_dict["data_split"])
    generator = pipeline("text2text-generation", model=model, tokenizer=tokenizer)
    proof_counter = 0
    success_counter = 0
    for logic in grouped.keys():
        try:
            repl = REPL(logic)
            for path in grouped[logic]:
                try:
                    proof = proofs.get_proof_json(path)
                    acts = proofs.full_actions_of(proof)
                    user_state0 = repl.apply(acts[0])
                    x = ' '.join([acts[0], proofs.USER_STATE_SEP, user_state0])
                    predicts = generator(
                        x, 
                        max_length=max_length, num_return_sequences=5,
                        num_beams=5
                    )
                    for predict in predicts:
                        _ = repl.apply(predict)
                        err = repl.latest_error()
                        if not err:
                            success_counter += 1
                            break
                        _ = repl.go_back()
                    proof_counter += 1
                except Exception as e:
                    logging.warning(f"Error processing proof at {path}: {e}")
                finally:
                    repl.reset()
        except Exception as e:
            logging.warning(f"Error initializing REPL for {logic}: {e}")
        finally:
            repl.shutdown()
    return proof_counter, success_counter


# EVALUATION LOOP

METRICS = {
    "validation": {
        "step": step_validation,
        "report": report_validation,
        "metrics0": {
            "step": 0,
            "loss_sum": 0.0,
            "losses": [],
            "non_pad_corrects_sum": 0,
            "total_non_pad_toks": 0,
            "accuracies": [],
            "first_correct_sum": 0,
            "total_first_toks": 0
        },
        "records0": {
            "steps": [],
            "avg_loss": [],
            "non_pad_accuracy": [],
            "first_token_accuracy": []
        }
    },
    "matching": {
        "step": step_matching,
        "report": report_matching,
        "metrics0": {
            "step": 0,
            "total_count": 0,
            "exacts_count": 0,
            "coincide_count": 0,
            "exact_matches": [],
            "mismatches": [],
            "coincidences": []
        },
        "records0": {
            "steps": [],
            "exact_accuracy": [],
            "coincide_accuracy": [],
            "exact_matches": [],
            "mismatches": [],
            "coincidences": []
        }
    }
}

def execute(eval_metric, dataloader, model, log_steps = 10000, step_kwargs={}, report_kwargs={}):
    step = eval_metric["step"]
    report = eval_metric["report"]
    metrics = eval_metric["metrics0"]
    records = eval_metric["records0"]
    model.eval()
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            metrics = step(batch_idx, batch, metrics, **step_kwargs)
            if batch_idx % log_steps == 0:
                records = report(metrics, records, **report_kwargs) 
                logging.info(f"Recorded evalutation progress.")
    _ = report(metrics, records, **report_kwargs)

def with_metric(eval_str, config_dict):
    eval_metric = METRICS[eval_str]
    def general_body(accelerator):
        model, tokenizer, dataset = load_model_tok_data(config_dict, accelerator=accelerator)
        model, dataloader = prepare_model_and_dataloader(model, tokenizer, dataset, batch_size=config_dict["batch_size"], accelerator=accelerator)
        if eval_str == "validation":
            step_kwargs = {"model": model}
            report_kwargs = {"accelerator": accelerator}
        elif eval_str == "matching":
            step_kwargs = {
                "tokenizer": tokenizer, 
                "model": model, 
                "max_length": tokenizer.model_max_length, 
                "max_matches": 20
            }
            report_kwargs = {"max_matches": 20, "accelerator": accelerator}
        execute(
            eval_metric, 
            dataloader, 
            model, 
            log_steps=1000, 
            step_kwargs=step_kwargs,
            report_kwargs=report_kwargs
        )
        logging.info(f"Finished evaluation.")
    
    ops.wrap_w_accelerator(lambda acc: general_body(acc))


if __name__ == "__main__":
    ops.configure_logging("t5_eval.log")
    try:
        config_dict = ops.get_json_dict(ops.parse_config_path(tool_explanation="Evaluate the transformer as specified in the input JSON configuration."))
        ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)
    
    with_metric("validation", config_dict)