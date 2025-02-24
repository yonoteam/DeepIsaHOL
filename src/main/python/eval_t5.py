# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for evaluating a Hugging Face T5 Model

import os
import logging

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from accelerate.utils import broadcast_object_list

import ops
import train_t5
import tokenizer_ops as tokops

def load_model_tok_data(config_dict, accelerator=None):
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

def prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=8):
    batch_size = batch_size // accelerator.num_processes
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    dataloader, model = accelerator.prepare(dataloader, model)
    return model, dataloader

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
    valid_unpad_mask = labels != 0
    corrects = (predictions == labels)
    unpadded_corrects = (corrects & valid_unpad_mask)
    unpadded_corrects_sum = unpadded_corrects.sum().item()
    num_unpadded_toks = valid_unpad_mask.sum().item()

    metrics["step"] = batch_idx + 1
    metrics["loss_sum"] += loss
    metrics["losses"].append(loss)
    metrics["unpadded_corrects_sum"] += unpadded_corrects_sum
    metrics["total_unpadded_toks"] += num_unpadded_toks
    metrics["accuracies"].append(unpadded_corrects_sum / num_unpadded_toks)
    metrics["first_correct_sum"] += corrects[:, 0].sum().item()
    metrics["total_first_toks"] += torch.numel(labels[:, 0])
    metrics["correct_toks_sum"] += corrects.sum().item()
    metrics["total_toks"] += torch.numel(labels)
    return metrics 

def report_validation1(metrics, records):
    records["latest_step"] = metrics["step"]
    records["loss_sum"].append(metrics["loss_sum"])
    records["unpadded_accuracy"].append(metrics["unpadded_corrects_sum"] / metrics["total_unpadded_toks"])
    records["first_token_accuracy"].append(metrics["first_correct_sum"] / metrics["total_first_toks"])
    records["prediction_accuracy"].append(metrics["correct_toks_sum"] / metrics["total_toks"])
    ops.save_dict_as_json(records, "validation_records.json")
    return records

def report_validationN(metrics, records, accelerator):
    accelerator.wait_for_everyone()
    local_vals = torch.tensor([
        metrics["step"],
        metrics["loss_sum"],
        metrics["unpadded_corrects_sum"],
        metrics["total_unpadded_toks"],
        metrics["first_correct_sum"],
        metrics["total_first_toks"],
        metrics["correct_toks_sum"],
        metrics["total_toks"]
    ], device=accelerator.device)

    if accelerator.is_main_process:
        global_step, global_loss_sum, global_unpadded_corrects, global_unpadded_toks, global_first_corrects, global_first_toks, global_correct_predicted, global_num_toks = accelerator.reduce(local_vals, reduction="sum")

        records["latest_step"] = global_step.item()
        records["loss_sum"].append(global_loss_sum.item())
        records["unpadded_accuracy"].append(global_unpadded_corrects.item() / global_unpadded_toks.item())
        records["first_token_accuracy"].append(global_first_corrects.item() / global_first_toks.item())
        records["prediction_accuracy"].append(global_correct_predicted.item() / global_num_toks.item())
        ops.save_dict_as_json(records, "validation_records.json")
    accelerator.wait_for_everyone()
    return records

def report_validation(metrics, records, accelerator=None):
    if accelerator is None or accelerator.num_processes <= 1:
        records = report_validation1(metrics, records)
    else:
        records = report_validationN(metrics, records, accelerator)
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
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)

    orig_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
    orig_output = tokenizer.batch_decode(labels, skip_special_tokens=True)
    predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    itps = zip(orig_input, orig_output, predictions)

    metrics["step"] = batch_idx + 1
    metrics = get_matches(metrics, itps, max_matches=max_matches)
    return metrics

def report_matching1(metrics, records):
    records["latest_step"] = metrics["step"]
    records["exact_accuracy"].append(metrics["exacts_count"] / metrics["total_count"])
    records["coincide_accuracy"].append(metrics["coincide_count"] / metrics["total_count"])
    ops.save_dict_as_json(records, "matching_records.json")
    return records

def report_matchingN(metrics, records, accelerator):
    accelerator.wait_for_everyone()
    local_vals = torch.tensor([
        metrics["step"],
        metrics["exacts_count"],
        metrics["total_count"],
        metrics["coincide_count"]
    ], device=accelerator.device)

    if accelerator.is_main_process:
        global_step, global_exact_count, global_total_count, global_coincide_count = accelerator.reduce(local_vals, reduction="sum")

        records["latest_step"] = global_step.item()
        records["exact_accuracy"].append(global_exact_count.item() / global_total_count.item())
        records["coincide_accuracy"].append(global_coincide_count.item() / global_total_count.item())
        ops.save_dict_as_json(records, "matching_records.json")
    accelerator.wait_for_everyone()
    return records

def report_matching(metrics, records, accelerator=None):
    if accelerator is None or accelerator.num_processes <= 1:
        records = report_matching1(metrics, records)
    else:
        records = report_matchingN(metrics, records, accelerator)
    return records

# EVALUATION LOOP

METRICS = {
    "validation": {
        "step": step_validation,
        "report": report_validation,
        "metrics0": {
            "step": 0,
            "loss_sum": 0.0,
            "losses": [],
            "unpadded_corrects_sum": 0,
            "total_unpadded_toks": 0,
            "accuracies": [],
            "first_correct_sum": 0,
            "total_first_toks": 0,
            "correct_toks_sum": 0,
            "total_toks": 0
        },
        "records0": {
            "latest_step": 0,
            "loss_sum": [],
            "unpadded_accuracy": [],
            "first_token_accuracy": [],
            "prediction_accuracy": []
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
            "latest_step": 0,
            "exact_accuracy": [],
            "coincide_accuracy": []
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
    _ = report(metrics, records, **report_kwargs)

def with_metric(eval_str, config_dict):
    eval_metric = METRICS[eval_str]
    def general_body(accelerator):
        model, tokenizer, dataset = load_model_tok_data(config_dict, accelerator=accelerator)
        model, dataloader = prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=config_dict["batch_size"])
        if eval_str == "validation":
            step_kwargs = {"model": model}
        elif eval_str == "matching":
            step_kwargs = {
                "tokenizer": tokenizer, 
                "model": model, 
                "max_length": tokenizer.model_max_length, "max_matches": 20
            }
        report_kwargs = {"accelerator": accelerator}
        execute(
            eval_metric, 
            dataloader, 
            model, 
            log_steps=10000, 
            step_kwargs=step_kwargs,
            report_kwargs=report_kwargs
        )
    
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