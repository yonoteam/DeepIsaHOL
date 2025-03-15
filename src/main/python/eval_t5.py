# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for evaluating a Hugging Face T5 Model

import logging

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from accelerate.utils import broadcast_object_list

import ops
import train_t5
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
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length", max_length=model.config.n_positions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    logging.info(f"Prepared model and dataloader.")
    return model, dataloader

def prepare_model_and_dataloaderN(model, tokenizer, dataset, accelerator, batch_size=8):
    batch_size = batch_size // accelerator.num_processes
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length", max_length=model.config.n_positions)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    dataloader, model = accelerator.prepare(dataloader, model)
    logging.info(f"Prepared model and dataloader.")
    return model, dataloader

def prepare_model_and_dataloader(model, tokenizer, dataset, batch_size=8, accelerator=None):
    if accelerator is None or accelerator.num_processes <= 1:
        model, dataloader = prepare_model_and_dataloader1(model, tokenizer, dataset, batch_size=batch_size)
    else:
        model, dataloader = prepare_model_and_dataloaderN(model, tokenizer, dataset, accelerator, batch_size=batch_size)
    for batch in dataloader:
        vals_shape = {k: v.shape for k, v in batch.items()}
        logging.info(f"The first batch info is:")
        logging.info(f"{vals_shape}")
        break
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
def upd_with_mask(
        upd_list: list, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        predicts: torch.Tensor, 
        mask: torch.Tensor, 
        max_matches=20
    ):
    if len(upd_list) < max_matches:
        positions_of_true = mask.nonzero(as_tuple=True)[0]
        if positions_of_true.numel() > 0:
            fst_pos = positions_of_true[0].item()
            upd_list.append([inputs[fst_pos].tolist(), targets[fst_pos].tolist(), predicts[fst_pos].tolist()])
    return upd_list

def get_matches(
        metrics: dict, 
        inputs: torch.Tensor, 
        targets: torch.Tensor, 
        predicts: torch.Tensor,
        pad_token:int = 0, 
        max_matches:int = 20
    ):
    def start_is_padded(t1):
        return torch.all(t1[:,0] == pad_token).item()
    
    if start_is_padded(predicts):
        predicts = predicts[:,1:]

    if targets.size(1) > predicts.size(1):
        trim_max = predicts.size(1)
        trimmed = targets.narrow(1, 0, trim_max)
    else: # assuming equal size in remaining dimensions
        trim_max = targets.size(1)
        trimmed = targets
    
    # total count
    batch_size = targets.size(0)
    metrics["total_count"] += batch_size

    # exact matches
    exact_mask = torch.all(trimmed == predicts[:,:trim_max], dim=1)
    metrics["exacts_count"] += exact_mask.sum().item()
    metrics["exact_matches"] = upd_with_mask(metrics["exact_matches"], inputs, targets, predicts, exact_mask, max_matches=max_matches)
    
    # coincidences
    coincide_mask = (trimmed[:,0] == predicts[:,0]) & ~exact_mask
    metrics["coincide_count"] += coincide_mask.sum().item()
    metrics["coincidences"] = upd_with_mask(metrics["coincidences"], inputs, targets, predicts, coincide_mask, max_matches=max_matches)

    # completely wrong
    wrong_mask = torch.all(trimmed != predicts[:,:trim_max], dim=1)
    metrics["all_wrong_count"] += wrong_mask.sum().item()
    metrics["mismatches"] = upd_with_mask(metrics["mismatches"], inputs, targets, predicts, wrong_mask, max_matches=max_matches)

    return metrics
    
def step_matching(batch_idx, batch, metrics, model=None, pad_token=0, eos_token=1, max_matches=20):
    input_ids = batch['input_ids']
    attention_mask = batch['attention_mask']
    labels = batch['labels']

    pad_mask = labels == -100
    labels[pad_mask] = pad_token
    eos_positions = (labels == eos_token).nonzero(as_tuple=True)[1]
    max_eos_position = eos_positions.max() + 5 if eos_positions.numel() > 0 else None


    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model = model.module

    predictions = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_eos_position)

    metrics["step"] = batch_idx + 1
    metrics = get_matches(metrics, input_ids, labels, predictions, pad_token=pad_token, max_matches=max_matches)
    return metrics

def report_matching1(metrics, records):
    records["steps"].append(metrics["step"])
    records["exact_accuracy"].append(metrics["exacts_count"] / metrics["total_count"])
    records["coincide_accuracy"].append(metrics["coincide_count"] / metrics["total_count"])
    records["all_wrong_ratio"].append(metrics["all_wrong_count"] / metrics["total_count"])
    records["exact_matches"] = metrics["exact_matches"]
    records["mismatches"] = metrics["mismatches"]
    records["coincidences"] = metrics["coincidences"]
    ops.save_dict_as_json(records, "matching_records.json")
    return records

def report_matchingN(metrics, records, accelerator):
    accelerator.wait_for_everyone()
    local_vals = torch.tensor([
        metrics["exacts_count"],
        metrics["total_count"],
        metrics["coincide_count"],
        metrics["all_wrong_count"]
    ], device=accelerator.device)

    global_exact_count, global_total_count, global_coincide_count, global_wrongs_count = accelerator.reduce(local_vals, reduction="sum")

    if accelerator.is_main_process:
        records["steps"].append(metrics["step"])
        records["exact_accuracy"].append(global_exact_count.item() / global_total_count.item())
        records["coincide_accuracy"].append(global_coincide_count.item() / global_total_count.item())
        records["all_wrong_ratio"].append(global_wrongs_count.item() / global_total_count.item())
        records["exact_matches"] = metrics["exact_matches"]
        records["mismatches"] = metrics["mismatches"]
        records["coincidences"] = metrics["coincidences"]
        ops.save_dict_as_json(records, "matching_records.json")
    else:
        records = None
    accelerator.wait_for_everyone()
    records = broadcast_object_list([records])[0]
    return records

def report_matching(metrics, records, accelerator=None):
    if accelerator is None or accelerator.num_processes <= 1:
        records = report_matching1(metrics, records)
        accel_prefix = ""
    else:
        records = report_matchingN(metrics, records, accelerator)
        accel_prefix = f"{accelerator.process_index}: "

    log_message = f"""{accel_prefix}Latest step ({records['steps'][-1]}) completed. 
        Coincidence accuracy is {records['coincide_accuracy'][-1]}."""
    logging.info(log_message)
    return records


# EVALUATION LOOP

METRICS = {
    "validation": {
        "step": step_validation,
        "report": report_validation,
        "metrics": {
            "step": 0,
            "loss_sum": 0.0,
            "losses": [],
            "non_pad_corrects_sum": 0,
            "total_non_pad_toks": 0,
            "accuracies": [],
            "first_correct_sum": 0,
            "total_first_toks": 0
        },
        "records": {
            "steps": [],
            "avg_loss": [],
            "non_pad_accuracy": [],
            "first_token_accuracy": []
        }
    },
    "matching": {
        "step": step_matching,
        "report": report_matching,
        "metrics": {
            "step": 0,
            "total_count": 0,
            "exacts_count": 0,
            "coincide_count": 0,
            "all_wrong_count": 0,
            "exact_matches": [],
            "coincidences": [],
            "mismatches": []
        },
        "records": {
            "steps": [],
            "exact_accuracy": [],
            "coincide_accuracy": [],
            "all_wrong_ratio": [],
            "exact_matches": [],
            "coincidences": [],
            "mismatches": []
        }
    }
}

def execute(eval_metric, dataloader, model, log_steps = 1000, step_kwargs={}, report_kwargs={}):
    step = eval_metric["step"]
    report = eval_metric["report"]
    metrics = eval_metric["metrics"]
    records = eval_metric["records"]
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
        elif eval_str == "matching":
            step_kwargs = {
                "model": model, 
                "pad_token": tokenizer.pad_token_id,
                "eos_token": tokenizer.eos_token_id,
                "max_matches": 20
            }
        report_kwargs = {"accelerator": accelerator}
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
    
    with_metric("matching", config_dict)