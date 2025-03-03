# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for training a Hugging Face T5 Model


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import logging
import numpy as np
import torch
import random

import isa_data
import ops
import tokenizer_ops as tokops

from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    AutoConfig, 
    T5ForConditionalGeneration, 
    DataCollatorForSeq2Seq, 
    get_scheduler
)
from accelerate.utils import broadcast_object_list


# CONFIGURATION

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=8):
    batch_size = batch_size // accelerator.num_processes
    # Dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    lr_scheduler = get_scheduler("constant", optimizer=optimizer)

    # Accelerate them
    train_dataloader = accelerator.prepare_data_loader(train_dataloader)
    valid_dataloader = accelerator.prepare_data_loader(valid_dataloader)
    log_dataloading(train_dataloader, accelerator)
    model, optimizer, lr_scheduler = accelerator.prepare(
            model, optimizer, lr_scheduler
    )
    return train_dataloader, valid_dataloader, model, optimizer, lr_scheduler

# LOGGING

def log_dataloading(dataloader, accelerator):
    for batch in dataloader:
        break
    pi = accelerator.process_index
    batch_shape = {k: v.shape for k, v in batch.items()}
    logging.info(f"{pi}: The first batch info is:")
    logging.info(f"{pi}: {batch_shape}")
    logging.info(f"{pi}: {batch['input_ids'][0][:10]}")

def log_exploding_gradients(model, batch_idx, accelerator, threshold=1e4):
    if batch_idx % 500 == 0:
        for name, param in model.named_parameters():
            if param.grad is not None:
                max_grad = param.grad.abs().max().item()
                if max_grad > threshold:
                    logging.warning(f"{accelerator.process_index}: Exploding gradient detected at batch {batch_idx}! Layer: {name}, max grad: {max_grad}")

def log_nan_inputs(batch_idx, batch, accelerator):
    for k, v in batch.items():
        if torch.isnan(v.float()).any():
            logging.warning(f"{accelerator.process_index}: NaN detected in {k} of batch {batch_idx}")
            logging.info(f"{accelerator.process_index}: batch {batch_idx} is {batch}")

def log_empty_labels(batch_idx, batch, accelerator):
    labels = batch["labels"]
    if labels.eq(-100).all():
        logging.warning(f"{accelerator.process_index}: Empty labels detected at batch {batch_idx}")
        logging.info(f"{accelerator.process_index}: batch {batch_idx} is {batch}")

def log_nan_loss(loss, batch_idx, batch, accelerator):
    if torch.isnan(loss).any():
        logging.error(f"{accelerator.process_index}: NaN loss detected at batch {batch_idx}")
        ops.save_batch(batch, f"nan_loss_batch{batch_idx}.pt")

# RETRIEVE MODEL

def initialize_model(model_name, finetuning, vocab_size, ctxt_length):
    config = AutoConfig.from_pretrained(model_name)
    vocab_size = config.vocab_size if finetuning else vocab_size
    config.update({
        "vocab_size": vocab_size,
        "n_positions": ctxt_length,
        "max_length": config.max_length,  # maximum generation length
        "d_model": config.d_model,
    })

    if finetuning:
        model = T5ForConditionalGeneration.from_pretrained(model_name, config=config)
    else:
        model = T5ForConditionalGeneration(config)
        logging.info(f"Trainig model from scratch.")
    logging.info(f"Initialized Hugging Face model of type {type(model)}.")
    return model

def load_latest_model(models_dir):
    latest_dir = ops.get_latest_dir_from(models_dir, adding_one=False)
    model = T5ForConditionalGeneration.from_pretrained(latest_dir)
    logging.info(f"Loaded Hugging Face model from {latest_dir} of type {type(model)}.")
    return model

def get_model(config_dict, vocab_size):
    _, _, models_dir = ops.get_directory_paths(config_dict, making_dirs=True)
    previous_model = ops.exists_previous("model", models_dir)

    if previous_model:
        model = load_latest_model(models_dir)
    else:
        ctxt_length = isa_data.get_context_length(config_dict["data_mode"])
        finetuning = True if config_dict["data_mode"].startswith("finetune") else False
        model = initialize_model(config_dict["model_name"], finetuning, vocab_size, ctxt_length)
    return model


# TRAINING LOOPS

def record(metrics, locals, accelerator, save_file="metrics.json"):
    accelerator.wait_for_everyone()
    local_stats = torch.tensor([locals["loss_sum"], locals["corrects_sum"], locals["valid_toks"], locals["train_step"]], device=accelerator.device)
    global_loss, global_corrects_sum, global_valid_toks, global_train_step = accelerator.reduce(local_stats, reduction="sum")
    if accelerator.is_main_process:
        avg_loss = global_loss.item() / global_train_step.item()
        metrics["loss"].append(avg_loss)
        metrics["accuracy"].append(global_corrects_sum.item() / global_valid_toks.item())
        metrics["steps"].append(global_train_step.item())
        logging.info(f"Current step's ({locals["train_step"]}) average loss is {avg_loss:.4f}")
        ops.save_dict_as_json(metrics, save_file)
    accelerator.wait_for_everyone()
    return metrics

def validate(model, dataloader, epoch, accelerator):
    model.eval()
    process_idx = accelerator.process_index
    metrics = {"loss": [], "accuracy": [], "steps": []}
    locals = {"loss_sum": 0.0, "corrects_sum": 0, "valid_toks": 0, "train_step": 0}
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                labels=batch["labels"]
            )
            loss = outputs.loss

            # Metrics
            predictions = outputs.logits.argmax(dim=-1)
            valids_mask = batch["labels"] != -100 # tokenizer.pad_token_id
            corrects = (predictions[valids_mask] == batch["labels"][valids_mask]).sum().item()
            locals["corrects_sum"] += corrects
            locals["valid_toks"] += valids_mask.sum().item()
            locals["loss_sum"] += loss.item()
            locals["train_step"] = batch_idx + 1

            if batch_idx % 1000 == 0:
                logging.info(f"{process_idx}: valid step number {batch_idx}")
                metrics = record(metrics, locals, accelerator, save_file=f"valid_metrics{epoch}.json")
    metrics = record(metrics, locals, accelerator, save_file=f"valid_metrics{epoch}.json")

def train(model, dataloader, optimizer, lr_scheduler, epoch, config_dict, accelerator):
    model.train()
    process_idx = accelerator.process_index
    metrics = {"loss": [], "accuracy": [], "steps": []}
    locals = {"loss_sum": 0.0, "corrects_sum": 0, "valid_toks": 0, "train_step": 0}
    for batch_idx, batch in enumerate(dataloader):
        # model.forward() and loss calculation
        outputs = model(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"]
        )
        loss = outputs.loss
        log_nan_loss(loss, batch_idx, batch, accelerator)
        
        # Backpropagation
        optimizer.zero_grad()
        accelerator.backward(loss) # loss.backward()
        # log_exploding_gradients(model, batch_idx, accelerator)
        optimizer.step()
        lr_scheduler.step()

        # Metrics
        predictions = outputs.logits.argmax(dim=-1)
        valids_mask = batch["labels"] != -100
        corrects = (predictions[valids_mask] == batch["labels"][valids_mask]).sum().item()
        locals["corrects_sum"] += corrects
        locals["valid_toks"] += valids_mask.sum().item()
        locals["loss_sum"] += loss.item()
        locals["train_step"] = batch_idx + 1

        # progress feedback
        if batch_idx % 1000 == 0:
            logging.info(f"{process_idx}: train step number {batch_idx}")
            metrics = record(metrics, locals, accelerator, save_file=f"train_metrics{epoch}.json")
        if batch_idx % 10000 == 0 and batch_idx > 0:
                if accelerator.is_main_process:
                    _, _, models_dir = ops.get_directory_paths(config_dict)
                    ops.save_hf_data_in(accelerator.unwrap_model(model), models_dir)
        accelerator.wait_for_everyone()

    logging.info(f"{process_idx}: Total number of batches was {batch_idx + 1}")
    logging.info(f"{process_idx}: Final learning rate was: {lr_scheduler.get_last_lr()[0]}")
    _ = record(metrics, locals, accelerator, save_file=f"valid_metrics{epoch}.json")
            
def do_epochs(train_dataloader, valid_dataloader, model, optimizer, lr_scheduler, accelerator, config_dict):
    num_epochs = config_dict["num_epochs"]
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1} of {num_epochs}")
        train(model, train_dataloader, optimizer, lr_scheduler, epoch, config_dict, accelerator)
        if accelerator.is_main_process:
            _, _, models_dir = ops.get_directory_paths(config_dict)
            ops.save_hf_data_in(accelerator.unwrap_model(model), models_dir)
            logging.info(f"Finished training loop. Checkpoint saved for epoch {epoch}.")
        accelerator.wait_for_everyone()
        validate(model, valid_dataloader, epoch, accelerator)
    if accelerator.is_main_process:
        logging.info("Training complete.")


# MAIN

def load_model_tok_data(accelerator, config_dict):
    if accelerator.is_main_process:
        # Tokenizer
        tokenizer = tokops.get_trained_tokenizer(config_dict, making_dirs=True)
        vocab_size = len(tokenizer)

        # Data
        train_data = tokops.get_dataset(tokenizer, config_dict, split = isa_data.SPLITS["TRAIN"])
        valid_data = tokops.get_dataset(tokenizer, config_dict, split = isa_data.SPLITS["VALID"])

        # Model
        model = get_model(config_dict, vocab_size)
    else:
        tokenizer = None
        train_data, valid_data = None, None
        model = None

    accelerator.wait_for_everyone()
    tokenizer = broadcast_object_list([tokenizer])[0]
    train_data = broadcast_object_list([train_data])[0]
    valid_data = broadcast_object_list([valid_data])[0]
    model = broadcast_object_list([model])[0]
    return model, tokenizer, train_data, valid_data

# TODO: make batch_size, lr, and vocab_size configurable from configuration JSON
def main(accelerator, config_dict):
    model, tokenizer, train_data, valid_data = load_model_tok_data(accelerator, config_dict)

    train_dataloader, valid_dataloader, model, optimizer, lr_scheduler = prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=config_dict["batch_size"])

    do_epochs(train_dataloader, valid_dataloader, model, optimizer, lr_scheduler, accelerator, config_dict)

if __name__ == "__main__":
    set_all_seeds(42)
    ops.configure_logging("t5_train.log")
    try:
        config_dict = ops.get_json_dict(ops.parse_config_path(tool_explanation="Train the transformer as specified in the input JSON configuration."))
        ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)

    ops.wrap_w_accelerator(lambda acc: main(acc, config_dict))
