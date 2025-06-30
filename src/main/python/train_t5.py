# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for training a Hugging Face T5 Model


# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"
import os
import logging
import numpy as np
import torch
import random

import tokenizer_ops as tokops

from torch.utils.data import DataLoader
from datasets import IterableDataset
from transformers import (
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    set_seed,
    AutoConfig,
    T5ForConditionalGeneration,
    get_scheduler
)
from accelerate.utils import broadcast_object_list

import dicts
import distrib
import save_ops
import config_ops

# CONFIGURATION

def set_all_seeds(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    set_seed(seed)

def prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=8):
    # Dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding="max_length", max_length=model.config.n_positions)
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
    logging.info(f"{pi}: The first batch info is: shape = {batch_shape}")
    logging.info(f"{pi}: first tokens = {batch['input_ids'][0][:20]}")

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

def log_nan_loss(loss, batch_idx, accelerator):
    if torch.isnan(loss).any():
        logging.error(f"{accelerator.process_index}: NaN loss detected at batch {batch_idx}")

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
        logging.info(f"Fine-tuning model with context length = {model.config.n_positions}")
    else:
        model = T5ForConditionalGeneration(config)
        logging.info(f"Trainig model from scratch with context length = {model.config.n_positions}")
    logging.info(f"Initialized Hugging Face model of type {type(model)}.")
    return model

def load_latest_model(models_dir):
    latest_dir = save_ops.get_latest_dir(models_dir, adding_one=False)
    model = T5ForConditionalGeneration.from_pretrained(latest_dir)
    logging.info(f"Loaded Hugging Face model from {latest_dir} of type {type(model)}.")
    return model

def get_model(config_dict, vocab_size):
    _, models_dir = save_ops.get_dirs(config_dict, making_dirs=True)
    previous_model = save_ops.exists_previous("model", models_dir)

    if previous_model:
        model = load_latest_model(models_dir)
    else:
        ctxt_length = config_ops.get_context_length(config_dict["data_format"])
        finetuning = True if config_dict["data_format"].startswith("finetune") else False
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
        dicts.save_as_json(metrics, save_file)
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
        log_nan_loss(loss, batch_idx, accelerator)
        
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
                    _, models_dir = save_ops.get_dirs(config_dict)
                    save_ops.save_in(accelerator.unwrap_model(model), models_dir)
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
            _, models_dir = save_ops.get_dirs(config_dict)
            save_ops.save_in(accelerator.unwrap_model(model), models_dir)
            logging.info(f"Finished training loop. Checkpoint saved for epoch {epoch}.")
        accelerator.wait_for_everyone()
        validate(model, valid_dataloader, epoch, accelerator)
    if accelerator.is_main_process:
        logging.info("Training complete.")


# MAIN

def load_model_tok_data(accelerator, config_dict):
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        # Tokenizer
        tokenizer = tokops.get_trained_tokenizer(config_dict, making_dirs=True)
        vocab_size = len(tokenizer)

        # Data
        train_kwargs = {
            "tokenizer": tokenizer,
            "json_data_dir": config_dict["data_dir"],
            "split": "train",
            "data_format": config_dict["data_format"]
        }
        train_data = IterableDataset.from_generator(
            tokops.t5_tokked_model_inputs,
            gen_kwargs=train_kwargs
        )
        valid_kwargs = train_kwargs.copy()
        valid_kwargs["split"] = "valid"
        valid_data = IterableDataset.from_generator(
            tokops.t5_tokked_model_inputs,
            gen_kwargs=valid_kwargs
        )

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
    logging.info(f"{accelerator.process_index}: Successfully broadcasted data, the evidence is that the type of model is {type(model)}")
    return model, tokenizer, train_data, valid_data

def main(accelerator, config_dict):
    model, tokenizer, train_data, valid_data = load_model_tok_data(accelerator, config_dict)

    train_dataloader, valid_dataloader, model, optimizer, lr_scheduler = prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=config_dict["hf_training_arguments"]["per_device_train_batch_size"])

    do_epochs(train_dataloader, valid_dataloader, model, optimizer, lr_scheduler, accelerator, config_dict)

# ALTERNATIVE APPROACH WITH TRAINER

def compute_t5_metrics(eval_preds: EvalPrediction):
    predictions, labels = eval_preds.predictions, eval_preds.label_ids

    valid_mask = (labels != -100)
    predicted_token_ids = np.argmax(predictions, axis=-1) # most likely token IDs
    valid_predictions = predicted_token_ids[valid_mask]
    valid_labels = labels[valid_mask]
    corrects_sum = (valid_predictions == valid_labels).sum()
    valid_toks = valid_mask.sum()
    accuracy = corrects_sum.item() / valid_toks.item() if valid_toks.item() > 0 else 0.0
    return {"accuracy": accuracy}

def get_training_args(config_dict):
    pre_args = config_dict["hf_training_arguments"]
    batches_per_epoch = config_dict["batches_per_epoch"]

    pre_args["max_steps"] = config_dict["num_epochs"] * batches_per_epoch
    pre_args["logging_dir"] = os.getcwd(),
    pre_args["logging_steps"] = max(1, batches_per_epoch // 100)
    pre_args["eval_strategy"] = "steps"
    pre_args["eval_steps"] = batches_per_epoch
    pre_args["output_dir"] = config_dict["models_dir"]
    pre_args["overwrite_output_dir"] = True,
    pre_args["save_strategy"] = "steps"
    pre_args["save_total_limit"] = 5,
    pre_args["save_steps"] = batches_per_epoch
    
    train_args = TrainingArguments(**pre_args)
    return train_args

def main_alt(accelerator, config_dict):
    model, tokenizer, train_data, valid_data = load_model_tok_data(accelerator, config_dict)
    data_collator=DataCollatorForSeq2Seq(
        tokenizer, 
        model, 
        padding="max_length", 
        max_length=model.config.n_positions
    )
    train_args = get_training_args(config_dict)
    trainer = Trainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        eval_dataset=valid_data,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_t5_metrics
    )
    trainer = accelerator.prepare(trainer)
    
    train_results = trainer.train()
    trainer.save_metrics("all")

    if accelerator.is_main_process:
        logging.info(f"Main process: Training complete.")
        trainer.save_model() # load_best_model_at_end
        trainer.save_metrics("train_results", train_results.metrics)
        trainer.save_state()
        logging.info(f"Main process: Model saved.")

if __name__ == "__main__":
    set_all_seeds(42)
    explanation = "Train the transformer as specified in the input JSON configuration."
    config_dict = config_ops.parse_path(explanation)
    config_ops.setup_logging("t5_train.log")
    distrib.wrap_w_accelerator(lambda acc: main(acc, config_dict))
