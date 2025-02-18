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

def get_remotes(config_dict):
    tokenizers_dir, datasets_dir, models_dir = ops.get_directory_paths(config_dict)
    model_path = os.path.join(models_dir, "1", "model.safetensors")
    dataset_path = os.path.join(datasets_dir, "datasets.pt")
    tokenizer_path = os.path.join(tokenizers_dir, "1", "tokenizer.json")
    if os.path.isfile(model_path):
        model_remote = False
    else:
        model_remote = True
    if os.path.isfile(dataset_path):
        dataset_remote = False
    else:
        dataset_remote = True
    if os.path.isfile(tokenizer_path):
        tokenizer_remote = False
    else:
        tokenizer_remote = True
    return model_remote, dataset_remote, tokenizer_remote

def prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=8):
    batch_size = batch_size // accelerator.num_processes
    # Dataloaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    # Optimizer and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
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

def log_exploding_gradients(model, accelerator, threshold=1e4):
    for name, param in model.named_parameters():
        if param.grad is not None:
            max_grad = param.grad.abs().max().item()
            if max_grad > threshold:
                logging.warning(f"{accelerator.process_index}: Exploding gradient detected! Layer: {name}, max grad: {max_grad}")

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
        logging.info(f"{accelerator.process_index}: batch {batch_idx} is {batch}")

# RETRIEVE MODEL

def initialize_model_from_scratch(model_name, vocab_size):
    model = T5ForConditionalGeneration.from_pretrained(model_name)
    
    # reset the configuration for that model
    config = AutoConfig.from_pretrained(
        model_name,
        vocab_size=vocab_size,
        n_ctx=model.config.d_model
    )
    model = T5ForConditionalGeneration(config)
    logging.info(f"Initialized Hugging Face model of type {type(model)}.")
    return model

def load_latest_model(models_dir):
    latest_dir = ops.get_latest_dir_from(models_dir, adding_one=False)
    model = T5ForConditionalGeneration.from_pretrained(latest_dir)
    logging.info(f"Loaded Hugging Face model from {latest_dir} of type {type(model)}.")
    return model

def get_model(config_dict, vocab_size, remote=False):
    model_name = config_dict["model_name"]
    _, _, models_dir = ops.get_directory_paths(config_dict)
    if remote:
        model = initialize_model_from_scratch(model_name, vocab_size)
    else:
        model = load_latest_model(models_dir)
    return model


# TRAINING LOOPS

def validate(model, dataloader, accelerator):
    model.eval()
    valid_loss = 0.0
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            outputs = model(
                input_ids=batch["input_ids"], 
                attention_mask=batch["attention_mask"], 
                labels=batch["labels"]
            )
            loss = outputs.loss
            valid_loss += accelerator.gather(loss).sum().item() / accelerator.num_processes
    avg_valid_loss = valid_loss / (batch_idx + 1)
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info(f"Validation Loss: {avg_valid_loss:.4f}")

def train(model, dataloader, optimizer, lr_scheduler, accelerator):
    model.train()
    train_loss = 0.0
    for batch_idx, batch in enumerate(dataloader):
        # model.forward() and loss calculation
        outputs = model(
            input_ids=batch["input_ids"], 
            attention_mask=batch["attention_mask"], 
            labels=batch["labels"]
        )
        loss = outputs.loss

        # Backpropagation
        optimizer.zero_grad()
        accelerator.backward(loss) # loss.backward()
        optimizer.step()
        lr_scheduler.step()
        train_loss += accelerator.gather(loss).sum().item() / accelerator.num_processes

        # progress feedback
        if batch_idx % 1000 == 0:
            logging.info(f"Train step number {batch_idx}")
        
    logging.info(f"{accelerator.process_index}: Total number of batches was {batch_idx + 1}")
    accelerator.wait_for_everyone()
    avg_train_loss = train_loss / (batch_idx + 1)
    if accelerator.is_main_process:
        # total = accelerator.gather(torch.tensor([batch_idx + 1], device=accelerator.device)).sum().item()
        # logging.info(f"Total number of steps was {total}")
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}, LearnRate: {lr_scheduler.get_last_lr()[0]}")
            
def do_epochs(train_dataloader, valid_dataloader, model, optimizer, lr_scheduler, accelerator, config_dict):
    _, _, models_dir = ops.get_directory_paths(config_dict)
    num_epochs = config_dict["num_epochs"]
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1} of {num_epochs}")
        train(model, train_dataloader, optimizer, lr_scheduler, accelerator)
        if accelerator.is_main_process:
            ops.save_hf_data_in(accelerator.unwrap_model(model), models_dir)
            logging.info(f"Finished training loop. Checkpoint saved for epoch {epoch}.")
        validate(model, valid_dataloader, accelerator)
        accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        logging.info("Training complete.")


# MAIN

def load_model_tok_data(accelerator, config_dict, model_remote=False):
    tokenizers_dir, _, models_dir = ops.get_directory_paths(config_dict)
    if accelerator.is_main_process:
        os.makedirs(models_dir, exist_ok=True)

    if accelerator.is_main_process:
        # Tokenizer
        tokenizer = tokops.load_latest_tokenizer(tokenizers_dir)
        vocab_size = len(tokenizer)

        # Data
        train_data = tokops.get_dataset(tokenizer, config_dict, split = tokops.TRAIN)
        valid_data = tokops.get_dataset(tokenizer, config_dict, split = tokops.VALID)

        # Model
        model = get_model(config_dict, vocab_size, remote=model_remote)
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
    model_remote, _, _ = get_remotes(config_dict)
    if accelerator.is_main_process:
        logging.info(f"Model has to be retrieved remotely?: {model_remote}")
        logging.info(f"Dataset has to be retrieved remotely?: {False}")
        logging.info(f"Tokenizer has to be retrieved remotely?: {False}")

    model, tokenizer, train_data, valid_data = load_model_tok_data(accelerator, config_dict, model_remote)

    train_dataloader, valid_dataloader, model, optimizer, lr_scheduler = prepare_for_multi_train(model, tokenizer, train_data, valid_data, accelerator, batch_size=8)

    do_epochs(train_dataloader, valid_dataloader, model, optimizer, lr_scheduler, accelerator, config_dict)

if __name__ == "__main__":
    set_all_seeds(42)
    ops.configure_logging("train_t5.log")
    try:
        config_dict = ops.get_config_dict(ops.parse_config_path(tool_explanation="Train the transformer as specified in the input JSON configuration."))
        ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)

    ops.wrap_w_accelerator(lambda acc: main(acc, config_dict))
