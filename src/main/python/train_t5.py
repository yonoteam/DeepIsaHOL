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
import accelerate_test

from torch.utils.data import DataLoader
from transformers import (
    set_seed,
    AutoConfig, 
    T5ForConditionalGeneration, 
    DataCollatorForSeq2Seq, 
    get_scheduler
)
from datasets import IterableDataset
from accelerate import Accelerator
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

# TODO: add support for HF datasets library
def load_datasets(tokenizer, mode, data_dir):
    train_data = IterableDataset.from_generator(tokops.generate_model_inputs, gen_kwargs={'tokenizer': tokenizer, 'json_data_dir': data_dir, 'split': 'train', 'mode': mode})
    valid_data = IterableDataset.from_generator(tokops.generate_model_inputs, gen_kwargs={'tokenizer': tokenizer, 'json_data_dir': data_dir, 'split': 'valid', 'mode': mode})
    test_data = IterableDataset.from_generator(tokops.generate_model_inputs, gen_kwargs={'tokenizer': tokenizer, 'json_data_dir': data_dir, 'split': 'test', 'mode': mode})
    logging.info(f"Datasets loaded.")
    return train_data, valid_data, test_data

def get_datasets(remote, mode, tokenizer, data_dir, datasets_dir):
    if remote:
         tokops.make_and_save_datasets(tokenizer, data_dir, datasets_dir, mode=mode)
    train_data, valid_data, test_data = load_datasets(tokenizer, mode, data_dir)
    return train_data, valid_data, test_data

# MODEL

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

# TODO: make batch_size, lr, and vocab_size configurable from configuration JSON
# TODO: make a get_size method for the dataset used
def train(model, train_dataloader, valid_dataloader, num_epochs, models_dir, accelerator):
    try:
        # Optimizer, dataloaders, and Scheduler
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
        lr_scheduler = get_scheduler("constant", optimizer=optimizer)
        train_dataloader, valid_dataloader, model, optimizer, lr_scheduler = accelerator.prepare(
            train_dataloader, valid_dataloader, model, optimizer, lr_scheduler
        )

        model.train()
        for epoch in range(num_epochs):
            logging.info(f"Epoch {epoch + 1} of {num_epochs}")
            train_loss = 0.0
            
            for batch_idx, batch in enumerate(train_dataloader):
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
                if batch_idx % 100 == 0:
                    logging.info(f"Train step number {batch_idx}")
            
            logging.info(f"Total number of steps was {batch_idx + 1}")
            avg_train_loss = train_loss / (batch_idx + 1)
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logging.info(f"Average Training Loss: {avg_train_loss:.4f}, LearnRate: {lr_scheduler.get_last_lr()[0]}")
                tokops.save_hf_data_in(accelerator.unwrap_model(model), models_dir)
                logging.info(f"Checkpoint saved for epoch {epoch}")

            # Validation Loop
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch_idx, batch in enumerate(valid_dataloader):
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
                logging.info("Training complete.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


# ALGORITHM
def main(config_dict):
    data_dir, all_models_dir, model_name, mode, num_epochs = ops.extract_params(config_dict)

    try:
        accelerator = Accelerator(mixed_precision="fp16")
        if accelerator.is_main_process:
            logging.info(f"Accelerator started on {accelerator.num_processes} processes.")
        
        accelerate_test.log_cuda_info(accelerator)
        
        model_remote, _, _ = get_remotes(config_dict)
        if accelerator.is_main_process:
            logging.info(f"Model has to be retrieved remotely?: {model_remote}")
            logging.info(f"Dataset has to be retrieved remotely?: {False}")
            logging.info(f"Tokenizer has to be retrieved remotely?: {False}")

        tokenizers_dir, _, models_dir = ops.get_directory_paths(config_dict)
        if accelerator.is_main_process:
            os.makedirs(models_dir, exist_ok=True)

        if accelerator.is_main_process:
            # Tokenizer
            tokenizer = tokops.load_latest_tokenizer(tokenizers_dir)
            vocab_size = len(tokenizer)

            # Data
            train_data, valid_data, _ = load_datasets(tokenizer, mode, data_dir)

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

        # Data Collator and Loaders
        data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

        batch_size = 8 // accelerator.num_processes
        train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
        valid_dataloader = DataLoader(valid_data, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

        # Training loop
        train(model, train_dataloader, valid_dataloader, num_epochs, models_dir, accelerator)
    
    except Exception as e:
        logging.error(f"Error in main: {e}")
        raise
    finally:
        accelerator.wait_for_everyone()
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                logging.error(f"Error destroying process group: {str(e)}")
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    set_all_seeds(42)
    ops.configure_logging("train_t5_test.log")
    try:
        config_dict = ops.get_config_dict(ops.parse_config_path(tool_explanation="Train the transformer as specified in the input JSON configuration."))
        params = ops.extract_params(config_dict)
        ops.check_params(params[0], params[1])
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)

    main(config_dict)
