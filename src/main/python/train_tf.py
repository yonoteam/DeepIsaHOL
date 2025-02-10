# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for training a Hugging Face T5 Model


import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import argparse
import json
import logging
import numpy as np
import torch
import random

import proofs
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

def configure_logging():
    # logfile to the current directory
    current_working_dir = os.getcwd()
    log_file = os.path.join(current_working_dir, "train_tf.log")
    
    # Set up logging configuration
    logging.basicConfig(
        filename=log_file,  # Log file in the current working directory
        level=logging.DEBUG,  # Capture all log levels
        format="%(asctime)s - %(levelname)s - %(message)s",
    )
    logging.info("Logging configured. Writing logs to %s", log_file)

def extract_params(config):
    try:
        data_dir = config["data_dir"]
        model_name = config["model_name"]
        all_models_dir = config["all_models_dir"]
        mode = config["mode"]
        num_epochs = int(config["num_epochs"])  # Ensure it's an integer
        return data_dir, all_models_dir, model_name, mode, num_epochs
    except KeyError as e:
        logging.error(f"Missing required configuration key: {e}")

def test_params(data_dir, all_models_dir):
    if not proofs.valid_data_dir(data_dir):
        message = f"""No subdirectory in '{data_dir}' contains  a 
        JSON file that starts with 'proof' and ends with '.json'.""".format()
        raise Exception(f"Error: {message}.")
    
    if not os.path.isdir(all_models_dir):
        message = f"""Input '{all_models_dir}' is not a directory."""
        raise Exception(f"Error: {message}")

def make_dir_vars(all_models_dir, model_name, mode):
    local_model_dir = os.path.join(all_models_dir, model_name)
    tokenizers_dir = os.path.join(local_model_dir, "tokenizers")
    datasets_dir = os.path.join(tokenizers_dir, f"datasets/{mode}")
    models_dir = os.path.join(local_model_dir, "models", mode)
    return tokenizers_dir, datasets_dir, models_dir

def get_remotes(all_models_dir, model_name, mode):
    tokenizers_dir, datasets_dir, models_dir = make_dir_vars(all_models_dir, model_name, mode)
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

# TOKENIZER

# works for both models and tokenizers
def save_hf_data_in(hf_data, saving_dir):
    # Determine the latest saving number
    subdirs = [
        os.path.join(saving_dir, d) for d in os.listdir(saving_dir)
        if os.path.isdir(os.path.join(saving_dir, d)) and d.isdigit()
    ]
    if subdirs:
        latest_number = max(int(os.path.basename(d)) for d in subdirs)
    else:
        latest_number = 0

    # Save the hugging face model or tokenizer
    new_dir = os.path.join(saving_dir, str(latest_number + 1))

    try:
        os.makedirs(new_dir, exist_ok=False)
        hf_data.save_pretrained(new_dir)
        logging.info(f"Saved Hugging Face data in {new_dir}")
    except FileExistsError:
        logging.warning(f"Directory '{new_dir}' already exists. Skipping save.")

# DATA
# TODO: add support for HF datasets library

# TODO: adapt for proofs.apply
def add_dir_data(mode_tok_data, json_data_dir):
    mode, tokenizer, train_data, valid_data, test_data = mode_tok_data
    for subdir, _, files in os.walk(json_data_dir):
        json_files = [file for file in files if file.startswith("proof") and file.endswith(".json")]
        total_proofs = len(json_files)
        train_size = int(total_proofs * 0.64)
        valid_size = int(total_proofs * 0.16)
        # test_size = total_proofs - train_size - valid_size
        for i, file in enumerate(json_files):
            json_path = os.path.join(subdir, file)
            proof_json = proofs.get_proof_json(json_path)
            if i < train_size:
                train_data = tokops.add_data_from_old((mode, tokenizer, train_data), proof_json)
            elif i < train_size + valid_size:
                valid_data = tokops.add_data_from_old((mode, tokenizer, valid_data), proof_json)
            else:
                test_data = tokops.add_data_from_old((mode, tokenizer, test_data), proof_json)
    return mode, tokenizer, train_data, valid_data, test_data

def make_datasets(mode, tokenizer, init_train, init_valid, init_test, data_dir):
    all_results = (mode, tokenizer, init_train, init_valid, init_test)
    imm_subdirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir()]
    for path in imm_subdirs:
        logging.info(f"Processing directory: {path}")
        all_results = add_dir_data(all_results, path)
    _, _, train_data, valid_data, test_data = all_results
    return train_data, valid_data, test_data

    
def save_datasets_in(train_data, valid_data, test_data, datasets_dir):
    dataset_dict = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
        }
    datasets_path = os.path.join(datasets_dir, 'datasets.pt')
    torch.save(dataset_dict, datasets_path)

def load_datasets(tokenizer, mode, data_dir):
    # datasets_path = os.path.join(datasets_dir, 'datasets.pt')
    # dataset_dict = torch.load(datasets_path, weights_only=True)
    # train_data = dataset_dict["train"]
    # valid_data = dataset_dict["valid"]
    # test_data = dataset_dict["test"]
    train_data = IterableDataset.from_generator(lambda: tokops.generate_model_inputs(tokenizer, data_dir, split='train', mode=mode))
    valid_data = IterableDataset.from_generator(lambda: tokops.generate_model_inputs(tokenizer, data_dir, split='valid', mode=mode))
    test_data = IterableDataset.from_generator(lambda: tokops.generate_model_inputs(tokenizer, data_dir, split='test', mode=mode))
    return train_data, valid_data, test_data

def get_datasets(remote, mode, tokenizer, data_dir, datasets_dir):
    # if remote:
    #     train_data, valid_data, test_data = make_datasets(mode, tokenizer, [], [], [], data_dir)
    #     save_datasets_in(train_data, valid_data, test_data, datasets_dir)
    train_data, valid_data, test_data = load_datasets(tokenizer, mode, data_dir)
    return train_data, valid_data, test_data

# MODEL

def get_init_model(remote, vocab_size, models_dir, model_name):
    if remote:
        # get hugging face transformers original model
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        
        # reset the configuration for that model
        config = AutoConfig.from_pretrained(
            model_name,
            vocab_size=vocab_size,
            n_ctx=model.config.d_model
        )
        model = T5ForConditionalGeneration(config)
        return model
    else:
        subdirs = [
                os.path.join(models_dir, d) for d in os.listdir(models_dir)
                if os.path.isdir(os.path.join(models_dir, d)) and d.isdigit()
            ]
        latest_dir = max(subdirs, key=lambda d: int(os.path.basename(d)))
        model = T5ForConditionalGeneration.from_pretrained(latest_dir)
        return model

# TODO: make batch_size, lr, and vocab_size configurable from configuration JSON
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
                    logging.info(f"Train step number {batch_idx} of {len(train_dataloader)}")

            avg_train_loss = train_loss / len(train_dataloader)
            
            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logging.info(f"Average Training Loss: {avg_train_loss:.4f}, LearnRate: {lr_scheduler.get_last_lr()[0]}")
                save_hf_data_in(accelerator.unwrap_model(model), models_dir)
                logging.info(f"Checkpoint saved for epoch {epoch}")

            # Validation Loop
            model.eval()
            valid_loss = 0.0
            with torch.no_grad():
                for batch in valid_dataloader:
                    outputs = model(
                        input_ids=batch["input_ids"], 
                        attention_mask=batch["attention_mask"], 
                        labels=batch["labels"]
                        )
                    loss = outputs.loss
                    valid_loss += accelerator.gather(loss).sum().item() / accelerator.num_processes

            avg_valid_loss = valid_loss / len(valid_dataloader)

            accelerator.wait_for_everyone()
            if accelerator.is_main_process:
                logging.info(f"Validation Loss: {avg_valid_loss:.4f}")
                logging.info("Training complete.")
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise


# ALGORITHM
def main(config):
    # Setup from config
    try:
        data_dir, all_models_dir, model_name, mode, num_epochs = extract_params(config)
        test_params(data_dir, all_models_dir)
    except Exception as e:
        logging.error(f"Could not setup from configuration file: '{e}'.")
        exit(1)

    try:
        accelerator = Accelerator(mixed_precision="fp16")
        if accelerator.is_main_process:
            logging.info(f"Accelerator started on {accelerator.num_processes} processes.")
        
        accelerate_test.log_cuda_info(accelerator)
        
        model_remote, dataset_remote, tokenizer_remote = get_remotes(all_models_dir, model_name, mode)
        if accelerator.is_main_process:
            logging.info(f"Model has to be retrieved remotely?: {model_remote}")
            logging.info(f"Dataset has to be retrieved remotely?: {dataset_remote}")
            logging.info(f"Tokenizer has to be retrieved remotely?: {tokenizer_remote}")

        tokenizers_dir, datasets_dir, models_dir = make_dir_vars(all_models_dir, model_name, mode)
        if accelerator.is_main_process:
            os.makedirs(datasets_dir, exist_ok=True)
            os.makedirs(models_dir, exist_ok=True)

        if accelerator.is_main_process:
            # Tokenizer
            tokenizer, tokenizer_dir = tokops.get_trained_tokenizer_and_tokdir(tokenizer_remote, data_dir, tokenizers_dir, model_name)
            vocab_size = len(tokenizer)
            logging.info(f"Tokenizer loaded. It's directory is: {tokenizer_dir}")

            # Data
            train_data, valid_data, test_data = get_datasets(dataset_remote, mode, tokenizer, data_dir, datasets_dir)
            logging.info(f"Datasets loaded. It's directory is: {datasets_dir}")

            # Model
            model = get_init_model(model_remote, vocab_size, models_dir, model_name)
            logging.info(f"Model loaded. It's directory is: {models_dir}")
        else:
            tokenizer = None
            train_data, valid_data, test_data = None, None, None
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
    # Configuration
    set_all_seeds(42)
    configure_logging()

    # Parser setup
    parser = argparse.ArgumentParser(description="Train the transformer as specified in the input JSON configuration.")
    parser.add_argument("config_path", type=str, help="Path to the JSON configuration file.")
    args = parser.parse_args()

    # Check if JSON configuration exists
    if not os.path.isfile(args.config_path):
        logging.error(f"The configuration file '{args.config_path}' does not exist.")
        exit(1)

    # Load the JSON configuration
    try:
        with open(args.config_path, "r") as file:
            config = json.load(file)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to parse the JSON file. {e}")
        exit(1)

    # Run main
    main(config)
