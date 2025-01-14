# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for training a Hugging Face T5 Model


import os
import json
import logging
import numpy as np
import torch
import proofs

from transformers import AutoTokenizer, AutoConfig
from transformers import T5ForConditionalGeneration
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq, get_scheduler
from accelerate import Accelerator

# PRELIMS

debug = True
ACTION_SEP = 'ACTION_SEP'
GOAL_SEP = 'GOAL_SEP'
TERM_SEP = 'TERM_SEP'
HYPS_SEP = 'HYPS_SEP'
VARS_SEP = 'VARS_SEP'
CONSTS_SEP = 'CONSTS_SEP'
TYPES_SEP = 'TYPES_SEP'
APPLY_KWS_SEP = 'APPLY_KWS_SEP'
ISAR_KWS_SEP = 'ISAR_KWS_SEP'
DEPS_SEP = 'DEPS_SEP'
NAME_SEP = 'NAME_SEP'
METHODS_SEP = 'METHODS_SEP'

def separator(sep, txt):
    return sep + ' ' + txt

# CONFIGURATION

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

def is_remote(all_models_dir, model_name, mode):
    tokenizers_dir, datasets_dir, models_dir = make_dir_vars(all_models_dir, model_name, mode)
    model_path = os.path.join(models_dir, "1", "model.safetensors")
    dataset_path = os.path.join(datasets_dir, "datasets.pt")
    tokenizer_path = os.path.join(tokenizers_dir, "1", "tokenizer.json")
    if os.path.isfile(model_path) and os.path.isfile(dataset_path) and os.path.isfile(tokenizer_path):
        return False
    return True

# TOKENIZER

def string_from(proof_json):
    str_list = []
    for step in proof_json['proof']['steps'][1:]:
        usr_act_str = " ".join([
            step['step']['user_state'], 
            separator(ACTION_SEP, step['step']['action']), 
            separator(GOAL_SEP, step['step']['term'])
        ])
        str_list.append(usr_act_str)
        
        str_list.append(HYPS_SEP)
        for hyp_dict in step['step'].get('hyps', []):
            for _, hyp in hyp_dict.items():
                str_list.append(separator(TERM_SEP, hyp))

        str_list.append(VARS_SEP)
        for var_dict in step['step'].get('variables', []):
            for _, var in var_dict.items():
                str_list.append(separator(TERM_SEP, var))

        str_list.append(VARS_SEP)
        for var_dict in step['step'].get('variables', []):
            for _, var in var_dict.items():
                str_list.append(separator(TERM_SEP, var))

        str_list.append(CONSTS_SEP)
        for const_dict in step['step'].get('constants', []):
            for _, const in const_dict.items():
                str_list.append(separator(TERM_SEP, const))

        str_list.append(TYPES_SEP)
        for type_var_dict in step['step'].get('type variables', []):
            for _, type_var in type_var_dict.items():
                str_list.append(separator(TERM_SEP, type_var))

    str_list.append(APPLY_KWS_SEP)
    for apply_kw in proof_json['proof'].get('apply_kwrds', []):
        str_list.append(separator(NAME_SEP, apply_kw['name']))

    str_list.append(ISAR_KWS_SEP)
    for isar_kw in proof_json['proof'].get('isar_kwrds', []):
        str_list.append(separator(NAME_SEP, isar_kw['name']))

    str_list.append(DEPS_SEP)
    for dep in proof_json['proof'].get('deps', []):
        str_list.append(separator(NAME_SEP, dep['thm']['name']))
        str_list.append(separator(TERM_SEP, dep['thm']['term']))
        
    str_list.append(METHODS_SEP)
    for method in proof_json['proof'].get('methods', []):
        str_list.append(separator(NAME_SEP, method['name']))

    return " ".join(str_list)

def load_hf_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return tokenizer
    except Exception:
        logging.error(f"'{model_name}' is not a valid local directory or Hugging Face model.")

def train_tokenizer(model_name, data_dir):
    try:
        tokenizer = load_hf_tokenizer(model_name)
        def get_training_corpus():
            imm_subdirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir()]
            for path in imm_subdirs:
                if debug:
                    logging.info(f"Processing directory: {path}")
                for subdir, _, files in os.walk(path):
                    json_files = [file for file in files if file.startswith("proof") and file.endswith(".json")]
                    for file in json_files:
                        json_path = os.path.join(subdir, file)
                        proof_json = proofs.get_proof_json(json_path)
                        yield string_from(proof_json)
        
        training_corpus = get_training_corpus()
        tokenizer = tokenizer.train_new_from_iterator(training_corpus, 52000) # TODO: calculate optimal vocabulary size
        return tokenizer
    except Exception as e:
        logging.error(f"training tokenizer for {model_name} using data in {data_dir}: {e}")

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
    os.makedirs(new_dir, exist_ok=False)
    hf_data.save_pretrained(new_dir)
    
def get_trained_tokenizer(remote, data_dir, tokenizers_dir, model_name):
    if remote:
        tokenizer = train_tokenizer(model_name, data_dir)
        save_hf_data_in(tokenizer, tokenizers_dir)
        return tokenizer
    else:
        subdirs = [
                os.path.join(tokenizers_dir, d) for d in os.listdir(tokenizers_dir)
                if os.path.isdir(os.path.join(tokenizers_dir, d)) and d.isdigit()
            ]
        latest_dir = max(subdirs, key=lambda d: int(os.path.basename(d)))
        tokenizer = AutoTokenizer.from_pretrained(latest_dir)
        return tokenizer


# DATA
# TODO: Create HuggingFace class

def add_data_from(mode_tok_data, proof_json):
    mode, tokenizer, data = mode_tok_data
    for step in proof_json['proof']['steps'][1:]:
        y = step['step']['action']
        xs = step['step']['user_state']
        if mode == 'state_prems':
            xs = xs + ' ' + separator(GOAL_SEP, proofs.orig_objective_of(proof_json))
            xs = xs + ' ' + separator(DEPS_SEP, ' ')
            for thm in proof_json['proof']['deps']:
                zs = separator(NAME_SEP, thm['thm']['name']) + ' ' + separator(TERM_SEP, thm['thm']['term'])
                xs = xs + ' ' + zs
        elif mode == 'state_prems_consts':
            xs = xs + ' ' + separator(GOAL_SEP, proofs.orig_objective_of(proof_json))
            xs = xs + ' ' + separator(DEPS_SEP, ' ')
            for thm in proof_json['proof']['deps']:
                zs = separator(NAME_SEP, thm['thm']['name']) + ' ' + separator(TERM_SEP, thm['thm']['term'])
                xs = xs + ' ' + zs
            xs = xs + ' ' + separator(CONSTS_SEP, ' ')
            for const in step['step']['constants']:
                for key in const.keys():
                    zs = separator(TERM_SEP, const[key])
                    xs = xs + ' ' + zs

        inputs = tokenizer(
            xs, 
            max_length=512, 
            truncation=True, 
            return_overflowing_tokens=True,            
            stride=10,
            padding="max_length", # TODO: delete? so that data_collator handles the padding efficiently?
            return_tensors="pt"
        )
        targets = tokenizer(
            y, 
            max_length=512, 
            truncation=True, 
            padding="max_length", 
            return_tensors="pt"
        )
        overflow_mapping = inputs["overflow_to_sample_mapping"]
        for i, input_ids in enumerate(inputs["input_ids"]):
            to_add = {
                "input_ids": input_ids,  # No need for squeeze here
                "attention_mask": inputs["attention_mask"][i],
                "overflow_sample_idx": overflow_mapping[i], # tells origin sample of this to_add
                "labels": targets["input_ids"].squeeze()
            }
            data.append(to_add)
    return data

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
                train_data = add_data_from((mode, tokenizer, train_data), proof_json)
            elif i < train_size + valid_size:
                valid_data = add_data_from((mode, tokenizer, valid_data), proof_json)
            else:
                test_data = add_data_from((mode, tokenizer, test_data), proof_json)
    return mode, tokenizer, train_data, valid_data, test_data

# TODO: adapt for proofs.gen_apply
def make_datasets(mode, tokenizer, init_train, init_valid, init_test, data_dir):
    all_results = (mode, tokenizer, init_train, init_valid, init_test)
    imm_subdirs = [entry.path for entry in os.scandir(data_dir) if entry.is_dir()]
    for path in imm_subdirs:
        if debug:
            logging.info(f"Processing directory: {path}")
        all_results = add_dir_data(all_results, path)
    _, _, train_data, valid_data, test_data = all_results
    return train_data, valid_data, test_data

# TODO: add support for HF datasets library
def save_datasets_in(train_data, valid_data, test_data, datasets_dir):
    dataset_dict = {
        "train": train_data,
        "valid": valid_data,
        "test": test_data
        }
    datasets_path = os.path.join(datasets_dir, 'datasets.pt')
    torch.save(dataset_dict, datasets_path)

def load_datasets(datasets_dir):
    datasets_path = os.path.join(datasets_dir, 'datasets.pt')
    dataset_dict = torch.load(datasets_path, weights_only=True)
    train_data = dataset_dict["train"]
    valid_data = dataset_dict["valid"]
    test_data = dataset_dict["test"]
    return train_data, valid_data, test_data

def get_datasets(remote, mode, tokenizer, data_dir, datasets_dir):
    if remote:
        train_data, valid_data, test_data = make_datasets(mode, tokenizer, [], [], [], data_dir)
        save_datasets_in(train_data, valid_data, test_data, datasets_dir)
        return train_data, valid_data, test_data
    else:
        train_data, valid_data, test_data = load_datasets(datasets_dir)
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

# TODO: add support for distributed training (e.g. torch.nn.DataParallel or Hugging Face's Accelerate library)
# TODO: make batch_size, lr, and vocab_size configurable from configuration JSON
def train(model, train_dataloader, valid_dataloader, num_epochs, device, models_dir):

    # Optimizer, accelerator, dataloader, and Scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    accelerator = Accelerator()
    train_dataloader, valid_dataloader, model, optimizer = accelerator.prepare(
        train_dataloader, valid_dataloader, model, optimizer
    )
    num_training_steps = len(train_dataloader) * num_epochs
    lr_scheduler = get_scheduler(
        "linear", optimizer=optimizer, num_warmup_steps=0, num_training_steps=num_training_steps
    )

    model.train()
    for epoch in range(num_epochs):
        logging.info(f"Epoch {epoch + 1} of {num_epochs}")
        train_loss = 0.0
        
        for batch_idx, batch in enumerate(train_dataloader):
            # Move data to device
            input_ids = batch["input_ids"] # .to(device)
            attention_mask = batch["attention_mask"] # .to(device)
            labels = batch["labels"] # .to(device)# torch.tensor(np.array(batch["labels"]), dtype=torch.int64).to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()
            accelerator.backward(loss) # loss.backward()
            optimizer.step()
            lr_scheduler.step()

            train_loss += loss.item()

            # progress feedback
            if batch_idx % 100 == 0:
                logging.info(f"Train step number {batch_idx} of {len(train_dataloader)}")

        avg_train_loss = train_loss / len(train_dataloader)
        logging.info(f"Average Training Loss: {avg_train_loss:.4f}")
        
        # Validation Loop
        model.eval()
        valid_loss = 0.0
        with torch.no_grad():
            for batch in valid_dataloader:
                input_ids = batch["input_ids"] # .to(device)
                attention_mask = batch["attention_mask"] # .to(device)
                labels = batch["labels"] # .to(device)

                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                valid_loss += accelerator.gather(loss).mean().item()

        avg_valid_loss = valid_loss / len(valid_dataloader)
        logging.info(f"Validation Loss: {avg_valid_loss:.4f}")
    
    save_hf_data_in(model, models_dir)


# ALGORITHM

def main(config):
    # Setup from config
    try:
        data_dir, all_models_dir, model_name, mode, num_epochs = extract_params(config)
        test_params(data_dir, all_models_dir)
    except Exception as e:
        logging.error(f"Could not setup from configuration file: '{e}'.")
        exit(1)
    
    remote = is_remote(all_models_dir, model_name, mode)
    tokenizers_dir, datasets_dir, models_dir = make_dir_vars(all_models_dir, model_name, mode)
    os.makedirs(datasets_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)

    # Tokenizer
    tokenizer = get_trained_tokenizer(remote, data_dir, tokenizers_dir, model_name)
    vocab_size = len(tokenizer)

    # Data
    train_data, valid_data, test_data = get_datasets(remote, mode, tokenizer, data_dir, datasets_dir)

    # Model
    model = get_init_model(remote, vocab_size, models_dir, model_name)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # not needed with HF accelerate
    model = model.to(device) # not needed with HF accelerate

    # Data Collator and Loaders
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)

    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=True, collate_fn=data_collator)
    valid_dataloader = DataLoader(valid_data, batch_size=8, shuffle=False, collate_fn=data_collator)

    # Training loop
    train(model, train_dataloader, valid_dataloader, num_epochs, device, models_dir)


if __name__ == "__main__":
    configure_logging()

    # Parser setup
    import argparse
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
