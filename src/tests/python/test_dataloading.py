# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Logging script for a (multi-process) run through the datasets
# that also tests that the model, tokenizer, and datasets are
# loaded correctly

import os
import sys
import logging

import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import dicts
import distrib
import eval_t5
import train_t5
import config_ops
from proofs.str_ops import FORMATS

def test_load_model_tok_data(config_dict, accelerator):
    try:
        model, tokenizer, train_data = eval_t5.load_model_tok_data(config_dict, accelerator=accelerator)
        logging.info("eval_t5.test_load_model_tok_data passed")
        return model, tokenizer, train_data
    except Exception as e:
        message = f"eval_t5.test_load_model_tok_data failed: {e}"
        logging.error(message)
        raise Exception(message)

def test_prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=8):
    try:
        model, dataloader = eval_t5.prepare_model_and_dataloader(model, tokenizer, dataset, batch_size=batch_size, accelerator=accelerator)
        logging.info("eval_t5.test_prepare_model_and_dataloader passed")
        return model, dataloader
    except Exception as e:
        message = f"eval_t5.test_prepare_model_and_dataloader failed: {e}"
        logging.error(message)
        raise Exception(message)

def log_model_forward(model, dataloader, accelerator):
    for batch in dataloader:
        break
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    pi = accelerator.process_index
    logging.info(f"{pi}: Successfully used model.forward() on first batch:")
    logging.info(f"{pi}: The ouputs items types are: ")
    outputs_shape = {k: type(v) for k, v in outputs.items()}
    logging.info(f"{pi}: {outputs_shape}")
    logging.info(f"{pi}: The loss is {outputs.loss} and the logits are {outputs.logits.shape}")

def log_dataset_info(dataloader, accelerator):
    train_t5.log_dataloading(dataloader, accelerator)
    total_samples = 0
    for batch_idx, batch in enumerate(dataloader):
        train_t5.log_nan_inputs(batch_idx, batch, accelerator)
        train_t5.log_empty_labels(batch_idx, batch, accelerator)
        batch_size = batch["input_ids"].shape[0]
        total_samples += batch_size
        if batch_idx % 1000 == 0:
            logging.info(f"Train step number {batch_idx}")
    accelerator.wait_for_everyone()
    logging.info(f"{accelerator.process_index}: Total number of batches was {batch_idx + 1}")
    if accelerator.is_main_process:
        batches = accelerator.gather(torch.tensor([batch_idx + 1], device=accelerator.device)).sum().item()
        samples = accelerator.gather(torch.tensor([total_samples], device=accelerator.device)).sum().item()
        logging.info(f"Total number of batches was {batches}")
        logging.info(f"Total number of samples was {samples}")

def main(accelerator, config_dict):
    model, tokenizer, dataset = test_load_model_tok_data(config_dict, accelerator)
    model, dataloader = test_prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=config_dict["batch_size"])
    log_model_forward(model, dataloader, accelerator)
    log_dataset_info(dataloader, accelerator)
    

if __name__ == "__main__":
    config_ops.setup_logging("test_dataloading.log")
    try:
        explanation = "tests that the model, tokenizer, and datasets are loaded correctly."
        path = config_ops.parse_config_path(tool_explanation=explanation)
        config_dict = dicts.load_json(path)
        config_ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)

    distrib.wrap_w_accelerator(lambda acc: main(acc, config_dict))
