# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Test file for 

import os
import sys
import logging

import torch

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import ops
import tokenizer_ops as tokops
import train_t5
import eval_t5

def test_load_model_tok_data(accelerator, config_dict, split=tokops.NONE):
    try:
        model, tokenizer, train_data = eval_t5.load_model_tok_data(accelerator, config_dict, split=split)
        logging.info("eval_t5.test_load_model_tok_data passed")
        return model, tokenizer, train_data
    except Exception as e:
        message = f"eval_t5.test_load_model_tok_data failed: {e}"
        logging.error(message)
        raise Exception(message)

def test_prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=8):
    try:
        model, dataloader = eval_t5.prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=batch_size)
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
    for batch_idx, batch in enumerate(dataloader):
        train_t5.log_nan_inputs(batch_idx, batch, accelerator)
        train_t5.log_empty_labels(batch_idx, batch, accelerator)
    accelerator.wait_for_everyone()
    logging.info(f"{accelerator.process_index}: Total number of steps was {batch_idx + 1}")
    if accelerator.is_main_process:
        total = accelerator.gather(torch.tensor([batch_idx + 1], device=accelerator.device)).sum().item()
        logging.info(f"Total number of steps was {total}")

def main(accelerator, config_dict):
    model, tokenizer, dataset = test_load_model_tok_data(accelerator, config_dict, split=tokops.TRAIN)
    model, dataloader = test_prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=24)
    log_model_forward(model, dataloader, accelerator)
    log_dataset_info(dataloader, accelerator)
    

if __name__ == "__main__":
    ops.configure_logging("test_dataloading.log")
    try:
        config_dict = ops.get_config_dict(ops.parse_config_path(tool_explanation="Train the transformer as specified in the input JSON configuration."))
        ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)

    ops.wrap_w_accelerator(lambda acc: main(acc, config_dict))
