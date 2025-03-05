# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Logging script for a (single-process) run through the datasets


import os
import sys
import logging


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import ops
import eval_t5

def main(config_dict):
    logging.info(f"Counting samples from the split: {config_dict['data_split']}")
    model, tokenizer, dataset = eval_t5.load_model_tok_data(config_dict)
    model, dataloader = eval_t5.prepare_model_and_dataloader(model, tokenizer, dataset, batch_size=config_dict["batch_size"])
    total_samples = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            vals_shape = {k: v.shape for k, v in batch.items()}
            logging.info(f"The first batch info is:")
            logging.info(f"{vals_shape}")
        batch_size = batch["input_ids"].shape[0]
        total_samples += batch_size
        if batch_idx % 1000 == 0 and batch_idx > 0:
            logging.info(f"Processed {batch_idx} batches so far")
    logging.info(f"Total number of batches was {batch_idx + 1}")
    logging.info(f"Total number of samples was {total_samples}")

if __name__ == "__main__":
    try:
        config_dict = ops.get_json_dict(ops.parse_config_path(tool_explanation="Tool to count (in a single process) the total number of samples in a dataset split."))
        ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)
    
    ops.configure_logging(f"count_samples_{config_dict['data_split']}.log")
    main(config_dict)