# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Logging script for a (single-process) run through the datasets


import os
import sys
import logging

from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import isa_data
import ops
import tokenizer_ops as tokops
import train_t5

def main(config_dict, split=isa_data.SPLITS["NONE"]):
    toks_dir, _, models_dir = ops.get_directory_paths(config_dict)
    tokenizer = tokops.load_latest_tokenizer(toks_dir)
    dataset = tokops.get_dataset(tokenizer, config_dict, split = split)
    model = train_t5.load_latest_model(models_dir)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=False, collate_fn=data_collator)
    total_samples = 0
    for batch_idx, batch in enumerate(dataloader):
        if batch_idx == 0:
            vals_shape = {k: v.shape for k, v in batch.items()}
            logging.info(f"The first batch info is:")
            logging.info(f"{vals_shape}")
        batch_size = batch["input_ids"].shape[0]
        total_samples += batch_size
        if batch_idx % 1000 == 0:
            logging.info(f"Processed {batch_idx} batches so far")
    logging.info(f"Total number of batches was {batch_idx + 1}")
    logging.info(f"Total number of samples was {total_samples}")

if __name__ == "__main__":
    ops.configure_logging("count_samples.log")
    try:
        config_dict = ops.get_config_dict(ops.parse_config_path(tool_explanation="Train the transformer as specified in the input JSON configuration."))
        ops.check_params(config_dict)
    except Exception as e:
        message = f"Loading configuration information: {e}"
        logging.error(message)
        raise Exception("Error " + message)
    
    main(config_dict)