# Mantainers: 
#   Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Test file for 

import sys
import os

from torch.utils.data import DataLoader
from datasets import IterableDataset
from transformers import (
    DataCollatorForSeq2Seq)

TEST_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(os.path.dirname(TEST_DIR))
MAIN_DIR = os.path.join(SRC_DIR, 'main/python')
sys.path.insert(0, MAIN_DIR)

import proofs
import directories as dirs
import tokenizer_ops as tokops
import train_t5

def main():
    tokenizer = tokops.load_latest_tokenizer(dirs.tokenizers_dir)
    train_data = IterableDataset.from_generator(tokops.generate_model_inputs, gen_kwargs={'tokenizer': tokenizer, 'json_data_dir': dirs.test_data_dir, 'split': tokops.TRAIN, 'mode': proofs.SPKT_MODE})
    model = train_t5.load_latest_model(dirs.models_dir)
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    train_dataloader = DataLoader(train_data, batch_size=8, shuffle=False, collate_fn=data_collator)
    for batch in train_dataloader:
        break
    print("The batch shape is:")
    print({k: v.shape for k, v in batch.items()})
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch["attention_mask"],
        labels=batch["labels"]
    )
    print("Successfully used model.forward() on first batch:")
    print(f"The ouputs items types are: ")
    print({k: type(v) for k, v in outputs.items()})
    print(f"The loss is {outputs.loss} and the logits are {outputs.logits.shape}")

if __name__ == "__main__":
    main()