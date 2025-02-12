# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for evaluating a Hugging Face T5 Model

import logging

import torch
from torch.utils.data import DataLoader
from datasets import IterableDataset
from accelerate import Accelerator
from transformers import DataCollatorForSeq2Seq
from accelerate.utils import broadcast_object_list

import ops
import train_t5
import accelerate_test
import tokenizer_ops as tokops

def prepare_mod_tok_data(accelerator, config_dict, batch_size=8, split=tokops.NONE):
    toks_dir, _, models_dir = ops.get_directory_paths(config_dict)
    if accelerator.is_main_process:
        tokenizer = tokops.load_latest_tokenizer(toks_dir)
        dataset = tokops.get_dataset(tokenizer, config_dict, split = split)
        model = train_t5.load_latest_model(models_dir)
    else:
        tokenizer = None
        dataset = None
        model = None
    
    accelerator.wait_for_everyone()
    tokenizer = broadcast_object_list([tokenizer])[0]
    dataset = broadcast_object_list([dataset])[0]
    model = broadcast_object_list([model])[0]

    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)

    dataloader, model = accelerator.prepare(dataloader, model)

    return model, tokenizer, dataloader


def compare_predictions(model, tokenizer, dataloader, max_length=512):
    model.eval()

    total_samples = 0
    exact_matches = 0
    mismatches = []

    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, labels=labels, max_length=max_length)
            
            original_input = tokenizer.batch_decode(input_ids, skip_special_tokens=True)
            original_outputs = tokenizer.batch_decode(labels, skip_special_tokens=True)
            predictions = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for input_text, predicted_text, target_text in zip(original_input, predictions, original_outputs):
                if predicted_text.strip() == target_text.strip():
                    exact_matches += 1
                else:
                    mismatches.append((input_text, predicted_text, target_text))
                
                total_samples += 1

    match_ratio = exact_matches / total_samples if total_samples > 0 else 0
    return exact_matches, total_samples, mismatches

def wrap_on_accel(f):
    try:
        accelerator = Accelerator(mixed_precision="fp16")
        if accelerator.is_main_process:
            logging.info(f"Accelerator started on {accelerator.num_processes} processes.")
        accelerate_test.log_cuda_info(accelerator)
        f(accelerator)
    except Exception as e:
        logging.error(f"{e}")
        raise e
    finally:
        accelerator.wait_for_everyone()
        if torch.distributed.is_initialized():
            try:
                torch.distributed.destroy_process_group()
            except Exception as e:
                logging.error(f"Error destroying process group: {str(e)}")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def on_data(config_dict, split=tokops.NONE):
    def predict_eval(accelerator, config_dict):
        model, tokenizer, dataloader = prepare_mod_tok_data(accelerator, config_dict, batch_size=8, split=split)
        compare_predictions(model, tokenizer, dataloader)
    
    wrap_on_accel(lambda acc: predict_eval(acc, config_dict))
