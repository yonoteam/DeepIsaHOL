# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for evaluating a Hugging Face T5 Model

import os
import logging

import torch
from torch.utils.data import DataLoader
from transformers import DataCollatorForSeq2Seq
from accelerate.utils import broadcast_object_list

import isa_data
import ops
import train_t5
import tokenizer_ops as tokops

def load_model_tok_data(accelerator, config_dict, split=isa_data.SPLITS["NONE"]):
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
    return model, tokenizer, dataset

def prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=8):
    batch_size = batch_size // accelerator.num_processes
    data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model, padding=True)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=data_collator)
    dataloader, model = accelerator.prepare(dataloader, model)
    return model, dataloader

def report(match_count, mismatches, total_samples, accelerator):
    if accelerator.num_processes > 1:
        match_count = accelerator.gather(torch.tensor([match_count], device=accelerator.device)).sum().item()
        total_samples = accelerator.gather(torch.tensor([total_samples], device=accelerator.device)).sum().item()
        all_mismatches = []
        if accelerator.is_main_process:
            all_mismatches = mismatches
    else:
        all_mismatches = mismatches
        
    accuracy = match_count / total_samples if total_samples > 0 else 0
    print(f"Accuracy = {accuracy}")
    if accelerator.is_main_process:
        save_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mismatches.txt")
        ops.save_tuple_list_as_txt(all_mismatches, save_file)
    accelerator.wait_for_everyone()

def execute(eval_metric, model, tokenizer, dataloader, accelerator, max_length=512):
    model.eval()
    match_count = 0
    total_samples = 0
    extending = True
    mismatches = []
    with torch.no_grad():
        for batch_idx, batch in enumerate(dataloader):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            batch_size = input_ids.shape[0]
            total_samples += batch_size
            outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=max_length)
            
            in_out_predicts = {
                "orig_input": tokenizer.batch_decode(input_ids, skip_special_tokens=True),
                "orig_output": tokenizer.batch_decode(labels, skip_special_tokens=True),
                "predictions": tokenizer.batch_decode(outputs, skip_special_tokens=True)
            }
            batch_count, batch_mismatches = eval_metric(in_out_predicts, extend = extending)
            match_count += batch_count
            extending = (len(mismatches) < 100)
            if extending:
                mismatches.extend(batch_mismatches)

    accelerator.wait_for_everyone()
    logging.info(f"Total number of batches: {batch_idx + 1}")
    logging.info(f"Last batch size: {batch_size}")
    return match_count, mismatches, total_samples

def exact_equality(in_out_predicts, extending=True):
    match_count = 0
    mismatches = []
    itps = zip(in_out_predicts["orig_input"], in_out_predicts["orig_output"], in_out_predicts["predictions"])
    for input, target, predict in itps:
        if predict.strip() == target.strip():
            match_count += 1
        elif extending and len(mismatches) < 2:
            mismatches.append((input, target, predict))
    return match_count, mismatches

def with_metric(eval_metric, config_dict, split=isa_data.SPLITS["NONE"]):
    def general_body(accelerator):
        model, tokenizer, dataset = load_model_tok_data(accelerator, config_dict, split=split)
        model, dataloader = prepare_model_and_dataloader(model, tokenizer, dataset, accelerator, batch_size=config_dict["batch_size"])
        match_count, mismatches, total_samples = execute(eval_metric, model, tokenizer, dataloader)
        report(match_count, mismatches, total_samples, accelerator)
    
    ops.wrap_w_accelerator(lambda acc: general_body(acc))
