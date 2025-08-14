# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for finetuning an Unsloth Gemma Model
# Following the Unsloth documentation at https://docs.unsloth.ai/basics/gemma-3n-how-to-run-and-fine-tune

import os
import re
import logging
import statistics

from typing import Optional

from unsloth import FastModel
from unsloth.chat_templates import (
    get_chat_template, 
    standardize_data_formats, 
    train_on_responses_only
)

from transformers import AutoTokenizer
from datasets import IterableDataset
from trl import SFTTrainer, SFTConfig

# os.environ['TORCHDYNAMO_CACHE_SIZE_LIMIT'] = '999999999'
import torch
torch._dynamo.config.cache_size_limit = 64

import dicts
import distrib
import config_ops
import tokenizer_ops as tokops

def configure_trainer_args(config_dict):
    pre_args = config_dict["hf_train_args"]
    batches_per_epoch = config_dict["batches_per_epoch"]
    # torch_dtype = config_ops.get_torch_float_type(config_dict["float_type"])

    # TrainingArguments
    # pre_args["fp16"] = True if torch_dtype == torch.float16 else False
    # pre_args["bf16"] = True if torch_dtype == torch.bfloat16 else False
    pre_args["max_steps"] = config_dict["num_epochs"] * batches_per_epoch
    pre_args["logging_dir"] = os.getcwd()
    pre_args["logging_steps"] = max(1, batches_per_epoch // 100)
    pre_args["output_dir"] = config_dict["models_dir"]
    pre_args["overwrite_output_dir"] = True
    pre_args["save_strategy"] = "steps"
    pre_args["save_total_limit"] = 5
    pre_args["save_steps"] = batches_per_epoch

    # Overwriting
    # pre_args["learning_rate"] = 2e-4 # based on QLoRA paper
    pre_args["max_grad_norm"] = 0.3 # based on QLoRA paper
    pre_args["warmup_ratio"] = 0.03 # based on QLoRA paper

    # patching
    pre_args["dataloader_num_workers"] = 0
    pre_args["dataloader_prefetch_factor"] = None

    # SFTConfig
    pre_args["dataset_text_field"] = "text"
    pre_args["packing"] = False
    pre_args["optim"] = "adamw_8bit"

    sft_args = SFTConfig(**pre_args)
    return sft_args

def gemma_generator(config_dict, tokenizer):
    data_dir = config_dict["data_dir"]
    split = config_dict["data_split"]
    data_format = config_dict["data_format"]
    max_length = tokops.get_gemma_context_length(data_format)
    for conversation in tokops.generate_gemma_inputs(data_dir, split, data_format):
        x = conversation["messages"][0]["content"][0]["text"]
        y = conversation["messages"][1]["content"][0]["text"]
        tok_x = tokenizer(x, return_tensors="pt")
        tok_y = tokenizer(y, return_tensors="pt")
        total_size = tok_x["input_ids"].shape[1] + tok_y["input_ids"].shape[1]
        if total_size > max_length:
            # message = f"""
            # Skipping pair due to exceeding context length:
            #     x: {x[:1000]}
            #     y: {y}.
            # """
            # logging.warning(message)
            continue
        yield conversation

def load_and_format_dataset(preproc, config_dict):
    tokenizer = preproc.tokenizer if hasattr(preproc, "tokenizer") else preproc
    dataset = IterableDataset.from_generator(
        gemma_generator,
        gen_kwargs=dict(
            config_dict = config_dict,
            tokenizer = tokenizer
        )
    )
    dataset = standardize_data_formats(dataset)

    def format_prompts(examples):
        convos = examples["messages"]
        texts = [preproc.apply_chat_template(
            convo, 
            tokenize = False,
            add_generation_prompt = False
            ).removeprefix('<bos>') for convo in convos]
        return { "text" : texts, }

    dataset = dataset.map(format_prompts, batched = True)
    return dataset

def load_training_objs(accelerator, config_dict):
    model, preprocessor = FastModel.from_pretrained(
        model_name = config_dict["model_name"],
        dtype = None, # None for Unsloth auto detection
        max_seq_length = tokops.get_gemma_context_length(config_dict["data_format"]),
        load_in_4bit = True,  # 4 bit quantization to reduce memory
        full_finetuning = False,
    )
    logging.info(f"Checkpoint: loaded fast model.")

    preprocessor = get_chat_template(
        preprocessor,
        chat_template = "gemma-3",
    )

    dataset = load_and_format_dataset(preprocessor, config_dict)
    logging.info(f"Checkpoint: loaded preprocessor and dataset.")

    model = FastModel.get_peft_model(
        model,
        finetune_vision_layers     = False,
        finetune_language_layers   = True,  # Should leave on!
        finetune_attention_modules = True,  # Attention good for GRPO
        finetune_mlp_modules       = True,  # SHould leave on always!

        r = 8,           # Larger = higher accuracy, but might overfit
        lora_alpha = 8,  # Recommended alpha == r at least
        lora_dropout = 0,
        bias = "none",
        random_state = 3407,
    )
    logging.info(f"Checkpoint: loaded peft model.")

    train_args = configure_trainer_args(config_dict)
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=dataset,
        eval_dataset = None,
        processing_class=preprocessor
    )
    trainer = train_on_responses_only(
        trainer,
        instruction_part = "<start_of_turn>user\n",
        response_part = "<start_of_turn>model\n",
    )
    logging.info("Checkpoint: Trainer initialised.")
    first_example = next(iter(trainer.train_dataset))
    expected_first_answer = preprocessor.decode(
        [preprocessor.pad_token_id if x == -100 else x for x in first_example["labels"]]
        ).replace(preprocessor.pad_token, "")
    logging.info(f"The evidence is: {expected_first_answer}")

    result = dict(
        model=model,
        preprocessor=preprocessor,
        dataset=dataset,
        train_args=train_args,
        trainer=trainer
    )
    return result

def main(accelerator, config_dict):
    model_tok_data_dict = load_training_objs(accelerator, config_dict)
    trainer = model_tok_data_dict["trainer"]

    logging.info(f"Starting training loop.")
    trainer.train()

    logging.info(f"Training completed.")
    logging.info(f"Saving progress.")
    trainer.save_model()

    logging.info(f"Saved progress. Bye!")

def load_tuned_objs(config_dict):
    model, preprocessor = FastModel.from_pretrained(
        model_name = config_dict["model_name"],
        max_seq_length = tokops.get_gemma_context_length(config_dict["data_format"]),
        load_in_4bit = True,
    )
    model.load_adapter(config_dict["models_dir"])
    # model = AutoModelForCausalLM.from_pretrained(config_dict["models_dir"])
    preprocessor = get_chat_template(
        AutoTokenizer.from_pretrained(config_dict["model_name"]),
        chat_template = "gemma-3",
    )
    return {
        "model": model,
        "preprocessor": preprocessor
    }

def extract_suggestion(text: str) -> Optional[str]:
    pattern = r"<SUGGESTION>(.*?)</SUGGESTION>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    return None

def suggest(tokenizer, model, conversation):
    inputs = tokenizer.apply_chat_template(
        conversation["messages"],
        add_generation_prompt = True,
        return_tensors = "pt",
        tokenize = True,
        return_dict = True,
    ).to("cuda")
    toked_outputs = model.generate(
        **inputs,
        max_new_tokens = 64,
        temperature = 1.0, top_p = 0.95, top_k = 64,
    )
    outputs = tokenizer.batch_decode(toked_outputs)
    return extract_suggestion(outputs[0])


def compute_stats(config_dict):
    tokenizer = AutoTokenizer.from_pretrained(config_dict["model_name"])
    data_dir = config_dict["data_dir"]
    split = config_dict["data_split"]
    data_format = config_dict["data_format"]
    max_steps = config_dict.get("batches_per_epoch", None)

    x_lengths = []
    y_lengths = []
    thresholds = [512, 1024, 2048, 4096, 8192]
    length_counters = {th: 0 for th in thresholds}
    gemma_iter = tokops.generate_gemma_inputs(data_dir, split, data_format)

    for i, conversation in enumerate(gemma_iter):
        if max_steps is not None and i >= max_steps:
            logging.info(f"Reached max steps of {max_steps}. Stopping counting.")
            break
        x = conversation["messages"][0]["content"][0]["text"]
        y = conversation["messages"][1]["content"][0]["text"]
        x_tok_len = len(tokenizer(x)["input_ids"])
        y_tok_len = len(tokenizer(y)["input_ids"])
        x_lengths.append(x_tok_len)
        y_lengths.append(y_tok_len)
        length_sum = x_tok_len + y_tok_len

        for threshold in thresholds:
            if length_sum >= threshold:
                length_counters[threshold] += 1
        if i % 1000 == 0 and i > 0:
            logging.info(f"Processed {i} examples so far")
    
    def get_stats(nums):
        total_tok_sum = sum(nums)
        return {
            "total_tok_sum": total_tok_sum,
            "avg": total_tok_sum / len(nums) if nums else 0,
            "max": max(nums, default=0),
            "min": min(nums, default=0),
            "median": statistics.median(nums) if nums else 0,
            "mode": statistics.mode(nums) if nums else 0
        }
    
    x_stats = get_stats(x_lengths)
    y_stats = get_stats(y_lengths)
    total_datapoints = len(x_lengths)
    result = {
        "x_total_toks": x_stats["total_tok_sum"],
        "y_total_toks": y_stats["total_tok_sum"],
        "x_avg": x_stats["avg"],
        "y_avg": y_stats["avg"],
        "x_max": x_stats["max"],
        "y_max": y_stats["max"],
        "x_min": x_stats["min"],
        "y_min": y_stats["min"],
        "x_median": x_stats["median"],
        "y_median": y_stats["median"],
        "x_mode": x_stats["mode"],
        "y_mode": y_stats["mode"],
        **{f"geq_{t}": length_counters[t] / total_datapoints if total_datapoints > 0 else 0 for t in thresholds},
        "total_datapoints": total_datapoints
    }
    final_message = f"""
    The data statistics for the split '{split}' with the format '{data_format}' are: 
    {dicts.to_string(result)}
    """
    logging.info(final_message)
    return conversation, result

if __name__ == "__main__":
    explanation = "Train Gemma as specified in the input JSON configuration."
    config_dict = config_ops.parse_path(explanation)

    task = config_dict["task"]
    if task == config_ops.count_dataset:
        config_ops.setup_logging("gemma_data_stats.log")
        logging.info("Starting counting process.")
        compute_stats(config_dict)
    else:
        config_ops.setup_logging("gemma_train.log")
        logging.info("Starting single finetuning Gemma process.")
        distrib.log_cuda_info_via_torch()
        main(0, config_dict)