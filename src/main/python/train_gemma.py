# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Utility for finetuning a Hugging Face Gemma Model
# Documentation at https://huggingface.co/docs/transformers/en/model_doc/gemma#gemma
# and https://ai.google.dev/gemma/docs/core/huggingface_text_finetune_qlora

import os
import logging

import torch

from peft import LoraConfig
from trl import (
    SFTConfig, 
    SFTTrainer
)

from datasets import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    BitsAndBytesConfig
)

from accelerate.utils import broadcast_object_list

import dicts
import proofs
import distrib
import config_ops
import tokenizer_ops as tokops

gemma_prompt = """Recommend the next Isabelle proof step given the context below:
{context}
"""

def to_gemma_format(input_text, target_text):
    return {
        "messages": [
            {"role": "user", "content": gemma_prompt.format(context=input_text)},
            {"role": "assistant", "content": f"<SUGGESTION>{target_text}</SUGGESTION>"}
        ]
    }

def generate_gemma_inputs(json_data_dir, split, data_format):
    for path in proofs.data_dir.generate_dataset_paths(json_data_dir, split):
        proof = dicts.load_json(path)
        for input_text, target_text in proofs.str_ops.inputs_targets_from(proof, data_format):
            yield to_gemma_format(input_text, target_text)

def get_torch_float_type(float_type_str):
    torch_type_mapping = {
        "float32": torch.float32,
        "float": torch.float32,
        "fp32": torch.float32,
        "float64": torch.float64,
        "double": torch.float64,
        "fp64": torch.float64,  
        "float16": torch.float16,
        "half": torch.float16,  
        "fp16": torch.float16,  
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16, 
        "tf32": torch.float32
    }
    float_type_str_lower = float_type_str.lower()
    if float_type_str_lower in torch_type_mapping:
        return torch_type_mapping[float_type_str_lower]
    else:
        raise ValueError(
            f"Unknown float type '{float_type_str}'. "
            f"Expected one of: {list(torch_type_mapping.keys())}"
        )

def init_gemma(model_name, torch_dtype):
    model_kwargs = dict(
        attn_implementation="eager",
        torch_dtype=torch_dtype,
        device_map="auto",
        # device_map={'': torch.cuda.current_device()}
    )
    model_kwargs["quantization_config"] = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=model_kwargs['torch_dtype'],
        bnb_4bit_quant_storage=model_kwargs['torch_dtype'],
    )
    model = AutoModelForCausalLM.from_pretrained(model_name, **model_kwargs)
    return model

def configure_lora():
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.05,
        r=16,
        bias="none",
        target_modules="all-linear",
        task_type="CAUSAL_LM",
        modules_to_save=["lm_head", "embed_tokens"]
    )
    return peft_config

def configure_trainer(config_dict):
    pre_args = config_dict["hf_train_args"]
    batches_per_epoch = config_dict["batches_per_epoch"]
    torch_dtype = get_torch_float_type(config_dict["float_type"])

    # TrainingArguments
    pre_args["max_steps"] = config_dict["num_epochs"] * batches_per_epoch
    pre_args["logging_dir"] = os.getcwd()
    pre_args["logging_steps"] = max(1, batches_per_epoch // 100)
    pre_args["output_dir"] = config_dict["models_dir"]
    pre_args["overwrite_output_dir"] = True
    pre_args["save_strategy"] = "steps"
    pre_args["save_total_limit"] = 5
    pre_args["save_steps"] = batches_per_epoch
    pre_args["fp16"] = True if torch_dtype == torch.float16 else False
    pre_args["bf16"] = True if torch_dtype == torch.bfloat16 else False

    # Overwriting
    pre_args["learning_rate"] = 2e-4 # based on QLoRA paper
    pre_args["max_grad_norm"] = 0.3 # based on QLoRA paper
    pre_args["warmup_ratio"] = 0.03 # based on QLoRA paper

    # SFTConfig
    pre_args["packing"] = False
    pre_args["optim"] = "adamw_torch_fused"

    sft_args = SFTConfig(**pre_args)
    return sft_args

def load_tok_data(config_dict, split):
    logging.info(f"Loading tokenizer and data.")
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-1b-it")
    dataset = IterableDataset.from_generator(
        generate_gemma_inputs,
        gen_kwargs=dict(
            json_data_dir = config_dict["data_dir"],
            split = split,
            data_format = config_dict["data_format"]
        )
    )
    logging.info(f"Loaded tokenizer and data.")
    return tokenizer, dataset

def load_model_tok_data_trainer(accelerator, config_dict):
    ftype_str = config_dict["float_type"]
    model_name = config_dict["model_name"]
    torch_dtype = get_torch_float_type(ftype_str)

    tokenizer, train_data = load_tok_data(config_dict, "train")
    logging.info(f"Loading model.")

    if accelerator is not None:
        if accelerator.is_main_process:
            model = init_gemma(model_name, torch_dtype)
        else:
            model = None
        accelerator.wait_for_everyone()
        model = broadcast_object_list([model])[0]
        logging.info(f"{accelerator.process_index}: Successfully broadcasted data, the evidence is that the type of model is {type(model)}")
    else:
        model = init_gemma(model_name, torch_dtype)

    logging.info(f"Loaded model.")
    logging.info(f"Configuring trainer.")
    train_args = configure_trainer(config_dict)
    peft_config = configure_lora()

    logging.info(f"Before Traner initialization let's see device map")
    for val in model.hf_device_map.values():
        logging.info(f"The value device is {val}")
    trainer = SFTTrainer(
        model=model,
        args=train_args,
        train_dataset=train_data,
        peft_config=peft_config,
        processing_class=tokenizer
    )
    result = dict(
        model=model,
        tokenizer=tokenizer,
        train_data=train_data,
        train_args=train_args,
        trainer=trainer
    )
    return result

def count_train_samples(config_dict):
    data_split = config_dict["data_split"]
    tokenizer, train_data = load_tok_data(config_dict, data_split)
    
    def tokenize(example, tokenizer):
        processed = tokenizer.apply_chat_template(
            example["messages"],
            return_dict=True,
            return_assistant_tokens_mask=False,
            tools=example.get("tools"),
            **example.get("chat_template_kwargs", {}),
        )
        return processed

    processed_dataset = train_data.map(
            tokenize,
            fn_kwargs={
                "tokenizer": tokenizer
            }
        )
    
    total_samples = 0
    for example_idx, example in enumerate(processed_dataset):
        if example_idx == 0:
            logging.info(f"The first example properties are:")
            example_shape = {k: type(v) for k, v in example.items()}
            logging.info(f"{example_shape}")
            example_str = dicts.to_string(example, max_chars=300)
            logging.info(f"{example_str}")
        example_size = len(example["input_ids"])
        total_samples += example_size
        if example_idx % 1000 == 0 and example_idx > 0:
            logging.info(f"Processed {example_idx} examples so far")
    logging.info(f"Total number of examples was {example_idx + 1}")
    logging.info(f"Total number of tokens was {total_samples}")
    return example

def main(accelerator, config_dict):
    # distrib.log_cuda_info_via_torch()
    result = load_model_tok_data_trainer(accelerator, config_dict)
    trainer = result["trainer"]

    logging.info(f"Configured trainer.")
    logging.info(f"Starting training.")
    trainer.train()

    logging.info(f"Training completed.")
    logging.info(f"Saving progress.")
    trainer.save()

    logging.info(f"Saved progress. Bye!")

if __name__ == "__main__":
    explanation = "Train Gemma as specified in the input JSON configuration."
    config_dict = config_ops.parse_path(explanation)
    config_ops.setup_logging("gemma_train.log")

    task = config_dict["task"]
    if task == config_ops.count_dataset:
        count_train_samples(config_dict)
    elif task == config_ops.finetune_model:
        if torch.cuda.is_available() and torch.cuda.device_count() > 1:
            distrib.wrap_w_accelerator(lambda acc: main(acc, config_dict))
        else:
            main(0, config_dict)
    else:
        raise ValueError(
            f"Undefined task '{task}' for training script."
            f"Expected one of: {list(config_ops.count_dataset, config_ops.finetune_model)}"
        )