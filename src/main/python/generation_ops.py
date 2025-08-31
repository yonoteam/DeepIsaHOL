

import re
import logging
from typing import Optional

import torch
import eval_t5

import proofs
import tokenizer_ops as tokops

def using_unsloth():
    return torch.cuda.is_available()

def get_model_type(config_dict):
    model_name = config_dict["model_name"]
    lower_case_model = model_name.lower()
    if "t5" in lower_case_model:
        model_type = "t5"
    elif "gemma" in lower_case_model:
        model_type = "gemma"
    else:
        raise ValueError(f"Unsupported model: {model_name}. Supported: T5, Gemma.")
    logging.info(f"Model type set to {model_type} based on model name {model_name}.")
    return model_type

def load_ollama_tuned_objs(config_dict):
    from llama_cpp import Llama, LlamaTokenizer

    model = Llama(
        model_path=config_dict["models_dir"],
        n_ctx=tokops.get_gemma_context_length(config_dict["data_format"]),
        n_gpu_layers=-1 # all on GPU if possible
        # 0  # CPU for all layers
    )
    preprocessor = LlamaTokenizer(model)
    return {
        "model": model,
        "preprocessor": preprocessor
    }

def load_tok_model(config_dict):
    model_type = get_model_type(config_dict)
    data_format = config_dict["data_format"]

    if model_type == "t5":
        model, tokenizer, _ = eval_t5.load_model_tok_data(config_dict)
        tok_max_length = tokops.get_t5_context_length(data_format)
        tokenizer.model_max_length = tok_max_length
        print(f"Model context length = {model.config.n_positions}")
        print(f"Tokenizer context length = {tokenizer.model_max_length}")
    elif model_type == "gemma":
        if using_unsloth():
            import train_unsloth_gemma
            tuned_objs = train_unsloth_gemma.load_tuned_objs(config_dict)
        else:
            tuned_objs = load_ollama_tuned_objs(config_dict)
        model = tuned_objs["model"]
        tokenizer = tuned_objs["preprocessor"]
        print(f"Model type = {type(model)}")
        print(f"Tokeinzer type = {type(tokenizer)}")
    return tokenizer, model

def extract_gemma_suggestion(text: str) -> Optional[str]:
    pattern = r"<SUGGESTION>(.*?)</SUGGESTION>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        return match.group(1)
    logging.warning(f"No suggestion found in generated text: {text}")
    return None

def generate_predicts(prf_info: dict, generation_config: dict) -> tuple[str, list[Optional[str]]]:
    data_format = generation_config["data_format"]
    model_type = generation_config["model_type"]
    gen_length = generation_config.get("gen_length", 4096)
    num_return_sequences = generation_config.get("num_return_sequences", 1)
    num_beams = generation_config.get("num_beams", 1)
    using_unsloth = generation_config.get("use_unsloth", False)

    usr_sep = proofs.str_ops.Separator["user_state"]
    xs = [
        prf_info.get("proof_so_far", ""), 
        usr_sep, 
        prf_info.get("last_usr_state", "")
    ]
    xs.extend(prf_info.get("proof_data", ""))

    x = " ".join(xs)

    if model_type == "t5":
        x = "isabelle next step: " + x if "finetune" in data_format else x  
        predicts = generation_config["generator"](
            x, 
            max_length=gen_length, 
            num_return_sequences=num_return_sequences,
            num_beams=num_beams
        )
        predicts = [p["generated_text"] for p in predicts]
    elif model_type == "gemma":
        if using_unsloth:
            conversation = tokops.to_gemma_format(x, "")
            generation_messages = [conversation["messages"][0]]
            
            predicts = generation_config["generator"](
                generation_messages, 
                max_new_tokens=gen_length, 
                num_return_sequences=num_return_sequences,
                num_beams=num_beams,
                temperature = 1.0,
                top_p = 0.95,
                top_k = 64
            )
            predicts = [extract_gemma_suggestion(p["generated_text"][1]["content"]) for p in predicts]
        else:
            prompt = f"<start_of_turn>user\n{tokops.gemma_prompt.format(context=x)}<end_of_turn>\n<start_of_turn>model"
            print(f"Prompt to Gemma:\n{prompt}")
            predicts = generation_config["generator"](
                prompt,
                max_tokens=gen_length,
                temperature=1.0,
                top_p=0.95,
                top_k=64
            )
            predicts = [extract_gemma_suggestion(p["text"]) for p in predicts["choices"]]
    return x, predicts