# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Part of project DeepIsaHOL. Generic operations for prompting LLMs for proof generation.

import re
import json
import logging
from typing import Optional

import torch
import eval_t5

import tokenizer_ops as tokops

def using_unsloth():
    return torch.cuda.is_available()

def get_model_type(config_dict):
    model_name = config_dict["model_name"]
    lower_case_model = model_name.lower()
    if "ollama" in lower_case_model:
        model_type = "ollama"
    elif "t5" in lower_case_model:
        model_type = "t5"
    elif "gemma" in lower_case_model:
        model_type = "gemma"
    elif "gpt" in lower_case_model or "openai" in lower_case_model:
        model_type = "openai"
    elif "gemini" in lower_case_model:
        model_type = "gemini"
    else:
        raise ValueError(f"Unsupported model type: {model_name}. Supported: T5, Gemma, Ollama, OpenAI and Gemini.")
    logging.info(f"Model type set to {model_type} based on model name {model_name}.")
    return model_type

def load_ollama_tuned_objs(config_dict):
    from llama_cpp import Llama, LlamaTokenizer

    model = Llama(
        model_path=config_dict["models_dir"],
        n_ctx=tokops.get_gemma_context_length(config_dict["data_format"]),
        n_gpu_layers=-1, # all on GPU if possible
        verbose=False
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

def extract_suggestion(text: str) -> Optional[str]:
    if not text:
        return None
    logging.info(f"Raw LLM output: {text}")

    # standard case
    pattern = r"<SUGGESTION>(.*?)</SUGGESTION>"
    match = re.search(pattern, text, re.DOTALL)
    if match:
        suggestion = match.group(1).strip()
        suggestion = suggestion.replace("```isar", "").replace("```", "").strip()
        return suggestion

    # truncated case
    pattern_open = r"<SUGGESTION>(.*)"
    match_open = re.search(pattern_open, text, re.DOTALL)
    if match_open:
        logging.warning("Found <SUGGESTION> but no closing tag. Returning partial content.")
        suggestion = match_open.group(1).strip()
        suggestion = suggestion.replace("```isar", "").replace("```", "").strip()
        return suggestion

    # markdown fallback: looking for code blocks if tags are missing
    code_block = r"```(?:isar)?\n(.*?)```"
    match_code = re.search(code_block, text, re.DOTALL)
    if match_code:
        logging.warning("No tags found. Returning Markdown code block content.")
        return match_code.group(1).strip()
        
    # last fallback
    if text.strip():
        logging.warning("No structure found. Returning raw text.")
        return text.strip()

    logging.warning(f"No suggestion found in generated text: {text}")
    return None

def generate_predicts(prf_info: dict, generation_config: dict) -> tuple[str, list[Optional[str]]]:
    data_format = generation_config["data_format"]
    model_type = generation_config["model_type"]
    gen_length = generation_config.get("gen_length", 4096)
    num_return_sequences = generation_config.get("num_return_sequences", 1)
    num_beams = generation_config.get("num_beams", 1)
    using_unsloth = generation_config["use_unsloth"]

    x_dict = {}
    x_dict["proof_so_far"] = prf_info.get("proof_so_far", "")
    x_dict["last_usr_state"] = prf_info.get("last_usr_state", "")
    x_dict.update(prf_info.get("proof_data", {}))
    x = json.dumps(x_dict)

    if model_type == "t5":
        x = "isabelle next step: " + x if "finetune" in data_format else x  
        # print(f"Generating for prompt: '{x}'...")
        predicts = generation_config["generator"](
            x, 
            max_new_tokens=gen_length, 
            num_return_sequences=num_return_sequences,
            num_beams=num_beams
        )
        # print(f"Generated {len(predicts)} sequences.")
        predicts = [p["generated_text"] for p in predicts]
    elif model_type == "ollama":
        response = generation_config["generator"].generate(
            model=generation_config["ollama_model"],
            prompt=tokops.llm_prompt.format(context=x),
            options={
                "num_predict": gen_length,
                "temperature": 1.0,
                "top_p": 0.95,
                "top_k": 64,
            }
        )
        generated_text = response.get("response", "")
        extracted_suggestion = extract_suggestion(generated_text)
        predicts = [extracted_suggestion] if extracted_suggestion is not None else ["No suggestion generated."]
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
            predicts = [extract_suggestion(p["generated_text"][1]["content"]) for p in predicts]
        else:
            prompt = f"<start_of_turn>user\n{tokops.llm_prompt.format(context=x)}<end_of_turn>\n<start_of_turn>model"
            # print(f"Prompt to Gemma:\n{prompt}")
            predicts = generation_config["generator"](
                prompt,
                max_tokens=gen_length,
                temperature=1.0,
                top_p=0.95,
                top_k=64,
                echo=False
            )
            predicts = [extract_suggestion(p["text"]) for p in predicts["choices"]]
    elif model_type == "openai":
        prompt = tokops.llm_prompt2.format(context=x)
        response = generation_config["generator"].chat.completions.create(
            model=generation_config["model_name"],
            messages=[{"role": "user", "content": prompt}],
            max_completion_tokens=gen_length,
            n=num_return_sequences,
            temperature=1.0
        )
        predicts = [extract_suggestion(choice.message.content) for choice in response.choices]
    elif model_type == "gemini":
        prompt = tokops.llm_prompt2.format(context=x)
        response = generation_config["generator"].models.generate_content(
            model=generation_config["model_name"],
            contents=prompt,
            config=generation_config["gen_config"]
        )
        if not response.candidates:
            logging.error("Gemini returned no candidates.")
            predicts = [None]
        elif not response.candidates[0].content.parts:
            finish_reason = response.candidates[0].finish_reason
            logging.error(f"Gemini returned empty content. Finish Reason: {finish_reason}")
            predicts = [None]
        else:
            predicts = [extract_suggestion(response.text)]

    # print(f"Prediction from model:\n{predicts[0]}")
    return x, predicts