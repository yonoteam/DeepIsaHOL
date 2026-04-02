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
        # print(f"Generating for prompt: '{x}'...")
        predicts = generation_config["generator"](
            x,
            max_new_tokens=gen_length,
            num_return_sequences=num_return_sequences,
            num_beams=num_beams
        )
        # print(f"Generated {len(predicts)} sequences.")
        predicts = [p["generated_text"] for p in predicts]
    elif model_type == "ollama":
        ollama_prompt = tokops.llm_prompt.format(context=x)
        ollama_options = {
            "num_predict": gen_length,
            "temperature": 1.0,
            "top_p": 0.95,
            "top_k": 64,
        }
        seen = set()
        predicts = []
        for _ in range(num_return_sequences):
            response = generation_config["generator"].generate(
                model=generation_config["ollama_model"],
                prompt=ollama_prompt,
                options=ollama_options,
            )
            generated_text = response.get("response", "")
            extracted = extract_suggestion(generated_text)
            if extracted and extracted not in seen:
                seen.add(extracted)
                predicts.append(extracted)
        if not predicts:
            predicts = ["No suggestion generated."]
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
        else:
            predicts = []
            for candidate in response.candidates:
                if candidate.content and candidate.content.parts:
                    predicts.append(extract_suggestion(candidate.content.parts[0].text))
                else:
                    logging.warning(f"Gemini candidate with empty content. Finish Reason: {candidate.finish_reason}")
            if not predicts:
                predicts = [None]

    # print(f"Prediction from model:\n{predicts[0]}")
    return x, predicts


def configure_generator(config_dict):
    """Build a generation_config dict with 'generator' and metadata for any model type.

    Shared by dfs.py (batch evaluation) and llm_server.py (socket server).
    """
    import os

    model_type = get_model_type(config_dict)
    data_format = config_dict["data_format"]

    generation_config = config_dict.get("generation_config", {}).copy()
    generation_config["data_format"] = data_format
    generation_config["model_type"] = model_type
    generation_config["use_unsloth"] = using_unsloth()

    if generation_config["use_unsloth"] or model_type in ("t5", "gemma"):
        from transformers import pipeline
        tokenizer, model = load_tok_model(config_dict)
        if model_type == "t5":
            generation_task = "text2text-generation"
        else:
            generation_task = "text-generation"
        generation_config["generator"] = pipeline(
            generation_task,
            model=model,
            tokenizer=tokenizer
        )
        logging.info(f"Loaded {model_type} model with HF pipeline")

    elif model_type == "ollama":
        import ollama
        ollama_model = config_dict["model_name"].removeprefix("ollama/")
        generation_config["generator"] = ollama.Client()
        generation_config["ollama_model"] = ollama_model
        try:
            models_list = generation_config["generator"].list()
            logging.info(f"Connected to Ollama server. Total available models: {len(models_list['models'])}")
        except Exception as e:
            logging.warning(f"Could not verify Ollama connection: {e}")

    elif model_type == "openai":
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY environment variable is not set.")
        generation_config["generator"] = OpenAI(api_key=api_key)
        generation_config["model_name"] = config_dict["model_name"]
        logging.info(f"Configured OpenAI client for model: {generation_config['model_name']}")

    elif model_type == "gemini":
        from google import genai
        from google.genai import types
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable is not set.")
        client = genai.Client(api_key=api_key)
        generation_config["generator"] = client
        generation_config["model_name"] = config_dict["model_name"]
        num_seqs = config_dict.get("generation_config", {}).get("num_return_sequences", 1)
        generation_config["gen_config"] = types.GenerateContentConfig(
            candidate_count=num_seqs,
            max_output_tokens=config_dict.get("generation_config", {}).get("gen_length", 4096),
            temperature=1.0
        )
        logging.info(f"Configured Gemini client for model: {config_dict['model_name']}")

    return generation_config
