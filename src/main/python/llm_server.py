# llm_server.py
# A cross-platform TCP server for hosting a local Hugging Face Transformers LLM.
# Listens on localhost:5006 for newline-delimited JSON requests and returns JSON responses.

import json
import time
import signal
import socket
import logging
import threading
from typing import BinaryIO

import proofs
import config_ops
# import generation_ops as genops

# from transformers import pipeline # after genops including unsloth

LOCAL_HOST = "127.0.0.1"
PORT = 5006
RECV_BUFFER = 4096 # buffer size for socket recv
shutdown_event = threading.Event()


def make_proof_info(str_mssg, data_format):
    try:
        proof_json = json.loads(str_mssg)
    except json.JSONDecodeError as e:
        logging.error(f"Failed to decode JSON from client message: {e}")
        proof_json = {} 
    return {
        "proof_so_far": proof_json.get("proof_so_far", ""),
        "last_usr_state": proof_json.get("user_state", ""),
        "proof_data": proofs.str_ops.add_spk_data(proof_json, [], data_format=data_format)
    }

def prompt_llm(buffer, generation_config):
    end_of_prompt = b"<|eop|>" 

    if end_of_prompt in buffer:
        client_byte_mssg, remaining_buffer = buffer.split(end_of_prompt, 1)
        client_str_mssg = client_byte_mssg.decode("utf-8")
        proof_info = make_proof_info(client_str_mssg, generation_config["data_format"])
        _, predicts = genops.generate_predicts(proof_info, generation_config)
        response_text = predicts[0] if predicts and predicts[0] is not None else "No prediction generated."
        response_bytes = response_text.encode("utf-8") + end_of_prompt
        return response_bytes, remaining_buffer

    return b"", buffer

def echo_input(buffer):
    end_of_prompt = b"<|eop|>" 

    if end_of_prompt in buffer:
        client_byte_mssg, remaining_buffer = buffer.split(end_of_prompt, 1)
        client_str_mssg = client_byte_mssg.decode("utf-8")
        response_text = client_str_mssg
        response_bytes = response_text.encode("utf-8") + end_of_prompt
        return response_bytes, remaining_buffer

    return b"", buffer

def handle_client(conn: BinaryIO, addr: tuple, generation_config):
    logging.info(f"Connected on address {addr}")
    buffer = b""
    try:
        conn.settimeout(1.0)
        while not shutdown_event.is_set():
            try:
                data = conn.recv(RECV_BUFFER)
            except socket.timeout:
                continue # only stop if shutdown_event is set
            except ConnectionResetError:
                logging.error(f"[{addr}] Connection reset by peer.")
                break
            except Exception as e:
                logging.error(f"[{addr}] Error receiving data: {e}")
                break

            if not data:
                logging.info(f"[{addr}] Client disconnected gracefully.")
                break
            buffer += data

            response, buffer = echo_input(buffer)
            if response:
                print(f"Response: {response.decode('utf-8')}")
                conn.sendall(response)

    except Exception as e:
        if not shutdown_event.is_set():
            logging.error(f"[{addr}] Error in handle_client: {e}")
    finally:
        conn.close()
        logging.info(f"[{addr}] Disconnected.")

# def configure_generation(config_dict):
#     model_type = genops.get_model_type(config_dict)
#     data_format = config_dict["data_format"]
#     tokenizer, model = genops.load_tok_model(config_dict)

#     if model_type == "t5":
#         generation_task = "text2text-generation"
#     elif model_type == "gemma":
#         generation_task = "text-generation"

#     generation_config = config_dict.get("generation_config", {}).copy()
#     generation_config["data_format"] = data_format
#     generation_config["model_type"] = model_type
#     generation_config["generator"] = pipeline(
#         generation_task,
#         model=model,
#         tokenizer=tokenizer
#     )
#     return generation_config

def launch_server(config_dict):
    generation_config = {} # configure_generation(config_dict)
    global server_socket_global
    server_socket_global = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_global.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # handle Ctrl-C begin
    def shutdown_signal_handler(sig, frame):
        shutdown_event.set()
        time.sleep(0.1)
    signal.signal(signal.SIGINT, shutdown_signal_handler) 
    # handle Ctrl-C end

    try:
        server_socket_global.bind((LOCAL_HOST, PORT))
        server_socket_global.listen(5)
        server_socket_global.settimeout(1.0) # check `shutdown_event` every 1 second
        logging.info(f"LLM server listening on {LOCAL_HOST}:{PORT}")

        active_threads = []
        while not shutdown_event.is_set():
            try:
                conn, addr = server_socket_global.accept()
                client_thread = threading.Thread(
                    target=handle_client,
                    args=(conn, addr, generation_config),
                    daemon=True
                )
                client_thread.start()
                active_threads.append(client_thread)
                active_threads = [t for t in active_threads if t.is_alive()]

            except socket.timeout:
                continue
            except OSError as e:
                if shutdown_event.is_set():
                    logging.error(f"LLM server socket error during shutdown: {e}")
                    break
                logging.error(f"Error accepting connection: {e}")
                break
            except Exception as e:
                logging.error(f"Unexpected error in LLM server main loop: {e}")
                break
    except Exception as e:
        logging.error(f"LLM server main loop error: {e}")
    finally:
        logging.info("LLM server shutdown sequence initiated...")
        shutdown_event.set() # ensure for the client threads
        if server_socket_global:
            logging.info("Closing LLM server socket.")
            server_socket_global.close()
        logging.info("LLM server has shut down.")


if __name__ == "__main__":
    info="Launches a server hosting an LLM awaiting for prompts as specified in the input JSON configuration."
    config_dict = config_ops.parse_path(tool_explanation=info)
    config_ops.setup_logging("llm_server.log")
    launch_server(config_dict)