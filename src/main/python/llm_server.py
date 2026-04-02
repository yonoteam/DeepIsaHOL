# Mantainers: 
# Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz
# 
# Part of project DeepIsaHOL. A server hosting LLMs for proof generation.
# Listens on localhost:5006 for <|eop|> delimited JSON requests and returns <|eop|> delimited responses.

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import time
import signal
import socket
import logging
import threading
from typing import BinaryIO

import ollama

import dicts
import proofs
import config_ops
import generation_ops as genops

LOCAL_HOST = "127.0.0.1"
PORT = 5006
RECV_BUFFER = 4096 # buffer size for socket recv
shutdown_event = threading.Event()


def make_proof_info(mssg_dict, data_format):
    proof_json = mssg_dict
    if not isinstance(proof_json, dict):
        logging.error(f"Make proof info received a None dictionary")
        proof_json = {}
    step_dict = {
        "proof_so_far": proof_json.get("proof_so_far", ""),
        "last_usr_state": proof_json.get("user_state", ""),
        "proof_data": proofs.str_ops.add_spk_data(proof_json, {}, data_format=data_format)
    }
    return step_dict

def prompt_llm(buffer, generation_config):
    end_of_prompt = b"<|eop|>" 

    if end_of_prompt not in buffer:
        return b"", buffer
    
    try:
        client_byte_mssg, remaining_buffer = buffer.split(end_of_prompt, 1)
        client_str_mssg = client_byte_mssg.decode("utf-8")
        # print(f"Received this from client: {client_str_mssg}")
        fixed_newlines = dicts.fix_json_line_breaks(client_str_mssg)
        # print(f"This is the result of 'fixing': {fixed_newlines}")
        proof_info = make_proof_info(fixed_newlines, generation_config["data_format"])
        print(f"Prepared client info is: {proof_info}")
        _, predicts = genops.generate_predicts(proof_info, generation_config)
        # print(f"First raw predictions from model: {predicts[0] if predicts else 'None'}")
        valid_predictions = [proofs.str_ops.fix_missing_quotations(p) for p in predicts if p is not None]
        # print(f"First predictions after filtering: {valid_predictions[0] if valid_predictions else 'None'}")
        if not valid_predictions:
            response_text = "prompt_llm: No prediction generated."
        else:
            response_text = "<SEPARATOR>".join(valid_predictions)
        print(f"The prepared response is: {response_text}")
        response_bytes = response_text.encode("utf-8") + end_of_prompt
        return response_bytes, remaining_buffer

    except Exception as e:
        logging.error(f"Error in prompt_llm: {e}", exc_info=True)
        error_msg = f"Error processing request: {str(e)}"
        return error_msg.encode("utf-8") + end_of_prompt, b""

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

            response, buffer = prompt_llm(buffer, generation_config) # echo_input(buffer)
            if response:
                print(f"Response: {response.decode('utf-8')}")
                conn.sendall(response)

    except Exception as e:
        if not shutdown_event.is_set():
            logging.error(f"[{addr}] Error in handle_client: {e}")
    finally:
        conn.close()
        logging.info(f"[{addr}] Disconnected.")

def configure_generation(config_dict):
    return genops.configure_generator(config_dict)

def launch_server(config_dict):
    generation_config = configure_generation(config_dict)
    global server_socket_global
    server_socket_global = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket_global.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    # handle Ctrl+C begin
    def shutdown_signal_handler(sig, frame):
        logging.info("Shutdown (Ctrl+C) signal received")
        shutdown_event.set()
        time.sleep(0.1)
    signal.signal(signal.SIGINT, shutdown_signal_handler) 
    # handle Ctrl+C end

    try:
        server_socket_global.bind((LOCAL_HOST, PORT))
        server_socket_global.listen(5)
        server_socket_global.settimeout(1.0) # check `shutdown_event` every 1 second
        mssg = f"LLM server with model '{config_dict['model_name']}' listening on {LOCAL_HOST}:{PORT}"
        print(mssg)
        logging.info(mssg)

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
        for thread in active_threads:
            thread.join(timeout=2.0)
            
        logging.info("Server shutdown complete")


if __name__ == "__main__":
    info="Launches a server hosting an LLM awaiting for prompts as specified in the input JSON configuration."
    config_dict = config_ops.parse_path(tool_explanation=info)
    config_ops.setup_logging("llm_server.log")
    launch_server(config_dict)