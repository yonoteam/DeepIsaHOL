"""
Maintainers:
    - Jonathan Julian Huerta y Munive huertjon[at]cvut[dot]cz

Operations for distributed model training and evaluation.
"""

import logging
import torch
from accelerate import Accelerator

def reduce_sum_int(value: int, accelerator, device=None):
    if not device:
        device=accelerator.device
    return accelerator.reduce(torch.tensor(value, device=accelerator.device), reduction="sum")

def log_cuda_info_via_torch():
    cuda_available = torch.cuda.is_available()
    logging.info(f"CUDA is available?: {cuda_available}")
    if cuda_available:
        logging.info(f"How many devices?: {torch.cuda.device_count()}")
        logging.info(f"What is the current one?: {torch.cuda.current_device()}")

def log_cuda_info(accelerator):
    if accelerator.is_main_process:
        logging.info(f"Accelerator started on {accelerator.num_processes} processes.")
        logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    accelerator.wait_for_everyone()
    device = accelerator.device
    logging.info(f"Process {accelerator.process_index}: Using device {device}")
    logging.info(f"torch.cuda.current_device(): {torch.cuda.current_device() if torch.cuda.is_available() else 'No CUDA available'}")

def wrap_w_accelerator(f):
    try:
        accelerator = Accelerator() # mixed_precision="bf16", "fp16"
        log_cuda_info(accelerator)

        # Main
        f(accelerator)

    except Exception as e:
        logging.error(f"Error running parallel operation with Accelerator: {e}")
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

if __name__ == "__main__":
    def logging_setup(acc):
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
    wrap_w_accelerator(logging_setup)