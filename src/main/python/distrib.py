import logging
import torch
from accelerate import Accelerator

def reduce_sum_int(value: int, accelerator, device=None):
    if not device:
        device=accelerator.device
    return accelerator.reduce(torch.tensor(value, device=accelerator.device), reduction="sum")

def log_cuda_info(accelerator):
    device = accelerator.device
    accelerator.wait_for_everyone()
    logging.info(f"Process {accelerator.process_index}: Using device {device}")
    accelerator.wait_for_everyone()
    logging.info(f"torch.cuda.current_device(): {torch.cuda.current_device() if torch.cuda.is_available() else 'No CUDA available'}")

def wrap_w_accelerator(f):
    try:
        accelerator = Accelerator(mixed_precision="bf16")
        if accelerator.is_main_process:
            logging.info(f"Accelerator started on {accelerator.num_processes} processes.")
            logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
        log_cuda_info(accelerator)

        # Main body
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

if __name__ == "__main__":
    try:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
        accelerator = Accelerator()
        log_cuda_info(accelerator)
    except Exception as e:
        logging.error(f"An error occurred: {e}")
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
    