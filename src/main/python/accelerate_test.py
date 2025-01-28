import logging
import torch
from accelerate import Accelerator

def log_cuda_info(accelerator):
    device = accelerator.device
    logging.info(f"Process {accelerator.process_index}: Using device {device}")
    logging.info(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    logging.info(f"torch.cuda.current_device(): {torch.cuda.current_device() if torch.cuda.is_available() else 'No CUDA available'}")

def main():
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

if __name__ == "__main__":
    main()
    