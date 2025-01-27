import torch
from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    device = accelerator.device
    print(f"Process {accelerator.process_index}: Using device {device}")
    print(f"torch.cuda.is_available(): {torch.cuda.is_available()}")
    print(f"torch.cuda.current_device(): {torch.cuda.current_device() if torch.cuda.is_available() else 'No CUDA available'}")

if __name__ == "__main__":
    main()