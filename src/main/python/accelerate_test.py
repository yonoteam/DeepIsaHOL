import torch
from accelerate import Accelerator

def main():
    accelerator = Accelerator()
    print(torch.cuda.is_available())

if __name__ == "main":
    main()