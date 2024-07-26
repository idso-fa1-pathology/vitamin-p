import argparse
import os
from datetime import datetime
import torch

def get_gpu_info():
    if not torch.cuda.is_available():
        return "No CUDA GPUs available"
    
    gpu_info = []
    for i in range(torch.cuda.device_count()):
        gpu_info.append(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    return "\n".join(gpu_info)

def main():
    parser = argparse.ArgumentParser(description="Test script for GPU job")
    parser.add_argument("-d", "--data_dir", required=True, help="Directory containing input data")
    parser.add_argument("-o", "--output_dir", required=True, help="Directory for output files")
    args = parser.parse_args()

    print(f"Data directory: {args.data_dir}")
    print(f"Output directory: {args.output_dir}")

    output_file = os.path.join(args.output_dir, "gpu_info.txt")
    
    try:
        os.makedirs(args.output_dir, exist_ok=True)  # Ensure output directory exists
        with open(output_file, 'w') as f:
            f.write(f"Hello from GPU job!\n")
            f.write(f"Current time: {datetime.now()}\n")
            f.write(f"Data directory: {args.data_dir}\n")
            f.write(f"Output directory: {args.output_dir}\n\n")
            f.write("GPU Information:\n")
            f.write(f"CUDA is available: {torch.cuda.is_available()}\n")
            f.write(f"Number of GPUs: {torch.cuda.device_count()}\n\n")
            f.write(get_gpu_info())
        print(f"Successfully wrote output to {output_file}")
    except Exception as e:
        print(f"Error writing to {output_file}: {str(e)}")

if __name__ == "__main__":
    main()