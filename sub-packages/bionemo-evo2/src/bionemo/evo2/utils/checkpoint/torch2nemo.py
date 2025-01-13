import argparse

from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import  PyTorchHyenaImporter

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to the Evo2 un-sharded (MP1) model checkpoint file.")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory path for the converted model.")
    parser.add_argument("--model-type", type=str, choices=["7b", "40b", "test"], default="7b",
                        help="Model size, choose between 7b, 40b, or test (4 layers, less than 1b).")
    return parser.parse_args()

if __name__ == "__main__":

    # Parse args.
    args = parse_args()

    # Hyena Model Config
    if args.model_type == "7b":
        evo2_config = llm.Hyena7bConfig()
    elif args.model_type == "40b":
        evo2_config = llm.Hyena40bConfig()
    elif args.model_type == "test":
        evo2_config = llm.HyenaTestConfig()
    else:
        raise ValueError(f"Invalid model type: {args.model_type}")

    importer = PyTorchHyenaImporter(args.model_path, model_config=evo2_config)
    importer.apply(args.output_dir)