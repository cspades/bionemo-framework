# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-Apache2
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import argparse

from nemo.collections import llm
from nemo.collections.llm.gpt.model.hyena import PyTorchHyenaImporter


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path", type=str, required=True, help="Path to the Evo2 un-sharded (MP1) model checkpoint file."
    )
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory path for the converted model.")
    parser.add_argument(
        "--model-type",
        type=str,
        choices=["7b", "40b", "test"],
        default="7b",
        help="Model size, choose between 7b, 40b, or test (4 layers, less than 1b).",
    )
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
