# SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Arc Institute. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Michael Poli. All rights reserved.
# SPDX-FileCopyrightText: Copyright (c) 2024 Stanford University. All rights reserved
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
        "--model-size",
        type=str,
        choices=["7b", "7b_arc_1m", "40b", "40b_arc_1m", "test"],
        default="7b",
        help="Model architecture to use, choose between 7b, 40b, or test (a sub-model of 4 layers, less than 1B parameters). '_arc_1m' models have GLU / FFN dimensions that support 1M context length when trained with TP<=8.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    # Parse args.
    args = parse_args()

    # Hyena Model Config
    if args.model_size == "7b":
        evo2_config = llm.Hyena7bConfig()
    elif args.model_size == "7b_arc_1m":
        evo2_config = llm.Hyena7bARCLongContextConfig()
    elif args.model_size == "40b":
        evo2_config = llm.Hyena40bConfig()
    elif args.model_size == "40b_arc_1m":
        evo2_config = llm.Hyena40bARCLongContextConfig()
    elif args.model_size == "test":
        evo2_config = llm.HyenaTestConfig()

    importer = PyTorchHyenaImporter(args.model_path, model_config=evo2_config)
    importer.apply(args.output_dir)
