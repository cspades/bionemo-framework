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

from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm.inference import generate, setup_model_and_tokenizer


def parse_args():
    """Parse arguments for Evo2 inference."""
    ap = argparse.ArgumentParser()

    # generation args:
    default_prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )
    ap.add_argument("--prompt", type=str, default=default_prompt, help="Prompt for generation")
    ap.add_argument(
        "--ckpt-dir", type=str, required=True, help="Path to checkpoint directory containing pre-trained Hyena model."
    )
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature during sampling")
    ap.add_argument("--top-k", type=int, default=4, help="Top K during sampling")
    ap.add_argument("--top-p", type=float, default=1.0, help="Top P during sampling")
    ap.add_argument("--cached-generation", type=bool, default=True, help="Use KV caching during generation")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Max new tokens during sampling")
    ap.add_argument("--repetition-penalty", type=float, default=1.0, help="Repetition penalty during sampling")
    ap.add_argument("--penalty-alpha", type=float, default=0.0, help="Penalty alpha during sampling")
    # output args:
    ap.add_argument("--sequence-fasta", type=str, default="sequence.fasta", help="Sequence fasta file")
    ap.add_argument("--proteins-fasta", type=str, default="proteins.fasta", help="Proteins fasta file")
    ap.add_argument("--structure-pdb", type=str, default="structure.pdb", help="Structure PDB file")
    # misc args:
    ap.add_argument("--devices", type=str, default="cuda:0", help="Device for generation")
    ap.add_argument("--seed", type=int, default=12345, help="Random seed")

    return ap.parse_args()


def main():
    """Inference workflow for Evo2."""
    # Parse args.
    args = parse_args()

    # Load and wrap model for inferencing.
    model, tokenizer = setup_model_and_tokenizer(args.ckpt_dir)

    # Generate.
    infer_params = CommonInferenceParams(
        args.temperature, args.top_k, args.top_p, return_log_probs=False, num_tokens_to_generate=args.max_new_tokens
    )
    # transformers generate method has more options than NeMo/Megatron.
    results = generate(model, tokenizer, args.prompt, random_seed=args.seed, inference_params=infer_params)
    print(results)


if __name__ == "__main__":
    main()
