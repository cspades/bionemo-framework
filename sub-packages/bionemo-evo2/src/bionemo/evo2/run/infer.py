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

import nemo.lightning as nl
import torch
from megatron.core.inference.common_inference_params import CommonInferenceParams
from nemo.collections.llm import generate
from nemo.utils import logging


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
    ap.add_argument(
        "--prompt",
        type=str,
        default=default_prompt,
        help="Prompt to generate text from Evo2. Defaults to a phylogenetic lineage tag for E coli.",
    )
    ap.add_argument(
        "--ckpt-dir", type=str, required=True, help="Path to checkpoint directory containing pre-trained Evo2 model."
    )
    ap.add_argument("--temperature", type=float, default=1.0, help="Temperature during sampling for generation.")
    ap.add_argument("--top-k", type=int, default=0, help="Top K during sampling for generation.")
    ap.add_argument("--top-p", type=float, default=0.0, help="Top P during sampling for generation.")
    ap.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of tokens to generate.")
    # compute args:
    ap.add_argument("--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1.")
    ap.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    ap.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    # output args:
    ap.add_argument(
        "--output-file",
        type=str,
        default=None,
        help="Output file containing the generated text produced by the Evo2 model. If not provided, the output will be logged.",
    )
    # extra:
    ap.add_argument(
        "--ckpt-format",
        type=str,
        choices=["torch_dist", "zarr"],
        default="torch_dist",
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated.",
    )

    return ap.parse_args()


def main():
    """Inference workflow for Evo2."""
    # Parse args.
    args = parse_args()

    # Create PTL trainer.
    trainer = nl.Trainer(
        accelerator="gpu",
        strategy=nl.MegatronStrategy(
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_parallel_size=args.context_parallel_size,
            pipeline_dtype=torch.bfloat16,
            ckpt_load_optimizer=False,  # Needs to be false for a normal model checkpoint.
            ckpt_save_optimizer=False,
            ckpt_async_save=False,
            save_ckpt_format=args.ckpt_format,
        ),
        log_every_n_steps=1,
        limit_val_batches=10,
        num_sanity_val_steps=0,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
        ),
    )

    # transformers generate method has more options than NeMo/Megatron.
    results = generate(
        path=args.ckpt_dir,
        prompts=[args.prompt],
        trainer=trainer,
        inference_params=CommonInferenceParams(
            args.temperature,
            args.top_k,
            args.top_p,
            return_log_probs=False,
            num_tokens_to_generate=args.max_new_tokens,
        ),
        text_only=True,
    )

    if torch.distributed.get_rank() == 0:
        if args.output_file is None:
            logging.info(results)
        else:
            with open(args.output_file, "w") as f:
                f.write(f"{results}\n")


if __name__ == "__main__":
    main()
