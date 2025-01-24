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


import gc
from pathlib import Path

import torch
from megatron.core.transformer.module import Float16Module

from bionemo.core.utils.dtypes import PrecisionTypes, get_autocast_dtype
from bionemo.esm2.data import tokenizer
from bionemo.esm2.model.model import ESM2Config
from bionemo.testing import megatron_parallel_state_utils


def eval_esm2(
    checkpoint_path: Path | str,
    sequences: list,
    precision: PrecisionTypes = "fp32",
    include_embeddings: bool = True,
    include_hiddens: bool = True,
    include_input_ids: bool = True,
    include_logits: bool = True,
    interactive: bool = False,
    tokenizer: tokenizer.BioNeMoESMTokenizer = tokenizer.get_tokenizer(),
):
    """Perform inference using the ESM2 model.

    Args:
        checkpoint_path (str): Path to the pre-trained ESM2 model checkpoint.
        sequences (list): List of input sequences for inference.
        precision (PrecisionTypes, optional): Precision type for computation. Defaults to "fp32".
        include_embeddings (bool, optional): Whether to include token embeddings in the output. Defaults to True.
        include_hiddens (bool, optional): Whether to include hidden states in the output. Defaults to True.
        include_input_ids (bool, optional): Whether to include input IDs in the output. Defaults to True.
        include_logits (bool, optional): Whether to include model logits in the output. Defaults to True.
        interactive (boo, optional): Whether to run forward evaluation of ESM2 in an interactive environment. Defaults to False.
        tokenizer: The tokenizer to use for tokenization. Defaults to the BioNeMoESMTokenizer.

    Returns:
        dict: A dictionary containing the outputs of the ESM2 model. The contents of the dictionary depend
        on the options specified in the function arguments.
    """
    with megatron_parallel_state_utils.distributed_model_parallel_state(interactive=interactive):
        tokens = tokenizer(sequences, return_tensors="pt", padding=True, truncation=True).to("cuda")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        dtype = get_autocast_dtype(precision)
        nemo_config = ESM2Config(
            initial_ckpt_path=checkpoint_path,
            include_embeddings=include_embeddings,
            include_hiddens=include_hiddens,
            include_input_ids=include_input_ids,
            skip_logits=not include_logits,
            params_dtype=dtype,
            pipeline_dtype=dtype,
            autocast_dtype=dtype,
            bf16=dtype is torch.bfloat16,
            fp16=dtype is torch.float16,
        )

        nemo_model = nemo_config.configure_model(tokenizer).to("cuda").eval()

        if dtype is torch.float16 or dtype is torch.bfloat16:
            nemo_model = Float16Module(nemo_config, nemo_model)
        nemo_output = nemo_model(input_ids, attention_mask)

        del nemo_model
        gc.collect()
        torch.cuda.empty_cache()

        return nemo_output
