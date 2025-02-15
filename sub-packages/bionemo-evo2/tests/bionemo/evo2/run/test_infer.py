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


from bionemo.core.data.load import load
from bionemo.evo2.run.infer import infer
from bionemo.testing.megatron_parallel_state_utils import clean_parallel_state_context


RANDOM_SEED = 42


def test_run_infer():
    # Create PTL trainer.
    tensor_parallel_size = 1
    pipeline_model_parallel_size = 1
    context_parallel_size = 1
    temperature = 1.0
    top_k = 0
    top_p = 0.0
    max_new_tokens = 1

    # Generation args.
    default_prompt = (
        "|d__Bacteria;"
        + "p__Pseudomonadota;"
        + "c__Gammaproteobacteria;"
        + "o__Enterobacterales;"
        + "f__Enterobacteriaceae;"
        + "g__Escherichia;"
        + "s__Escherichia|"
    )

    # TODO (dorotat) remove PBSS source once the model is available on NGC
    checkpoint_path = load("evo2/7b-8k-zarr:1.1", source="pbss")

    with clean_parallel_state_context():
        infer(
            prompt=default_prompt,
            ckpt_dir=checkpoint_path,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            max_new_tokens=max_new_tokens,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_model_parallel_size=pipeline_model_parallel_size,
            context_parallel_size=context_parallel_size,
        )
