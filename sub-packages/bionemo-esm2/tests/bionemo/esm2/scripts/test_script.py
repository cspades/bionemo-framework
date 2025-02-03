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


import torch

from bionemo.esm2.api import ESM2Config
from bionemo.esm2.scripts.infer_esm2 import infer_model


ckpt_path = "sub-packages/results/esm2/dev/checkpoints/checkpoint-step=4-consumed_samples=320.0-last"


result_dir = "sub-packages/results-test"
infer_model(
    data_path="sub-packages/esm2_data/protein_dataset.csv",
    checkpoint_path=ckpt_path,
    results_path=result_dir,
    min_seq_length=1024,
    prediction_interval="epoch",
    include_hiddens=True,
    precision="fp32",
    include_embeddings=True,
    include_input_ids=True,
    include_logits=True,
    micro_batch_size=3,  # dataset length (10) is not multiple of 3; this validates partial batch inference
    config_class=ESM2Config,
)
results = torch.load(f"{result_dir}/predictions__rank_0.pt")
