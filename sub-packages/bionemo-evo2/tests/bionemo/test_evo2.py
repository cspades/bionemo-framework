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


import logging
from pathlib import Path
from typing import Literal, Set

import pytest
import torch
from megatron.core.transformer.module import Float16Module
from nemo.collections import llm
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning.io.pl import MegatronCheckpointIO
from transformer_engine.pytorch.utils import get_cudnn_version, get_device_compute_capability

from bionemo.core.data.load import load
from bionemo.llm.utils.weight_utils import (
    MegatronModelType,
    _key_in_filter,
    _munge_key_megatron_to_nemo2,
    _munge_sharded_tensor_key_megatron_to_nemo2,
)
from bionemo.testing.megatron_parallel_state_utils import distributed_model_parallel_state


logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)  # Capture all levels in the logger itself


def load_weights_sharded_inplace_nemo2_to_mcore(
    model: MegatronModelType,
    distributed_checkpoint_dir: str | Path,
    skip_keys_with_these_prefixes: Set[str],
    ckpt_format: Literal["zarr", "torch_dist"] = "zarr",
):
    logger.info("Start setting up state dict")
    sharded_state_dict = {
        _munge_key_megatron_to_nemo2(k): _munge_sharded_tensor_key_megatron_to_nemo2(v)
        for k, v in model.sharded_state_dict().items()
        if not _key_in_filter(
            k, skip_keys_with_these_prefixes
        )  # and "_extra_state" not in k  # extra state is needed for fp8 sharded states
    }
    MegatronCheckpointIO(save_ckpt_format=ckpt_format).load_checkpoint(
        distributed_checkpoint_dir, sharded_state_dict=sharded_state_dict
    )

@pytest.mark.parametrize("seq_len", [8_192, 16_384])
def test_golden_values(seq_len:int):
    """Step 1:
    # add local .ssh/*.pub key to eos ~/.ssh/authorized_keys
    mkdir -p arc_model/checkpoints/
    rsync -avz --progress --partial login-eos01.eos.clusters.nvidia.com:/lustre/fsw/healthcareeng_bionemo/arc_evo2/savanna_outputs/interleaved_hyena_7b arc_model/checkpoints/
    mkdir -p arc_model/gold_standards/
    rsync -avz --progress --partial login-eos01.eos.clusters.nvidia.com:/lustre/fsw/healthcareeng_bionemo/arc_evo2/savanna_outputs/interleaved_7b_golden_value.pt arc_model/gold_standards/
    rsync -avz --progress --partial login-eos01.eos.clusters.nvidia.com:/lustre/fsw/healthcareeng_bionemo/arc_evo2/savanna_outputs/final_7b_no_fp8_golden_value.pt arc_model/gold_standards/
    """
    try:
        evo2_7b_checkpoint_weights: Path = load("evo2/7b-8k-zarr:1.0") / "weights"
        gold_standard_no_fp8 = load("evo2/7b-8k-nofp8-te-goldvalue-testdata:1.0")
    except ValueError as e:
        if e.args[0].endswith("does not have an NGC URL."):
            raise ValueError(
                "Please re-run test with `BIONEMO_DATA_SOURCE=pbss py.test ...`, "
                "one or more files are missing from ngc."
            )
        else:
            raise e
    with torch.inference_mode(), distributed_model_parallel_state():
        hyena_config = llm.Hyena7bConfig(use_te=True, seq_length=seq_len)
        tokenizer = get_nmt_tokenizer(
            "byte-level",
        )
        raw_megatron_model = hyena_config.configure_model(tokenizer).eval().cuda()
        device = raw_megatron_model.parameters().__next__().device
        load_weights_sharded_inplace_nemo2_to_mcore(raw_megatron_model, evo2_7b_checkpoint_weights, {}, "zarr")
        model = Float16Module(hyena_config, raw_megatron_model)
        input_seq = "GAAATTAGCGCGTCCGGAATGATACGAGGGGAAACGAAATTTTGAATTAATGGAGAAAAAAGACGAGAAACCTTAAGCAAAAAAATTTTAGCTTCGAATATTTATTAATTTCTGAGATGTTGTTAAACGATTTTCGATTCCAAGTTGTGCGCACGAACGTTATTGCAAATAAATGCTGCTTATTCGGATGTTTCCACGATCTTTGTTGCAATGGTAGTCGAGTACCCGATAACCCAATTTCGTTACATCGGCCTATCTGTAGAATATCCAATCTATGGTTCATAAAAAATCTGATCGTTTGTTTTTAAGAAATTAAACGCGTTAAATTGAACGAATTTCGAATACCGGTCTTAGCGAAGGACCTCCCCTCTTGCTTGCGTATTGCCCCGCGAAATTTCTTTTCGGCGATGAACGATACAAAAAATTCTATCGAATGTTACTTCTATTCTCTGCCTCGTCTATGACTTGGAGATTGGTCTATGTCGTTCGTTTTCTCGCGAGTTTCCAATATGTCCGTAGTATGTGAACGCTGGTATTCGTGAAGATAAATTATTGTTTTTACAATTTCTTTCAAAAATATATAATTTTAATTTATATAAT"
        input_ids = torch.tensor(tokenizer.text_to_ids(input_seq)).int().unsqueeze(0).to(device)
        position_ids = torch.arange(len(input_seq)).unsqueeze(0).to(device)
        attention_mask = None
        outputs = model(input_ids=input_ids, position_ids=position_ids, attention_mask=attention_mask)
        gold_standard_no_fp8 = torch.load(gold_standard_no_fp8).to(
            device=outputs.device, dtype=outputs.dtype
        )
        our_generation_str = "".join(
            [chr(idx) for idx in outputs.softmax(dim=-1).argmax(dim=-1).flatten().detach().cpu().numpy().tolist()]
        )
        their_generation_str_no_fp8 = "".join(
            [
                chr(idx)
                for idx in gold_standard_no_fp8.softmax(dim=-1)
                .argmax(dim=-1)
                .flatten()
                .detach()
                .cpu()
                .numpy()
                .tolist()
            ]
        )
        char_matches_ours_v_theirs_no_fp8 = [
            our_generation_str[i] == their_generation_str_no_fp8[i] for i in range(len(their_generation_str_no_fp8))
        ]
        token_similarity_vs_no_fp8 = sum(char_matches_ours_v_theirs_no_fp8) / len(char_matches_ours_v_theirs_no_fp8)
        # We can get exact very tight numerical precision on H100 with cudnn 9.5+ (nvidia docker 24.10-py3 or better)
        if get_cudnn_version() >= (9, 5, 0) and get_device_compute_capability() >= (9, 0):
            assert token_similarity_vs_no_fp8 == 1.0
            torch.testing.assert_close(outputs, gold_standard_no_fp8)
        else:
            assert token_similarity_vs_no_fp8 >= 0.996
            torch.testing.assert_close(outputs, gold_standard_no_fp8, atol=0.3, rtol=3)
