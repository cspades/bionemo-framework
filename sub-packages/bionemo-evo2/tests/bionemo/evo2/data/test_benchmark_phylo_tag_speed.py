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

import random
import timeit
from typing import Tuple

import torch
from nemo.collections.llm.gpt.data.megatron.hyena.evo2_dataset import Evo2Dataset


def _construct_taxonomy_token(dropout: float = 0.0) -> str:
    """Construct a special Taxonomy token for natural language prompting of DNA generation models.

    Args:
        dropout (float): The probability of dropping out segments of the lineage. Defaults to 0.0.

    Returns:
        Optional[str]: The constructed taxonomy token or None if lineage is None.
    """
    # If dropout > 0, randomly drop out segments of the lineage for training on incomplete lineages.
    return "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
        "somekingdom" if random.random() >= dropout else None,
        "somephylum" if random.random() >= dropout else None,
        "someclass" if random.random() >= dropout else None,
        "someorder" if random.random() >= dropout else None,
        "somefamily" if random.random() >= dropout else None,
        "lineage.genus" if random.random() >= dropout else None,
        "lineage.speciescactaca" if random.random() >= dropout else None,
    )


def mask_phylogenetic_tags_old(tokenized_sequence, terminal_tag_char, other_tag_chars, eod_token_id):
    """
    Optimized version to create a phylonetic tag mask for batched tokenized sequences with correct handling of partial tags.
    Args:
    - tokenized_sequence (torch.Tensor): A batched tensor of shape (batch_size, seq_length).
    - terminal_tag_char (int): The token ID representing the start and end of a phylogenetic tag ('|').
    - other_tag_chars (set of int): A set of token IDs that are uniquely part of the tag ('_', ';', etc.).
    - eod_token_id (int): The token ID representing the end-of-document (EOD).
    Returns:
    - mask_vector (torch.Tensor): A batched mask of the same shape as tokenized_sequence where
      1 represents non-tag tokens and 0 represents tokens within the masked region.
    """
    device = tokenized_sequence.device
    batch_size, seq_len = tokenized_sequence.shape
    mask_vector = torch.ones_like(tokenized_sequence, dtype=torch.int, device=device)

    # To address when unbalanced tags are present
    terms = torch.tensor([0, seq_len - 1], device=device)
    other_tags = torch.tensor(list(other_tag_chars), device=device)
    for batch_idx in range(batch_size):
        tag_term_locs = torch.where(tokenized_sequence[batch_idx] == terminal_tag_char)[0]
        tag_end_locs = torch.where(tokenized_sequence[batch_idx] == eod_token_id)[0]

        merged_tags = torch.cat((terms, tag_term_locs, tag_end_locs)).sort()[0]
        merged_tags = merged_tags.unique()

        start = 0  # First and last locations are always added
        for end in merged_tags[1:]:
            if torch.isin(tokenized_sequence[batch_idx][start:end], other_tags).sum() > 0:
                # end token is not part of the tag
                if eod_token_id == tokenized_sequence[batch_idx][end]:
                    end = end - 1
                if eod_token_id == tokenized_sequence[batch_idx][start]:
                    start = start + 1

                mask_vector[batch_idx][start : (end + 1)] = 0
            start = end
    return mask_vector


def benchmark_phylo_tag_masking(num_iterations: int = 1000) -> Tuple[float, float]:
    """Benchmark the performance of phylogenetic tag masking functions.

    Args
        num_iterations: Number of iterations to run for timing
    """
    tax_token = _construct_taxonomy_token(dropout=0.0)
    sequence_alpha = (
        tax_token[2:]
        + "".join(random.choice("ACGTacgt") for _ in range(5000))
        + tax_token[:-25]
        + "0"
        + tax_token[36:]
        + "".join(random.choice("ACGTacgt") for _ in range(5000))
    )
    sequence = torch.tensor([ord(t) if t != "0" else 0 for t in sequence_alpha], dtype=torch.int32)

    # Time the new implementation
    new_time = timeit.timeit(
        lambda: Evo2Dataset.mask_phylogenetic_tags(sequence.unsqueeze(0), 124, {95, 59, 32}, 0),
        number=num_iterations,
    )
    print(f"New implementation average time: {new_time/num_iterations:.6f} seconds")

    # Time the old implementation
    old_time = timeit.timeit(
        lambda: mask_phylogenetic_tags_old(sequence.unsqueeze(0), 124, {95, 59, 32}, 0),
        number=num_iterations,
    )
    return old_time, new_time


def test_phylo_tag_masking_speed():
    num_iterations = 1000
    old_time, new_time = benchmark_phylo_tag_masking(num_iterations=num_iterations)
    assert old_time / num_iterations > new_time / num_iterations


if __name__ == "__main__":
    num_iterations = 1000
    old_time, new_time = benchmark_phylo_tag_masking(num_iterations=num_iterations)
    print(f"Old implementation average time: {old_time/num_iterations:.6f} seconds")
    print(f"New implementation average time: {new_time/num_iterations:.6f} seconds")
    print(f"Speed improvement: {(old_time/new_time - 1)*100:.2f}%")
