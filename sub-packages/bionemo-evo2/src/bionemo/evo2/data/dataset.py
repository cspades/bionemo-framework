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


from typing import ClassVar, Dict, Optional

import torch
from megatron.core.datasets.gpt_dataset import GPTDataset


class Evo2Dataset(GPTDataset):
    """Dataset for training Evo2."""

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0

    def __getitem__(self, idx: Optional[int]) -> Dict[str, torch.Tensor]:
        """Get data at the specified index."""
        databatch: dict = super().__getitem__(idx)
        labels = databatch.get("labels", None)
        loss_mask = databatch.get("loss_mask", None)
        if labels is None or loss_mask is None:
            # No next-token labels or loss to mask.
            return databatch

        # Mask special label tags in loss.
        control_mask = torch.isin(labels, torch.tensor(self.CONTROL_TAGS, device=labels.device))
        loss_mask[control_mask] = 0
        phylotag_mask = Evo2Dataset.mask_phylogenetic_tags(
            labels,
            self.TAG_BOUNDS,
            self.TAG_CHARS,
            self.config.tokenizer.eod if self.config.tokenizer is not None else self.DEFAULT_EOD,
        )
        databatch["loss_mask"] = loss_mask * phylotag_mask

        return databatch

    @staticmethod
    def mask_phylogenetic_tags(tokenized_sequence, terminal_tag_char, other_tag_chars, eod_token_id):
        """Optimized version to create a phylonetic tag mask for batched tokenized sequences with correct handling of partial tags.

        Args:
            tokenized_sequence (torch.Tensor): A batched tensor of shape (batch_size, seq_length). If (seq_length,) is detected, it will be converted into a (1, seq_length) tensor.
            terminal_tag_char (int): The token ID representing the start and end of a phylogenetic tag ('|').
            other_tag_chars (set of int): A set of token IDs that are uniquely part of the tag ('_', ';', etc.).
            eod_token_id (int): The token ID representing the end-of-document (EOD).

        Returns:
            mask_vector (torch.Tensor): A batched mask of the same shape as tokenized_sequence where 1 represents non-tag tokens and 0 represents tokens within the masked region.
        """
        device = tokenized_sequence.device
        if len(tokenized_sequence.shape) == 1:
            tokenized_sequence = tokenized_sequence.unsqueeze(dim=0)
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
