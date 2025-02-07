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


import pytest
import torch
from nemo.collections.llm.gpt.data.megatron.hyena import Evo2Dataset


@pytest.fixture
def tag_tokens():
    """Standard tokens for phylogenetic tag tests, defined in Evo2_DataseT:

    CONTROL_TAGS: ClassVar[list[int]] = [64, 35]  # '@' tag for splice splits/windows, '#' for contig splits
    TAG_BOUNDS = 124  # start and end delim: '|'
    TAG_CHARS: ClassVar[set[int]] = {95, 59, 32}  # chars only found in control tags: _, ;, space
    DEFAULT_EOD = 0
    """
    return {
        "terminal": 124,  # |
        "other_chars": {95, 59, 32},  # _, ;, space
        "eod": 0,  # end of document token
    }


def test_mask_phylogenetic_tags_with_eod(tag_tokens):
    """Tests handling of EOD tokens within tag context.

    Since we want to ensure the model only learns to output {A,C,G,T}, even EOD tokens
    within a tag context should be masked to prevent the model from learning to
    output non-DNA tokens.

    Example sequence: token | _ EOD | token
    Expected masking:   1   0 0  0  0   1
    """
    sequence = torch.tensor([65, 124, 95, 0, 124, 65])  # token|_<EOD>|token

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # |
        other_tag_chars=tag_tokens["other_chars"],  # _, ;, space
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor([1, 0, 0, 0, 0, 1])
    assert torch.equal(mask, expected_mask)


def test_mask_phylogenetic_tags_middle(tag_tokens):
    """Tests masking a phylogenetic tag that appears in the middle of a DNA sequence.

    The sequence contains:
    1. Normal DNA (ATG)
    2. A phylo tag (|info_tag|)
    3. More DNA (TCGA)

    Expected behavior: The DNA should be unmasked (1s) while everything between
    and including the pipe characters should be masked (0s), as it's a valid phylo tag.
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,
            105,
            110,
            102,
            111,
            95,
            116,
            97,
            103,
            124,  # |info_tag|
            84,
            67,
            71,
            65,  # TCGA
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],  # |
        other_tag_chars=tag_tokens["other_chars"],  # _, ;, space
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            1,
            1,
            1,  # DNA unmasked
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # phylo tag masked
            1,
            1,
            1,
            1,  # DNA unmasked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_start(tag_tokens):
    """Tests handling a sequence that starts with a partial phylogenetic tag.

    The sequence starts with characters that would be inside a phylo tag,
    followed by a closing pipe and DNA. Since we want to prevent the model from
    learning non-DNA outputs, we mask all potential tag characters even without
    complete tag delimiters.

    Sequence: "tag;_|ATG" (starting mid-tag)
    Expected: All tag characters and delimiters masked, only DNA unmasked
    """
    sequence = torch.tensor(
        [
            116,
            97,
            103,
            59,
            95,  # tag;_
            124,  # |
            65,
            84,
            71,  # ATG
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,  # partial tag start masked
            0,  # closing pipe masked
            1,
            1,
            1,  # DNA unmasked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_mask_partial_tag_end(tag_tokens):
    """Tests handling a sequence that ends with a partial phylogenetic tag.

    The sequence contains DNA followed by an opening pipe and tag characters,
    but no closing pipe. Per requirements, we aggressively mask any potential
    tag characters to ensure the model only learns DNA bases {A,C,G,T}.

    Sequence: "ATG|info_" (ending mid-tag)
    Expected: DNA unmasked, all tag-related characters masked
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,  # |
            105,
            110,
            102,
            111,
            95,  # info_
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            1,
            1,
            1,  # DNA unmasked
            0,  # opening pipe masked
            0,
            0,
            0,
            0,
            0,  # partial tag end masked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_standalone_tag(tag_tokens):
    """Tests masking of a single complete tag with no surrounding sequence.

    Tests that a standalone tag (|tag_|) is fully masked since it contains
    non-DNA characters. This ensures the model only learns to output
    {A,C,G,T} tokens.

    Sequence: |tag_|
    Expected: All tokens masked (all zeros)
    """
    sequence = torch.tensor([124, 116, 97, 103, 95, 124])  # |tag_|
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    expected = torch.tensor([0, 0, 0, 0, 0, 0])  # All masked
    assert torch.equal(mask, expected)


def test_sequence_starting_with_tag(tag_tokens):
    """Tests sequence that begins with a complete tag followed by DNA.

    Verifies that when a sequence starts with a complete tag followed by
    DNA bases, the tag portion is masked while the DNA portion remains
    unmasked.

    Sequence: |tag_|ATG
    Expected: Tag masked (zeros), DNA unmasked (ones)
    """
    sequence = torch.tensor(
        [
            124,
            116,
            97,
            103,
            95,
            124,  # |tag_|
            65,
            84,
            71,  # ATG
        ]
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    expected = torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1])  # Tag masked, DNA unmasked
    assert torch.equal(mask, expected)


def test_sequence_ending_with_tag(tag_tokens):
    """Tests sequence that ends with a complete tag.

    Verifies that when a sequence ends with a complete tag, the DNA portion
    remains unmasked while the entire tag portion is masked.

    Sequence: ATG|tag_|
    Expected: DNA unmasked (ones), tag masked (zeros)
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,
            116,
            97,
            103,
            95,
            124,  # |tag_|
        ]
    )
    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )
    expected = torch.tensor([1, 1, 1, 0, 0, 0, 0, 0, 0])  # DNA unmasked, tag masked
    assert torch.equal(mask, expected)


def test_mask_multiple_tags(tag_tokens):
    """Tests handling multiple phylogenetic tags in sequence, demonstrating state transitions.

    This tests how the masking switches states between phylo and non-phylo regions:
    1. Starts in non-phylo state with DNA
    2. Switches to phylo state at first pipe (with tag chars)
    3. Switches back to non-phylo at closing pipe
    4. Pattern repeats for second tag

    Sequence: "ATG|tag_1|CG|tag_2|AT"
    Expected: Only DNA sequences should remain unmasked
    """
    sequence = torch.tensor(
        [
            65,
            84,
            71,  # ATG
            124,
            116,
            97,
            103,
            95,
            49,
            124,  # |tag_1|
            67,
            71,  # CG
            124,
            116,
            97,
            103,
            95,
            50,
            124,  # |tag_2|
            65,
            84,  # AT
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            1,
            1,
            1,  # DNA unmasked
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # first tag masked
            1,
            1,  # DNA unmasked
            0,
            0,
            0,
            0,
            0,
            0,
            0,  # second tag masked
            1,
            1,  # DNA unmasked
        ]
    )
    assert torch.equal(mask, expected_mask)


def test_mask_dna_after_pipe(tag_tokens):
    """Tests the scenario where we have a pipe followed by DNA sequence.

    This tests the edge case of a pipe character appearing at the start of a sequence.
    Even if DNA follows, we mask the pipe character to prevent the model from
    learning to output non-DNA tokens.

    Sequence: "|ATG" (pipe followed by DNA)
    Expected: Pipe masked, DNA unmasked
    """
    sequence = torch.tensor(
        [
            124,  # |
            65,
            84,
            71,  # ATG
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor([0, 1, 1, 1])  # Pipe masked, DNA unmasked
    assert torch.equal(mask, expected_mask)


def test_ambiguous_dna_char_followed_by_tag_start(tag_tokens):
    """Tests handling of an ambiguous DNA character followed by a tag start.

    When we see a character that could be either DNA or the end of a truncated tag
    followed by a pipe, we should mask both for safety since we can't disambiguate
    whether the character was part of a tag.

    Sequence: "t|AAAT" (t could be DNA or end of tag)
    Expected: First t and pipe masked (0), AAAT unmasked (1)
    """
    sequence = torch.tensor(
        [
            116,  # t (ambiguous - could be DNA or end of tag)
            124,  # |
            65,  # A
            65,  # A
            65,  # A
            84,  # T
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor([0, 0, 1, 1, 1, 1])  # Ambiguous t and pipe masked, DNA unmasked
    assert torch.equal(mask, expected_mask)


def test_dna_followed_by_unambiguous_tag_start(tag_tokens):
    """Tests handling of DNA sequence followed by clear tag start.

    When we see DNA followed by |d, it's unambiguous - the d clearly indicates
    the start of a phylogenetic tag (d__), so we can safely unmask the DNA and
    mask the tag portion.

    Sequence: "AAAT|d" (AAAT is DNA, |d starts tag)
    Expected: AAAT unmasked (1), |d masked (0)
    """
    sequence = torch.tensor(
        [
            65,  # A
            65,  # A
            65,  # A
            84,  # T
            124,  # |
            100,  # d (clearly starts d__ tag)
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor([1, 1, 1, 1, 0, 0])  # DNA unmasked, tag start masked
    assert torch.equal(mask, expected_mask)


def test_double_partial_tags_with_dna_middle(tag_tokens):
    """Tests a sequence that has partial tags at both ends with DNA in the middle.

    Tests the specific case where a sequence slice cuts through phylogenetic tags
    on both ends, with valid DNA sequence in the middle. The behavior we want is:
    1. The partial tag at the start should be masked
    2. The DNA in the middle should be unmasked
    3. The partial tag at the end should be masked

    Sequence: "cacata|acagataaaataTACAGGGAATA|d__"
    Expected: First partial tag masked (0s), middle DNA unmasked (1s), end tag masked (0s)
    """
    sequence = torch.tensor(
        [
            99,
            97,
            99,
            97,
            116,
            97,  # cacata
            124,  # |
            97,
            99,
            97,
            103,
            97,
            116,
            97,
            97,
            97,
            97,
            116,
            97,  # acagataaaata
            84,
            65,
            67,
            65,
            71,
            71,
            71,
            65,
            65,
            84,
            65,  # TACAGGGAATA
            124,  # |
            100,
            95,
            95,  # d__
        ]
    )

    mask = Evo2Dataset.mask_phylogenetic_tags(
        tokenized_sequence=sequence,
        terminal_tag_char=tag_tokens["terminal"],
        other_tag_chars=tag_tokens["other_chars"],
        eod_token_id=tag_tokens["eod"],
    )

    expected_mask = torch.tensor(
        [
            0,
            0,
            0,
            0,
            0,
            0,  # partial start tag masked
            0,  # pipe masked
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # middle DNA unmasked
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,
            1,  # middle DNA unmasked
            0,  # pipe masked
            0,
            0,
            0,  # partial end tag masked
        ]
    )

    assert torch.equal(mask, expected_mask)
