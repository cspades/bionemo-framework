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


from pathlib import Path

import pytest

from bionemo.evo2.data.preprocess import Evo2Preprocessor
from bionemo.evo2.utils.config import Evo2PreprocessingConfig
from bionemo.noodles.nvfaidx import NvFaidx


ALU_SEQUENCE: str = (
    "GGCCGGGCGCGGTGGCTCACGCCTGTAATCCCAGCACTTTGGGAGGCCGAGGCGGGCGGATCACGAGGTC"
    "aggagatcgagaccatcctggctaacacggtgaaaccccgtctctactaaaaatacaaaaaattagccgggc"
    "GTGGTGGCGCGCGCCTGTAATCCCAGCTACTCGGGAGGCTGAGGCAGGAGAATGGCGTGAACCCGGGAGGCG"
    "GAGCTTGCAGTGAGCCGAGATCGCGCCACTGCACTCCAGCCTGGGCGACAGAGCGAGACTCCGTCTCAAAAA"
)


def create_preprocessing_config(
    tmp_path: Path, sample_data_path: Path, output_prefix: str = "test_alu_uint8_distinct"
) -> Evo2PreprocessingConfig:
    """Creates a preprocessing configuration with test settings."""
    config_dict = {
        "datapaths": [str(sample_data_path)],
        "output_dir": str(tmp_path),
        "output_prefix": output_prefix,
        "train_split": 0.6,
        "validation_split": 0.2,
        "test_split": 0.2,
        "overwrite": True,
        "embed_reverse_complement": True,
        "random_reverse_complement": 0.0,
        "random_lineage_dropout": 0.0,
        "include_sequence_id": False,
        "transcribe": "back_transcribe",
        "indexed_dataset_dtype": "uint8",
        "tokenizer_type": "Byte-Level",
        "vocab_file": None,
        "vocab_size": None,
        "merges_file": None,
        "pretrained_tokenizer_model": None,
        "special_tokens": None,
        "fast_hf_tokenizer": True,
        "append_eod": True,
        "enforce_sample_length": None,
        "ftfy": False,
        "workers": 1,
        "preproc_concurrency": 100000,
        "chunksize": 25,
        "drop_empty_sequences": True,
        "nnn_filter": True,
    }
    return Evo2PreprocessingConfig(**config_dict)


def create_fasta_file(
    fasta_file_path: Path,
    num_sequences: int,
    sequence_length: int,
    repeating_dna_pattern: str = ALU_SEQUENCE,
    max_line_length: int = 80,
) -> Path:
    """Creates a fasta file with the given number of sequences, sequence length, and repeating dna pattern. Each contig uses a shifted version of the repeating pattern."""
    with open(fasta_file_path, "w") as f:
        for i in range(num_sequences):
            # get the repeating pattern shifted by i for this contig
            repeat_pattern_for_contig = repeating_dna_pattern[i:] + repeating_dna_pattern[:i]
            # repeat the pattern enough times to reach the desired sequence length
            if sequence_length <= len(repeat_pattern_for_contig):
                contig_output = repeat_pattern_for_contig[:sequence_length]
            else:
                # Calculate how many complete repeats we need
                num_repeats = sequence_length // len(repeat_pattern_for_contig)
                remainder = sequence_length % len(repeat_pattern_for_contig)
                contig_output = repeat_pattern_for_contig * num_repeats + repeat_pattern_for_contig[:remainder]
            # verify the length of the contig is as expected
            assert len(contig_output) == sequence_length
            # Fold the contig output into lines of max_line_length
            contig_output = "\n".join(
                contig_output[i : i + max_line_length] for i in range(0, sequence_length, max_line_length)
            )
            # write to the fasta file with the actual contig_output, not the repeating pattern
            f.write(f">contig_{i}\n{contig_output}\n")
    return fasta_file_path


@pytest.mark.parametrize("target_sequence_length, num_sequences", [(123, 3), (1234, 2), (12345, 1)])
def test_created_fasta_file_has_expected_length(
    tmp_path: Path, num_sequences: int, target_sequence_length: int
) -> None:
    fasta_file_path = tmp_path / "test.fasta"
    create_fasta_file(fasta_file_path, num_sequences, target_sequence_length, repeating_dna_pattern=ALU_SEQUENCE)
    assert fasta_file_path.stat().st_size > 0
    idx = NvFaidx(fasta_file_path)
    for i, (seq_name, sequence) in enumerate(sorted(idx.items())):
        assert seq_name == f"contig_{i}"
        assert len(sequence) == target_sequence_length
        if i == 0:
            assert ALU_SEQUENCE[:target_sequence_length] in sequence


def test_preprocessor_creates_expected_files(tmp_path: Path) -> None:
    """Verifies that preprocessing creates all expected output files."""
    test_fasta_file_path = create_fasta_file(tmp_path / "test.fasta", num_sequences=10, sequence_length=10000)
    output_dir = tmp_path / "processed_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    preprocessing_config = create_preprocessing_config(
        tmp_path / "processed_data", test_fasta_file_path, output_prefix="test_alu_uint8_distinct"
    )
    preprocessor = Evo2Preprocessor(preprocessing_config)
    preprocessor.preprocess_offline(preprocessing_config)

    # Check that all expected files exist
    output_dir = Path(preprocessing_config.output_dir)
    prefix = preprocessing_config.output_prefix
    expected_files = [
        output_dir / Path(prefix + "_byte-level_" + split + suffix)
        for suffix in [".bin", ".idx"]
        for split in ["train", "val", "test"]
    ]
    for file_path in expected_files:
        assert file_path.exists(), f"Expected file {file_path} was not created"
        assert file_path.stat().st_size > 0, f"File {file_path} is empty"

    # Check that no unexpected files were created
    all_files = [f for f in output_dir.iterdir() if f.is_file()]
    assert set(all_files) == set(expected_files), "Unexpected files were created"
