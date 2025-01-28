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


from pathlib import Path

import pytest

from bionemo.evo2.data.preprocess import Evo2Preprocessor
from bionemo.evo2.utils.config import Evo2PreprocessingConfig


@pytest.fixture
def preprocessing_config(tmp_path: Path) -> Evo2PreprocessingConfig:
    """Creates a preprocessing configuration with test settings."""
    # grab dir where test located
    test_dir = Path(__file__).parent

    config_dict = {
        "datapaths": [str(test_dir / "test_datasets" / "mmseqs_results_rep_seq_distinct_sample_sequences.fasta")],
        "output_dir": str(tmp_path),
        "output_prefix": "test_promoters_uint8_distinct",
        "train_split": 1.0,
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


@pytest.fixture
def preprocessor(preprocessing_config: Evo2PreprocessingConfig) -> Evo2Preprocessor:
    """Creates an Evo2Preprocessor instance with test configuration."""
    return Evo2Preprocessor(preprocessing_config)


def test_preprocessor_creates_expected_files(
    preprocessor: Evo2Preprocessor, preprocessing_config: Evo2PreprocessingConfig
) -> None:
    """Verifies that preprocessing creates all expected output files."""
    preprocessor.preprocess_offline(preprocessing_config)

    # Check that all expected files exist
    expected_files = [
        "test_promoters_uint8_distinct_byte-level_train.bin",
        "test_promoters_uint8_distinct_byte-level_train.idx",
    ]

    for filename in expected_files:
        file_path = Path(preprocessing_config.output_dir) / filename
        assert file_path.exists(), f"Expected file {file_path} was not created"
        assert file_path.stat().st_size > 0, f"File {file_path} is empty"
