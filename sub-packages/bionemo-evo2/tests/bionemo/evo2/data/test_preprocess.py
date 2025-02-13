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

from bionemo.core.data.load import load
from bionemo.evo2.data.preprocess import Evo2Preprocessor
from bionemo.evo2.utils.config import Evo2PreprocessingConfig


@pytest.fixture
def sample_data_path() -> Path:
    # TODO(@dorotat) replace source with ngc when artefacts are published
    data_path = load("evo2/sample-data-raw:1.0", source="pbss")
    return data_path


@pytest.fixture
def output_prefix() -> str:
    return "test_promoters_uint8_distinct"


@pytest.fixture
def preprocessing_config(tmp_path: Path, output_prefix: str, sample_data_path: Path) -> Evo2PreprocessingConfig:
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
