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
from typing import Literal

from pydantic import BaseModel


class Evo2PreprocessingConfig(BaseModel):
    """Class specifying the configuration schema for Evo2 data preprocessing."""

    # Paths
    datapaths: list[Path] = []
    output_dir: None | Path = None
    output_prefix: None | str = None
    # Datasplit
    train_split: float = 0.7
    valid_split: float = 0.2
    test_split: float = 0.1
    # Evo Taxonomy
    taxonomy_path: None | Path = None
    # Raw Preprocessing Transforms
    gzip_data: bool = False
    embed_reverse_complement: bool = False
    random_reverse_complement: bool = False
    subsequence_length: None | int = None
    include_sequence_id: bool = False
    transcribe: None | Literal["transcribe", "back_transcribe"] = None
    force_uppercase: bool = False
    # Tokenizer
    tokenizer_type: Literal[
        "Byte-Level",
        "HuggingFace",
        "SentencePiece",
        "Regex",
        "Megatron",
        "Tiktoken",
    ] = "Byte-Level"
    vocab_file: None | Path = None
    vocab_size: None | int = 512
    merges_file: None | Path = None
    # Either a named pretrained tokenizer model, or a path to a SentencePiece tokenizer.
    pretrained_tokenizer_model: None | str = None
    special_tokens: None | dict[str, str] = {}
    fast_hf_tokenizer: bool = False
    append_eod: bool = False
    enforce_sample_length: None | int = None
    ftfy: bool = False
    indexed_dataset_dtype: str = "uint8"
    # Compute
    workers: int = 1
    preproc_concurrency: int = 10000
    # Filters
    drop_empty_sequences: bool = False
    nnn_filter: bool = False
    # RNG
    seed: None | int = None
