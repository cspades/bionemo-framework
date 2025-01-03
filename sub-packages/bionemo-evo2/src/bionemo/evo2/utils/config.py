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


class StipedHyena2BlendedDatasetConfig(BaseModel):
    """Pydantic model class that specifies indexed datasets, dataset weights, and datasplits assignments for training."""
    dataset_prefix: None | str = None
    dataset_weight: None | float = None
    dataset_split: Literal["train", "validation", "test"]


class StripedHyena2TaxonomyLineage(BaseModel):
    """Pydantic model class that defines the source lineage of a DNA sequence."""
    kingdom: None | str = None
    phylum: None | str = None
    clazz: None | str = None
    order: None | str = None
    family: None | str = None
    genus: None | str = None
    species: None | str = None


class StipedHyena2PreprocessingConfig(BaseModel):
    """Pydantic model class specifying the configuration schema for a preprocessed IndexedDataset (.bin, .idx)."""
    # Paths
    datapaths: list[Path] = []
    output_dir: None | Path = None
    output_prefix: None | str = None
    # Random Datasplit
    train_split: float = 0.7
    valid_split: float = 0.2
    test_split: float = 0.1
    # Overwrite existing binaries. Otherwise, skip already preprocessed datasets.
    overwrite: bool = False
    # Raw Preprocessing Transforms
    embed_reverse_complement: bool = False
    random_reverse_complement: float = 0.0
    random_lineage_dropout: float = 0.0
    include_sequence_id: bool = False
    transcribe: None | Literal["transcribe", "back_transcribe"] = None
    force_uppercase: bool = False
    indexed_dataset_dtype: str = "uint8"
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
    # Compute
    # NOTE: If preprocessing short individual sequences (< 1000 bp), do NOT use multiprocessing
    # (workers > 1) because sequence-level parallel IPC will dominate the preprocessing time!
    workers: int = 1
    preproc_concurrency: int = 100000
    chunksize: int = 1
    # Filters
    drop_empty_sequences: bool = False
    nnn_filter: bool = False
    # RNG
    seed: None | int = None
    # StipedHyena2 Taxonomic Lineage Tags
    # SeqID Sub-String Indexing: "ABC" will have taxonomy data from "A".
    taxonomy_data: dict[str, StripedHyena2TaxonomyLineage] = {}