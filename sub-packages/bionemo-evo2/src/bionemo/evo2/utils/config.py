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


from collections import defaultdict
from pathlib import Path
from typing import Literal, Optional

import yaml
from pydantic import BaseModel, model_validator


class Evo2BlendedDatasetConfig(BaseModel):
    """Configuration for blended dataset specifications.

    Validates and constructs dataset paths, weights and splits configuration.
    Ensures dataset paths exist and are properly resolved relative to base data path.

    Attributes:
        dataset_path: Base directory path for datasets. Used to resolve relative dataset prefixes.
        dataset_prefix: Path prefix for dataset files. Can be absolute or relative to dataset_path.
        dataset_weight: Weight factor for this dataset during blending (0-1).
        dataset_split: Dataset partition - 'train', 'validation' or 'test'.

    Raises:
        ValueError: If dataset path doesn't exist or prefix can't be resolved.
    """

    dataset_path: str | None = None
    dataset_prefix: str
    dataset_weight: float
    dataset_split: Literal["train", "validation", "test"]

    @model_validator(mode="before")
    @classmethod
    def validate_dataset_prefix(cls, values: dict) -> dict:
        """Ensure dataset_prefix paths exist and are properly resolved or are relative to base dataset_path if provided."""
        dataset_path = Path(values.get("dataset_path")) if values.get("dataset_path") else None
        prefix = Path(values.get("dataset_prefix"))

        if not prefix.is_absolute():
            if dataset_path:
                prefix = dataset_path / prefix
            else:
                prefix = Path(prefix).resolve()
        parent = prefix.parent
        stem = prefix.stem
        if not parent.exists():
            raise ValueError(f"dataset_prefix parent path does not exist: {parent}")
        matching_files = list(parent.glob(f"{stem}.*"))
        if not matching_files:
            raise ValueError(f"dataset_prefix file does not exist: {prefix}")
        values["dataset_prefix"] = str(prefix)
        return values


class Evo2TaxonomyLineage(BaseModel):
    """Pydantic model class that defines the source lineage of a DNA sequence."""

    kingdom: None | str = None
    phylum: None | str = None
    clazz: None | str = None
    order: None | str = None
    family: None | str = None
    genus: None | str = None
    species: None | str = None


class Evo2PreprocessingConfig(BaseModel):
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
    transcribe: None | Literal["transcribe", "back_transcribe"] = None
    force_uppercase: bool = False
    indexed_dataset_dtype: str = "uint8"
    # Tokenization Transforms
    append_eod: bool = True
    enforce_sample_length: None | int = None
    ftfy: bool = False
    # NeMo Tokenizer Configuration
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
    tokenizer_model_name: None | str = None
    pretrained_tokenizer_model: None | str = None
    special_tokens: None | dict[str, str] = {}
    fast_hf_tokenizer: bool = False
    # Compute Configuration
    # NOTE: If preprocessing a large amount of short individual sequences (< 1000 bp), do NOT use
    # multiprocessing (workers > 1) because sequence-level parallel IPC will dominate the preprocessing time!
    workers: int = 1
    preproc_concurrency: int = 100000
    chunksize: int = 1
    # Filters
    drop_empty_sequences: bool = False
    nnn_filter: bool = False
    # RNG
    seed: None | int = None
    # Evo2 Taxonomic Lineage Tags
    # SeqID Sub-String Indexing: "ABC" will have taxonomy data from "A".
    taxonomy_data: dict[str, Evo2TaxonomyLineage] = {}
    # Periodicity of injecting phylogenetic lineage tags in the sequence prior to tokenization.
    prompt_spacer_length: int = 131072


def parse_dataset_config(dataset_config_path: str, dataset_path: Optional[str] = None):
    """Parse the blended training datasplit configuration and renormalize data split weights for training Hyena.

    Args:
        dataset_config_path (str): Path to the dataset configuration YAML file.
        dataset_path (str): Path to the dataset directory. Defaults to None.

    Returns:
        defaultdict: A dictionary where keys are dataset splits and values are lists containing the normalized weight
                     and dataset prefix for each split.
    """
    blended_dataset_config = defaultdict(list)
    weight_sums = defaultdict(float)
    with open(dataset_config_path, "r") as config_file:
        dataset_config_batch = yaml.safe_load(config_file)
        for dataset_config in dataset_config_batch:
            # Validate.
            config_model = Evo2BlendedDatasetConfig(dataset_path=dataset_path, **dataset_config)
            # Integrate the weights for renormalization.
            weight_sums[config_model.dataset_split] += abs(config_model.dataset_weight)
        for dataset_config in dataset_config_batch:
            # Validate.
            config_model = Evo2BlendedDatasetConfig(dataset_path=dataset_path, **dataset_config)
            # Add indexed dataset to split and associate with blended training weight.
            blended_dataset_config[config_model.dataset_split].extend(
                [config_model.dataset_weight / weight_sums[config_model.dataset_split], config_model.dataset_prefix]
            )
    return blended_dataset_config
