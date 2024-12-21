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


"""Module containing data preprocessing and splitting functions for Evo2 in BioNeMo.

It can also be utilized as a script to dump pre-processed data to JSON.

TODO(@cye): Add commentary and config interface.
"""

import argparse
import gzip
import multiprocessing as mp
import random
from contextlib import contextmanager
from pathlib import Path
from threading import Semaphore

import numpy as np
import pandas as pd
import torch
import yaml
from Bio import Seq, SeqIO
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from nemo.utils import logging

from bionemo.evo2.data.resources.phyla_kingdom_map import PHYLA_TO_KINGDOM
from bionemo.evo2.data.tokenizer import Evo2Tokenizer
from bionemo.evo2.utils.config import Evo2PreprocessingConfig


@contextmanager
def preprocessing_context_manager(seed: int | None = None):
    """Context manager for Evo2 preprocessing RNG."""
    # Track current state.
    current_state = random.getstate()
    try:
        # Set random seed.
        random.seed(seed)
        yield seed
    finally:
        # Restore random state.
        random.setstate(current_state)


class Evo2Preprocessor:
    """Data preprocessing class for Evo2."""

    VBAR = "|"
    PROMPT_SPACER_LENGTH = 131_072

    def __init__(self, params: Evo2PreprocessingConfig | None = None):
        """Initialize Evo2Preprocessor."""
        self.params: Evo2PreprocessingConfig = params if params is not None else Evo2PreprocessingConfig()
        self.tokenizer: Evo2Tokenizer = Evo2Tokenizer(self.params)
        self.id_to_taxonomy: dict | None = (
            self._load_evo_taxonomy(self.params.taxonomy_path) if self.params.taxonomy_path is not None else None
        )

    @staticmethod
    def _subsequence_generator(sequence: Seq.Seq, subsequence_length: int | None = None, offset: int | None = None):
        subsequence_length = subsequence_length if isinstance(subsequence_length, int) else len(sequence)
        step_size = offset if isinstance(offset, int) else subsequence_length
        for i in range(0, len(sequence), step_size):
            yield sequence[i : i + subsequence_length]

    @staticmethod
    def _random_reverse_complement(seq: Seq.Seq, prob: float = 0.5):
        if random.random() < prob:
            return seq.reverse_complement()
        else:
            return seq

    @staticmethod
    def _reverse_complement_expansion(seq: Seq.Seq):
        return [seq, seq.reverse_complement()]
    
    @staticmethod
    def _train_val_test_split(train_weight: float, val_weight: float, test_weight: float):
        # Generate random number.
        roll = random.random()
        # Rectify and normalize split ratios.
        total_weight = abs(train_weight) + abs(val_weight) + abs(test_weight)
        if total_weight <= 0:
            raise ValueError("Train-validation-test split proportions cannot be zero.")
        train_split = abs(train_weight) / total_weight
        test_split = abs(test_weight) / total_weight
        split = "train"
        if roll > train_split:
            if roll < 1 - test_split:
                split = "val"
            else:
                split = "test"
        return split

    @staticmethod
    def _get_evo_seq_id(filename: str):
        """TODO(@cye) Consider deprecating the Taxonomy resources from Arc in favor of an explicit SeqID -> Taxonomy mapping via config."""
        try:
            return ".".join(filename.split("/")[-1].split(".")[:-1])
        except Exception:
            return None

    @staticmethod
    def _get_evo_phyla_from_lineage_string(lineage_str: str):
        """TODO(@cye) Consider deprecating the Taxonomy resources from Arc in favor of an explicit SeqID -> Taxonomy mapping via config."""
        try:
            return lineage_str.split(";")[1].split("_")[-1]
        except Exception:
            return None

    @staticmethod
    def _load_evo_taxonomy(fname):
        """TODO(@cye) Consider deprecating the Taxonomy resources from Arc in favor of an explicit SeqID -> Taxonomy mapping via config."""
        df = pd.read_csv(fname, sep="\t")
        id_to_taxonomy = {}
        for _, row in df.iterrows():
            lineage_string = (
                f'd__{row["kingdom"]};'
                f'p__{row["phylum"]};'
                f'c__{row["class"]};'
                f'o__{row["order"]};'
                f'f__{row["family"]};'
                f'g__{row["genus"]};'
                f's__{row["species"]}'
            )
            id_to_taxonomy[row["genome_id"]] = lineage_string
        return id_to_taxonomy

    @staticmethod
    def _yield_sequences_from_files(fnames: list, semaphore: Semaphore, gzip_data: bool = False):
        """Iterator over sequences within multiple input documents. Arguments for multiprocessing tasks.

        Utilized to limit the amount of sequences streamed into memory.
        TODO(@cye): Just do the fasta parsing ourselves if there's no weird formats.
        """

        def yielder(fname, semaphore):
            # Open file.
            with gzip.open(fname, "rt") if gzip_data else open(fname, "r") as f:
                for record in SeqIO.parse(f, "fasta"):
                    semaphore.acquire()
                    # Yield filename and record within fasta.
                    yield str(fname), record

        for fname in fnames:
            semaphore.acquire()
            yield from yielder(fname, semaphore)

    def configure(self, params: Evo2PreprocessingConfig | None = None):
        """Configure a new Evo2PreprocessingConfig for Evo2Preprocessor."""
        self.params = params if params is not None else Evo2PreprocessingConfig()
        self.id_to_taxonomy = (
            self._load_evo_taxonomy(self.params.taxonomy_path) if self.params.taxonomy_path is not None else None
        )

    def preprocess_data(self, filepath: str, record) -> list[dict]:
        """Preprocess Evo2 fasta datapaths."""
        # Retrieve EVO taxonomy metadata if id_to_taxonomy is provided.
        lineage_string = (
            self.id_to_taxonomy.get(self._get_evo_seq_id(str(filepath)), None)
            if isinstance(self.id_to_taxonomy, dict)
            else None
        )
        phyla = self._get_evo_phyla_from_lineage_string(lineage_string) if lineage_string is not None else None
        kingdom = PHYLA_TO_KINGDOM.get(phyla, None) if phyla is not None else None
        if isinstance(self.id_to_taxonomy, dict) and (lineage_string is None or kingdom is None):
            logging.info(f"No taxonomy lineage metadata detected for {filepath}. Skipping datafile...")
            return []

        # Preprocess data.
        preproc_data = []
        with preprocessing_context_manager(
            self.params.seed + hash(filepath) if self.params.seed is not None else None
        ):
            seq = record.seq
            # Randomly reverse complement the sequence.
            seq = self._random_reverse_complement(seq, prob=0.5) if self.params.random_reverse_complement else seq
            seqs_to_parse = self._reverse_complement_expansion(seq) if self.params.embed_reverse_complement else [seq]
            for seq in seqs_to_parse:
                if self.params.force_uppercase:
                    seq = seq.upper()
                if self.params.transcribe == "transcribe":
                    seq = seq.transcribe()
                elif self.params.transcribe == "back_transcribe":
                    seq = seq.back_transcribe()
                if self.params.drop_empty_sequences and len(seq) == 0:
                    continue
                if self.params.nnn_filter and "NNN" in seq.upper():
                    continue
                taxonomy_token = (
                    self.VBAR + lineage_string.upper() + self.VBAR if isinstance(lineage_string, str) else None
                )
                target_length = (
                    # Full sequence length minus bandwidth for the special Taxonomy token.
                    self.PROMPT_SPACER_LENGTH - len(taxonomy_token)
                    if isinstance(taxonomy_token, str)
                    # Chunk into subsequences. If None, then default to sequence length.
                    else self.params.subsequence_length
                )
                for i, subseq in enumerate(self._subsequence_generator(seq, target_length, target_length)):
                    preproc_data_record = {
                        "text": taxonomy_token + str(subseq) if taxonomy_token is not None else str(subseq),
                    }
                    if self.params.include_sequence_id:
                        preproc_data_record["id"] = f"{record.id}_{i}"
                    # Tokenize the sequence.
                    preproc_data_record["tokens"] = self.tokenizer.tokenize(
                        preproc_data_record["text"],
                        use_ftfy=self.params.ftfy,
                        enforce_sample_length=self.params.enforce_sample_length,
                        append_eod=self.params.append_eod,
                        drop_empty_sequences=self.params.drop_empty_sequences,
                    )
                    preproc_data.append(preproc_data_record)
        return preproc_data

    def preprocess_data_task(self, file_record):
        """Wrapper function to unpack args for preprocess_data."""
        return self.preprocess_data(*file_record)

    def preprocess_generator(self, preproc_config: Evo2PreprocessingConfig):
        """Main function to preprocess data for Evo2."""
        # Configure preprocessor.
        self.configure(preproc_config)

        # Instantiate multiprocessing pool.
        semaphore = Semaphore(preproc_config.preproc_concurrency + preproc_config.workers)
        if preproc_config.workers > 1:
            pool = mp.Pool(preproc_config.workers)
            # Ordered imap for downstream seeded splitting.
            preproc_tasks = pool.imap(
                evo2_preprocessor.preprocess_data_task,
                Evo2Preprocessor._yield_sequences_from_files(
                    preproc_config.datapaths, semaphore, preproc_config.gzip_data
                ),
                chunksize=25,
            )
        else:
            preproc_tasks = (
                evo2_preprocessor.preprocess_data_task(x)
                for x in Evo2Preprocessor._yield_sequences_from_files(
                    preproc_config.datapaths, semaphore, preproc_config.gzip_data
                )
            )

        # Preprocess data and split results into train, test, and split.
        with preprocessing_context_manager(preproc_config.seed if preproc_config.seed is not None else None):
            for result in preproc_tasks:
                # Release semaphore for the task associated with the result.
                semaphore.release()
                # Randomly assign all sequences from this document to train, val, or test.
                split = Evo2Preprocessor._train_val_test_split(preproc_config.train_split, preproc_config.valid_split, preproc_config.test_split)
                for sequence in result:
                    sequence["split"] = split
                    yield sequence

    def preprocess_offline(self, preproc_config: Evo2PreprocessingConfig):
        """Offline data preprocessing script for Evo2."""
        # Process output directory.
        output_dir = preproc_config.output_dir
        if output_dir is None:
            output_dir = Path.cwd()
        # Build train, validation, and test datasplits.
        BIN = ".bin"
        TRAIN_SUFFIX = "_train"
        VAL_SUFFIX = "_val"
        TEST_SUFFIX = "_test"
        config_prefix = "{}_{}".format(
            preproc_config.output_prefix, preproc_config.tokenizer_type.lower().replace(" ", "")
        )
        train_bin_path = Path(output_dir) / (config_prefix + TRAIN_SUFFIX + BIN)
        val_bin_path = Path(output_dir) / (config_prefix + VAL_SUFFIX + BIN)
        test_bin_path = Path(output_dir) / (config_prefix + TEST_SUFFIX + BIN)
        dataset_dtype = getattr(np, preproc_config.indexed_dataset_dtype)
        train_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(train_bin_path), dtype=dataset_dtype)
        val_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(val_bin_path), dtype=dataset_dtype)
        test_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(test_bin_path), dtype=dataset_dtype)

        # Preprocess data and split results into train, validation, or test.
        for sequence in self.preprocess_generator(preproc_config):
            if sequence["split"] == "train":
                train_builder.add_item(torch.Tensor(sequence["tokens"]))
            elif sequence["split"] == "val":
                val_builder.add_item(torch.Tensor(sequence["tokens"]))
            elif sequence["split"] == "test":
                test_builder.add_item(torch.Tensor(sequence["tokens"]))
        # IMPORTANT TODO(@cye): Split documents by filename instead of all datasets
        # into one document, to check that BlendedDataset weighting make sense.
        train_builder.end_document()
        val_builder.end_document()
        test_builder.end_document()

        # Write preprocessed index sdata to disk.
        IDX = ".idx"
        train_idx_path = Path(output_dir) / (config_prefix + TRAIN_SUFFIX + IDX)
        val_idx_path = Path(output_dir) / (config_prefix + VAL_SUFFIX + IDX)
        test_idx_path = Path(output_dir) / (config_prefix + TEST_SUFFIX + IDX)
        train_builder.finalize(idx_path=str(train_idx_path))
        val_builder.finalize(idx_path=str(val_idx_path))
        test_builder.finalize(idx_path=str(test_idx_path))


def parse_args():
    """Parse arguments for Evo2 preprocessing."""
    parser = argparse.ArgumentParser(description="Preprocess datapaths for Evo2.")
    parser.add_argument("-c", "--config", type=str, required=True, help="Path to Evo2 data preprocessing config JSON.")
    return parser.parse_args()


if __name__ == "__main__":
    # Parse arguments.
    args = parse_args()
    # Read config YAML.
    with open(args.config, "r") as yaml_fs:
        evo2_preproc_config_batch = yaml.safe_load(yaml_fs)
    # Instantiate Evo2Preprocessor.
    evo2_preprocessor = Evo2Preprocessor()
    for config in evo2_preproc_config_batch:
        # Convert into Evo2PreprocessingConfig.
        evo2_preproc_config = Evo2PreprocessingConfig(**config)
        # Preprocess data specified in config.
        evo2_preprocessor.preprocess_offline(evo2_preproc_config)
