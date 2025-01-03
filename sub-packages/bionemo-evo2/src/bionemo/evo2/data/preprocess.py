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
"""

import argparse
import gzip
import multiprocessing as mp
import os
import random
import time
from contextlib import contextmanager
from pathlib import Path
from threading import Semaphore
from typing import Optional

import numpy as np
import pandas as pd
import torch
import yaml
from megatron.core.datasets.indexed_dataset import IndexedDatasetBuilder
from nemo.utils import logging

from bionemo.evo2.data.tokenizer import StripedHyena2Tokenizer
from bionemo.evo2.utils.config import StipedHyena2PreprocessingConfig, StripedHyena2TaxonomyLineage
from bionemo.noodles import back_transcribe_sequence, complement_sequence, reverse_sequence, transcribe_sequence
from bionemo.noodles.nvfaidx import NvFaidx


class StipedHyena2Preprocessor:
    """Data preprocessing class for Evo2."""

    PROMPT_SPACER_LENGTH = 131_072
    BIN = ".bin"
    IDX = ".idx"
    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    def __init__(self, params: StipedHyena2PreprocessingConfig | None = None):
        """Initialize StipedHyena2Preprocessor."""
        self.tokenizer: StripedHyena2Tokenizer = StripedHyena2Tokenizer(params)

    @staticmethod
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

    @staticmethod
    def _get_output_filename(config: StipedHyena2PreprocessingConfig, ext: str = None, split: str = None, temp: bool = False) -> Path:
        # Get output directory. Defaults to CWD.
        output_dir = config.output_dir
        if output_dir is None:
            output_dir = Path.cwd()
        # Pickup output file prefix.
        config_prefix = "{}_{}".format(
            config.output_prefix, config.tokenizer_type.lower().replace(" ", "")
        )
        output_filepath = Path(output_dir) / (config_prefix + (f"_{split}" if split is not None else "") + (ext if ext is not None else "") + (".tmp" if temp else ""))
        return output_filepath

    @staticmethod
    def _subsequence_generator(sequence: str, subsequence_length: int | None = None, offset: int | None = None):
        subsequence_length = subsequence_length if subsequence_length is not None else len(sequence)
        step_size = offset if offset is not None else subsequence_length
        for i in range(0, len(sequence), step_size):
            yield sequence[i : i + subsequence_length]

    @staticmethod
    def _random_reverse_complement(seq: str, prob: float = 0.0, seed: int = None):
        with StipedHyena2Preprocessor.preprocessing_context_manager(
            seed if seed is not None else None
        ):
            if random.random() < prob:
                return complement_sequence(reverse_sequence(seq))
            else:
                return seq

    @staticmethod
    def _reverse_complement_expansion(seq: str):
        return [seq, complement_sequence(reverse_sequence(seq))]
    
    @staticmethod
    def _train_val_test_split(train_weight: float, val_weight: float, test_weight: float, seed: int = None):
        with StipedHyena2Preprocessor.preprocessing_context_manager(
            seed if seed is not None else None
        ):
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
    def _construct_taxonomy_token(lineage: StripedHyena2TaxonomyLineage, dropout: float = 0.0, seed: int = None) -> Optional[str]:
        """Construct a special Taxonomy token for natural language prompting of DNA generation models."""
        # If dropout > 0, randomly drop out segments of the lineage for training on incomplete lineages.
        with StipedHyena2Preprocessor.preprocessing_context_manager(
            seed if seed is not None else None
        ):
            return "|d__{};p__{};c__{};o__{};f__{};g__{};s__{}|".format(
                lineage.kingdom if random.random() >= dropout else None,
                lineage.phylum if random.random() >= dropout else None,
                lineage.clazz if random.random() >= dropout else None,
                lineage.order if random.random() >= dropout else None,
                lineage.family if random.random() >= dropout else None,
                lineage.genus if random.random() >= dropout else None,
                lineage.species if random.random() >= dropout else None,
            ) if lineage is not None else None

    def preprocess_data(self, filepath: str, seqid: str, seq: str, seq_idx: int, config: StipedHyena2PreprocessingConfig):
        """Preprocess Evo2 fasta datapaths."""

        # Timing.
        start = time.time()
        # Retrieve taxonomy lineage string if SeqID has associated taxonomy data.
        # Note: Better implemented as a suffix tree substring dictionary, but convenient
        # for identifying a large amount of sequences with identical lineages.
        # Slow for extremely large dictionaries of (SeqID Substr, Taxonomy) pairs.
        lineage = None
        for id, tax in config.taxonomy_data.items():
            # Taxonomy ID is a substring of Seq ID.
            if id in seqid:
                lineage = tax
                break

        # Preprocess data.
        preproc_data = []
        with self.preprocessing_context_manager(
            config.seed + hash(filepath) + seq_idx if config.seed is not None else None
        ):
            # Randomly reverse complement the sequence.
            seq = self._random_reverse_complement(seq, prob=config.random_reverse_complement)
            seqs_to_parse = self._reverse_complement_expansion(seq) if config.embed_reverse_complement else [seq]
            for seq in seqs_to_parse:
                # Sequence Modifiers
                if config.force_uppercase:
                    seq = seq.upper()
                if config.transcribe == "transcribe":
                    seq = transcribe_sequence(seq)
                elif config.transcribe == "back_transcribe":
                    seq = back_transcribe_sequence(seq)
                if config.drop_empty_sequences and len(seq) == 0:
                    continue
                if config.nnn_filter and "NNN" in seq.upper():
                    continue

                # Construct taxonomy token with random dropout on the lineage categories per sequence.
                taxonomy_token = self._construct_taxonomy_token(lineage, dropout=config.random_lineage_dropout)
                
                # Inject taxonomy lineage tokens every PROMPT_SPACER_LENGTH tokens in the sequence.
                # If the taxonomy lineage token is not provided, then just take the original sequence.
                target_length = (
                    self.PROMPT_SPACER_LENGTH - len(taxonomy_token)
                    if taxonomy_token is not None
                    else None
                )
                taxonomy_injected_sequence = [
                    taxonomy_token + str(subseq) if taxonomy_token is not None else str(subseq)
                    for subseq in self._subsequence_generator(seq, target_length, target_length)
                ]

                # Wrap and tokenize.
                preproc_data_record = {
                    "text": "".join(taxonomy_injected_sequence),
                }
                if config.include_sequence_id:
                    preproc_data_record["id"] = f"{seqid}"
                preproc_data_record["tokens"] = self.tokenizer.tokenize(
                    preproc_data_record["text"],
                    use_ftfy=config.ftfy,
                    enforce_sample_length=config.enforce_sample_length,
                    append_eod=config.append_eod,
                    drop_empty_sequences=config.drop_empty_sequences,
                )
                preproc_data.append(preproc_data_record)
        end = time.time()
        return preproc_data, end - start

    def preprocess_data_task(self, file_sequence_config):
        """Wrapper function to unpack args for preprocess_data."""
        return self.preprocess_data(*file_sequence_config)
    
    @staticmethod
    def _yield_sequences_from_files(config: StipedHyena2PreprocessingConfig, semaphore: Semaphore):
        """Iterator over sequences within multiple input documents. Arguments for multiprocessing tasks.

        Utilized to limit the amount of sequences streamed into memory.
        """

        def yielder(fname, semaphore):
            # Read FASTA.
            index = NvFaidx(fname)
            for i, (seqid, sequence) in enumerate(index.items()):
                semaphore.acquire()
                # Yield filename and sequence within fasta.
                yield str(fname), seqid, sequence, i, config

        for fname in config.datapaths:
            semaphore.acquire()
            yield from yielder(fname, semaphore)

    def preprocess_generator(self, preproc_config: StipedHyena2PreprocessingConfig):
        """Main function to preprocess data for Evo2."""

        # Instantiate multiprocessing pool. Use semaphore to limit the amount of sequences to read into memory.
        semaphore = Semaphore(preproc_config.preproc_concurrency + preproc_config.workers)
        if preproc_config.workers > 1:
            pool = mp.Pool(preproc_config.workers)
            # Ordered imap for downstream seeded splitting.
            preproc_tasks = pool.imap(
                self.preprocess_data_task,
                self._yield_sequences_from_files(
                    preproc_config, semaphore
                ),
                chunksize=preproc_config.chunksize,
            )
        else:
            preproc_tasks = (
                self.preprocess_data_task(x)
                for x in self._yield_sequences_from_files(
                    preproc_config, semaphore
                )
            )

        # Preprocess data and split results into train, test, and split.
        with self.preprocessing_context_manager(preproc_config.seed if preproc_config.seed is not None else None):
            for result, elapsed_time in preproc_tasks:
                # Release semaphore for the task associated with the result.
                semaphore.release()
                # Randomly assign all sequences to train, validation, or test.
                split = self._train_val_test_split(preproc_config.train_split, preproc_config.valid_split, preproc_config.test_split)
                for sequence in result:
                    sequence["split"] = split
                    yield sequence, elapsed_time

    def preprocess_offline(self, preproc_config: StipedHyena2PreprocessingConfig):
        """Offline data preprocessing script for Evo2."""

        # Validate if binaries have already been produced for the given config and overwrite is set to False.
        if any(self._get_output_filename(preproc_config, ext, split).is_file() for ext, split in zip([self.BIN, self.IDX], [self.TRAIN, self.VAL, self.TEST])):
            if not preproc_config.overwrite:
                # Skip this dataset!
                logging.info(f"Skipped overwriting (overwrite: False) existing preprocessed data: {preproc_config.output_prefix}")
                return
            else:
                logging.info(f"Overwriting (overwrite: True) existing preprocessed data: {preproc_config.output_prefix}")

        # Instantiate indexed data builders.
        dataset_dtype = getattr(np, preproc_config.indexed_dataset_dtype)
        temp_train_bin = self._get_output_filename(preproc_config, self.BIN, self.TRAIN, temp=True)
        temp_val_bin = self._get_output_filename(preproc_config, self.BIN, self.VAL, temp=True)
        temp_test_bin = self._get_output_filename(preproc_config, self.BIN, self.TEST, temp=True)
        train_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(temp_train_bin), dtype=dataset_dtype)
        val_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(temp_val_bin), dtype=dataset_dtype)
        test_builder: IndexedDatasetBuilder = IndexedDatasetBuilder(bin_path=str(temp_test_bin), dtype=dataset_dtype)
        logging.info(f"Created temporary binary datasets: {temp_train_bin} {temp_val_bin} {temp_test_bin}")

        # Preprocess data and split results into train, validation, or test.
        avg_preproc_time = 0.0
        avg_index_time = 0.0
        count = 0
        for sequence, elapsed_time in self.preprocess_generator(preproc_config):
            index_start_time = time.time()
            if sequence["split"] == "train":
                train_builder.add_item(torch.Tensor(sequence["tokens"]))
                train_builder.end_document()
            elif sequence["split"] == "val":
                val_builder.add_item(torch.Tensor(sequence["tokens"]))
                val_builder.end_document()
            elif sequence["split"] == "test":
                test_builder.add_item(torch.Tensor(sequence["tokens"]))
                test_builder.end_document()
            index_end_time = time.time()
            # Update average preprocessing and indexing time.
            avg_preproc_time = (avg_preproc_time * count + elapsed_time) / (count + 1)
            avg_index_time = (avg_index_time * count + index_end_time - index_start_time) / (count + 1)
            count += 1
        
        # Report timing.
        logging.info(f"Average preprocessing time per sequence: {avg_preproc_time}")
        logging.info(f"Average indexing time per sequence: {avg_index_time}")
        logging.info(f"Number of sequences processed: {count}")

        # Write preprocessed index data to disk. Rename temporary binaries to denote preprocessing completion.
        train_builder.finalize(idx_path=str(self._get_output_filename(preproc_config, self.IDX, self.TRAIN)))
        val_builder.finalize(idx_path=str(self._get_output_filename(preproc_config, self.IDX, self.VAL)))
        test_builder.finalize(idx_path=str(self._get_output_filename(preproc_config, self.IDX, self.TEST)))
        os.rename(temp_train_bin, self._get_output_filename(preproc_config, self.BIN, self.TRAIN))
        os.rename(temp_val_bin, self._get_output_filename(preproc_config, self.BIN, self.VAL))
        os.rename(temp_test_bin, self._get_output_filename(preproc_config, self.BIN, self.TEST))


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
        hyena2_preproc_config_batch = yaml.safe_load(yaml_fs)
    for config in hyena2_preproc_config_batch:
        start = time.time()
        # Convert into StipedHyena2PreprocessingConfig.
        hyena2_preproc_config = StipedHyena2PreprocessingConfig(**config)
        # Instantiate StipedHyena2Preprocessor.
        hyena2_preprocessor = StipedHyena2Preprocessor(hyena2_preproc_config)
        # Preprocess data specified in config.
        hyena2_preprocessor.preprocess_offline(hyena2_preproc_config)
        end = time.time()
        logging.info(f"Finished preprocessing {hyena2_preproc_config.output_prefix} ({hyena2_preproc_config.datapaths}) in {end - start:.3f} seconds with {hyena2_preproc_config.workers} workers.")
