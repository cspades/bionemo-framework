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

import torch
import functools
from pathlib import Path
from typing import List, Literal, Optional, Sequence
from bionemo.llm.data import masking, types

import numpy as np
from nemo.lightning.data import WrappedDataLoader
from torch.utils.data import Dataset
from nemo.lightning.pytorch.plugins import MegatronDataSampler
from nemo.utils import logging
from pytorch_lightning.utilities.types import EVAL_DATALOADERS, TRAIN_DATALOADERS
from tokenizers import Tokenizer

from bionemo.core.data.multi_epoch_dataset import MultiEpochDatasetResampler
from bionemo.core.utils import random_utils
from bionemo.geneformer.data.singlecell.dataset import SingleCellDataset
from bionemo.geneformer.tokenizer.gene_tokenizer import GeneTokenizer
from bionemo.llm.data import collate
from bionemo.llm.data.datamodule import MegatronDataModule
from bionemo.llm.utils.datamodule_utils import infer_num_samples


Mode = Literal["train", "validation", "test", "predict"]

__all__: Sequence[str] = ("SingleCellDataModule",)

class MockTokenizer:
    def __init__(self, vocab_size):
        # Does not need to be accurate as this is all being used for testing context length memory usage.
        self.mask_token_id = 2
        self.all_special_ids = [2]
        self.special_tokens= [2]
        self.vocab_size = vocab_size

class FakeFixedLengthDataset(Dataset):
    def __init__(self, seq_length, dataset_length, vocab_size):
        self.seq_length = seq_length
        self.dataset_length = dataset_length
        self.vocab_size = vocab_size
        self.tokenizer = MockTokenizer(vocab_size)

    def __len__(self):
        return self.dataset_length

    def __getitem__(self, _) -> types.BertSample:
        token_ids = np.random.randint(0, self.vocab_size, self.seq_length)

        with torch.no_grad(), torch.device("cpu"):
            masked_tokens, labels, loss_mask = masking.apply_bert_pretraining_mask(
            tokenized_sequence=torch.from_numpy(token_ids),
            random_seed=1337,
            mask_config=masking.BertMaskConfig(
                tokenizer=self.tokenizer,
                random_tokens=range(2,8),
                mask_prob=0.15,
                mask_token_prob=0.8,
                random_token_prob=0.0
            ),
        )
        return {
            "text": masked_tokens,
            "types": torch.zeros_like(masked_tokens, dtype=torch.int64),
            "attention_mask": torch.ones_like(masked_tokens, dtype=torch.int64),
            "labels": labels,
            "loss_mask": loss_mask,
            "is_random": torch.zeros_like(masked_tokens, dtype=torch.int64),
        }



class FixedLengthDataModule(MegatronDataModule):
    """ Slimmed down dataset and datamodule that always yields samples of seq_length tokens. 

    Intended use is to find limits of context length for various BERT configuraitons. 
    """
    def __init__(  # noqa: D107
        self,
        seq_length: int = 2048,
        micro_batch_size: int = 4,
        global_batch_size: int = 8,
        rampup_batch_size: Optional[List[int]] = None,
        seed: int = 42,
        num_workers: int = 10,  # TODO can this be automatically set?
        persistent_workers: bool = True,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.seq_length = seq_length
        self.seed = seed
        self.num_workers = num_workers
        self.persistent_workers = persistent_workers
        self.pin_memory = pin_memory

        self._train_ds = FakeFixedLengthDataset(self.seq_length, vocab_size=80000, dataset_length=21_000_000)
        self._validation_ds = FakeFixedLengthDataset(self.seq_length, vocab_size=80000, dataset_length=21_000_000)
        self._test_ds = FakeFixedLengthDataset(self.seq_length, vocab_size=80000, dataset_length=21_000_000)
        self._predict_ds = FakeFixedLengthDataset(self.seq_length, vocab_size=80000, dataset_length=21_000_000)
        self.tokenizer = self._train_ds.tokenizer

        # All datasets must have a sampler
        self.data_sampler = MegatronDataSampler(
            seq_len=self.seq_length,
            micro_batch_size=micro_batch_size,
            global_batch_size=global_batch_size,
            rampup_batch_size=rampup_batch_size,
        )


    def train_dataloader(self) -> TRAIN_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._train_ds, mode="train")

    def val_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._validation_ds, mode="validation")

    def test_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._test_ds, mode="test")

    def predict_dataloader(self) -> EVAL_DATALOADERS:  # noqa: D102
        return self._create_dataloader(self._predict_ds, mode="predict", drop_last=False)

    def _create_dataloader(self, dataset, mode: Mode, **kwargs) -> WrappedDataLoader:
        return WrappedDataLoader(
            mode=mode,
            dataset=dataset,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers,
            **kwargs,
        )
