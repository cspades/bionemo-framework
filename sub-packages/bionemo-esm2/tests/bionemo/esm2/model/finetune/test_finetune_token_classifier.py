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


import pandas as pd
import pytest
import torch

from bionemo.core.data.load import load
from bionemo.esm2.data import tokenizer
from bionemo.esm2.model.finetune.dataset import InMemoryCSVDataset, InMemoryPerTokenValueDataset
from bionemo.esm2.model.finetune.finetune_token_classifier import (
    ESM2FineTuneTokenConfig,
    ESM2FineTuneTokenModel,
    MegatronConvNetHead,
)
from bionemo.llm.data.collate import MLM_LOSS_IGNORE_INDEX
from bionemo.llm.data.label2id_tokenizer import Label2IDTokenizer
from bionemo.testing import megatron_parallel_state_utils


# To download a 8M internally pre-trained ESM2 model
pretrain_ckpt_path = load("esm2/nv_8m:2.0")


def data_to_csv(data, tmp_path):
    """Create a mock protein dataset."""
    csv_file = tmp_path / "protein_dataset.csv"
    # Create a DataFrame
    df = pd.DataFrame(data, columns=["sequences", "labels"])

    # Save the DataFrame to a CSV file
    df.to_csv(csv_file, index=False)
    return csv_file


@pytest.fixture
def dataset(dummy_data_per_token_classification_ft, tmp_path):
    return InMemoryPerTokenValueDataset(data_to_csv(dummy_data_per_token_classification_ft, tmp_path))


@pytest.fixture
def config():
    return ESM2FineTuneTokenConfig(encoder_frozen=True, cnn_dropout=0.1, cnn_hidden_dim=32, cnn_num_classes=5)


@pytest.fixture
def finetune_token_model(config):
    with megatron_parallel_state_utils.distributed_model_parallel_state():
        model = config.configure_model(tokenizer.get_tokenizer())
        yield model


def test_dataset_getitem(dataset, dummy_data_per_token_classification_ft):
    assert isinstance(dataset, InMemoryCSVDataset)
    assert isinstance(dataset.label_tokenizer, Label2IDTokenizer)
    assert dataset.label_cls_eos_id == MLM_LOSS_IGNORE_INDEX

    assert len(dataset) == len(dummy_data_per_token_classification_ft)
    sample = dataset[0]
    assert isinstance(sample, dict)
    assert "text" in sample
    assert "labels" in sample
    assert "loss_mask" in sample
    assert isinstance(sample["text"], torch.Tensor)
    assert isinstance(sample["labels"], torch.Tensor)
    assert sample["labels"].dtype == torch.int64
    assert isinstance(sample["loss_mask"], torch.Tensor)


def test_transofrm_label(dataset, dummy_data_per_token_classification_ft):
    pre_transfrom = dummy_data_per_token_classification_ft[0][1]
    label_ids = torch.tensor(dataset.label_tokenizer.text_to_ids(pre_transfrom))
    cls_eos = torch.tensor([dataset.label_cls_eos_id])
    post_transform = torch.cat((cls_eos, label_ids, cls_eos))

    assert torch.equal(dataset.transform_label(pre_transfrom), post_transform)


def test_ft_config(config):
    assert config.initial_ckpt_skip_keys_with_these_prefixes == ["classification_head"]
    assert config.encoder_frozen
    assert config.cnn_dropout == 0.1
    assert config.cnn_hidden_dim == 32
    assert config.cnn_num_classes == 5


def test_ft_model_initialized(finetune_token_model):
    assert isinstance(finetune_token_model, ESM2FineTuneTokenModel)
    assert isinstance(finetune_token_model.classification_head, MegatronConvNetHead)
    assert finetune_token_model.post_process
    assert not finetune_token_model.include_hiddens_finetuning
