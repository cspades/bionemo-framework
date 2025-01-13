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
from nemo.lightning import io

from bionemo.core.data.load import load
from bionemo.esm2.model.finetune.dataset import InMemoryPerTokenValueDataset, InMemorySingleValueDataset
from bionemo.esm2.model.finetune.finetune_regressor import ESM2FineTuneSeqConfig
from bionemo.esm2.model.finetune.finetune_token_classifier import ESM2FineTuneTokenConfig
from bionemo.esm2.scripts.finetune_esm2 import train_model as finetune
from bionemo.testing import megatron_parallel_state_utils
from bionemo.testing.callbacks import MetricTracker


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


@pytest.mark.needs_gpu
def test_esm2_finetune_token_classifier(
    tmp_path,
    dummy_data_per_token_classification_ft,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        simple_ft_checkpoint, simple_ft_metrics, trainer = finetune(
            train_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            valid_data_path=data_to_csv(dummy_data_per_token_classification_ft, tmp_path),
            experiment_name="finetune_new_head_token_classification",
            restore_from_checkpoint_path=str(pretrain_ckpt_path),
            num_steps=n_steps_train,
            num_nodes=1,
            devices=1,
            min_seq_length=None,
            max_seq_length=1024,
            result_dir=tmp_path / "finetune",
            limit_val_batches=2,
            val_check_interval=n_steps_train // 2,
            log_every_n_steps=n_steps_train // 2,
            num_dataset_workers=10,
            lr=1e-5,
            micro_batch_size=4,
            accumulate_grad_batches=1,
            resume_if_exists=False,
            precision="bf16-mixed",
            dataset_class=InMemoryPerTokenValueDataset,
            config_class=ESM2FineTuneTokenConfig,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
        )

        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

        encoder_requires_grad = [
            p.requires_grad for name, p in trainer.model.named_parameters() if "classification_head" not in name
        ]
        assert not all(encoder_requires_grad), "Pretrained model is not fully frozen during fine-tuning"


@pytest.mark.needs_gpu
def test_esm2_finetune_regressor(
    tmp_path,
    dummy_data_single_value_regression_ft,
    n_steps_train: int = 50,
    seed: int = 42,
):
    with megatron_parallel_state_utils.distributed_model_parallel_state(seed):
        simple_ft_checkpoint, simple_ft_metrics, trainer = finetune(
            train_data_path=data_to_csv(dummy_data_single_value_regression_ft, tmp_path),
            valid_data_path=data_to_csv(dummy_data_single_value_regression_ft, tmp_path),
            experiment_name="finetune_new_head_regression",
            restore_from_checkpoint_path=str(pretrain_ckpt_path),
            num_steps=n_steps_train,
            num_nodes=1,
            devices=1,
            min_seq_length=None,
            max_seq_length=1024,
            result_dir=tmp_path / "finetune",
            limit_val_batches=2,
            val_check_interval=n_steps_train // 2,
            log_every_n_steps=n_steps_train // 2,
            num_dataset_workers=10,
            lr=1e-5,
            micro_batch_size=4,
            accumulate_grad_batches=1,
            resume_if_exists=False,
            precision="bf16-mixed",
            dataset_class=InMemorySingleValueDataset,
            config_class=ESM2FineTuneSeqConfig,
            metric_tracker=MetricTracker(metrics_to_track_val=["loss"], metrics_to_track_train=["loss"]),
        )

        weights_ckpt = simple_ft_checkpoint / "weights"
        assert weights_ckpt.exists()
        assert weights_ckpt.is_dir()
        assert io.is_distributed_ckpt(weights_ckpt)
        assert simple_ft_metrics.collection_train["loss"][0] > simple_ft_metrics.collection_train["loss"][-1]

        encoder_requires_grad = [
            p.requires_grad for name, p in trainer.model.named_parameters() if "regression_head" not in name
        ]
        assert not all(encoder_requires_grad), "Pretrained model is not fully frozen during fine-tuning"
