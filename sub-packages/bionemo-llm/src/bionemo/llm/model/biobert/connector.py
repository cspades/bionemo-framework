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

import tarfile
from abc import ABC, abstractmethod
from pathlib import Path
from typing import (
    Any,
    Dict,
    Generic,
    Type,
)

import torch
import yaml
from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer
from nemo.lightning import io, teardown

from bionemo.core.utils.dtypes import get_autocast_dtype
from bionemo.llm.model.biobert.lightning import BioBertLightningModule
from bionemo.llm.model.biobert.model import BioBertGenericConfig, MegatronBioBertModelT
from bionemo.llm.model.biobert.transformer_specs import BiobertSpecOption
from bionemo.llm.utils.weight_utils import nemo1_to_nemo2_biobert_key_mapping


class GenericBioBertNeMo1LightningModuleConnector(
    io.ModelConnector[Dict[str, torch.Tensor], BioBertLightningModule], Generic[MegatronBioBertModelT], ABC
):
    """A generic ModuleConnector for going between nemo1 and nemo2 checkpoints of BERT based models. This is a Path object

    Typically you need to

    Args:
        io: _description_
    """

    @abstractmethod
    def get_config_class(self) -> Type[BioBertGenericConfig[MegatronBioBertModelT]]:
        raise NotImplementedError("Implement me")

    def init(self) -> BioBertLightningModule:
        return BioBertLightningModule(
            self.config,
            self.tokenizer,
        )

    def apply(self, output_path: Path) -> Path:
        nemo1_path = str(self)  # self is a Path object
        with tarfile.open(nemo1_path, "r") as old_ckpt:
            ckpt_file = old_ckpt.extractfile("./model_weights.ckpt")
            old_weights = torch.load(ckpt_file)
        target = self.init()
        trainer = self.nemo_setup(target)
        self.convert_state(old_weights, target)
        self.nemo_save(output_path, trainer)

        print(f"Converted NeMo1, model at {self} saved to {output_path}")

        teardown(trainer, target)
        del trainer, target

        return output_path

    @staticmethod
    def is_te_mapping(model: BioBertLightningModule) -> bool:
        """Check for TE layers, for now infer this from the config."""
        return model.config.biobert_spec_option in {
            BiobertSpecOption.bert_layer_with_transformer_engine_spec,
            BiobertSpecOption.bert_layer_with_transformer_engine_and_qk_ln_spec,
        }

    def convert_state(self, source: Dict[str, torch.Tensor], target: BioBertLightningModule) -> BioBertLightningModule:
        te_mapping = self.is_te_mapping(target)  # check for TE layers.
        new_state_dict_from_old = {}
        for k, v in source.items():
            new_key = nemo1_to_nemo2_biobert_key_mapping(k, new_model_prefix="", te_mapping=te_mapping)
            new_state_dict_from_old[new_key] = v
        target.module.load_state_dict(new_state_dict_from_old, strict=not te_mapping)
        return target

    @property
    @abstractmethod
    def tokenizer(self) -> "AutoTokenizer":
        raise NotImplementedError("Implement this method")
        # from nemo.collections.common.tokenizers.huggingface.auto_tokenizer import AutoTokenizer

        # return AutoTokenizer(self.save_hf_tokenizer_assets(str(self)))

    def get_nemo1_config(self) -> Dict[str, Any]:
        # First read from nemo file and get config settings
        nemo1_path = str(self)  # self inherits from PosixPath
        with tarfile.open(nemo1_path, "r") as old_ckpt:
            config_yaml = old_ckpt.extractfile("./model_config.yaml")
            if config_yaml is None:
                raise ValueError("Config cannot be None in nemo1 checkpoint")
            return yaml.safe_load(config_yaml.read())

    @property
    def config(self) -> BioBertGenericConfig[MegatronBioBertModelT]:
        nemo1_settings = self.get_nemo1_config()
        cfg_class = self.get_config_class()
        autocast_dtype = get_autocast_dtype(nemo1_settings["precision"])
        output = cfg_class(
            params_dtype=autocast_dtype,
            pipeline_dtype=autocast_dtype,
            autocast_dtype=autocast_dtype,
            fp16=autocast_dtype == torch.float16,
            bf16=autocast_dtype == torch.bfloat16,
            **{k: v for k, v in nemo1_settings.items() if k in dir(cfg_class)},
        )
        return output
