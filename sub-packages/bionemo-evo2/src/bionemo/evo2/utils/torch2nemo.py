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


import argparse
from dataclasses import dataclass

import torch
from nemo.collections.llm.gpt.model.hyena import HyenaConfig, PyTorchHyenaImporter


"""
python torch2nemo.py --model-ckpt pretrained_model/global_step199400 --output-nemo-ckpt nemo_pretrained_model/evo2_nemo_pretrained.nemo
"""


@dataclass
class Hyena40bPretrainedConfig(HyenaConfig):
    """Fixes SDHSDH instead of SHDSDH in the original 40b, probably a typo there?"""

    hybrid_override_pattern: str = "SDH*SDHSDH*SDHSDH*SDHSDH*SDHSDH*SDH*SDHSDH*SDHSDH*"
    num_layers: int = 50
    seq_length: int = 8192
    hidden_size: int = 8192
    num_groups_hyena: int = 8192
    num_groups_hyena_medium: int = 512
    num_groups_hyena_short: int = 512
    make_vocab_size_divisible_by: int = 8
    tokenizer_library: str = "byte-level"
    mapping_type: str = "base"
    ffn_hidden_size: int = 21888
    gated_linear_unit: bool = True
    num_attention_heads: int = 64
    use_cpu_initialization: bool = False
    hidden_dropout: float = 0.0
    attention_dropout: float = 0.0
    params_dtype: torch.dtype = torch.bfloat16
    normalization: str = "RMSNorm"
    add_qkv_bias: bool = False
    add_bias_linear: bool = False
    layernorm_epsilon: float = 1e-6
    # fp8: str = 'hybrid'
    # fp8_amax_history_len: int = 16
    # fp8_amax_compute_algo: str = "max"
    recompute_granularity: str = "full"
    recompute_method: str = "uniform"
    recompute_num_layers: int = 2
    hyena_init_method: str = "small_init"
    hyena_output_layer_init_method: str = "wang_init"
    hyena_filter_no_wd: bool = True


def parse_args():
    """Parse args."""
    parser = argparse.ArgumentParser(description="PyTorch to NeMo converter")
    parser.add_argument("--model-ckpt", type=str, required=True, help="Path to the PyTorch model checkpoint.")
    parser.add_argument("--output-nemo-ckpt", type=str, required=True, help="Path to the NeMo model checkpoint.")
    return parser.parse_args()


if __name__ == "__main__":
    # print(torch.load("pretrained_model/global_step199400/mp_rank_00_model_states.pt").keys())
    args = parse_args()
    pretrained_config: HyenaConfig = Hyena40bPretrainedConfig()
    importer = PyTorchHyenaImporter(args.model_ckpt, model_config=pretrained_config)
    importer.apply(args.output_nemo_ckpt)


"""
{
  # Logging
  'use_wandb': true,
  "print_mem_alloc_stats": false,
  "log_memory_stats": true,
  "log_memory_alloc_counts": false,
  # MP / PP config
  'pipe_parallel_size': 0,
  'model_parallel_size': 2,
  'sequence_parallel': true,

  # Zero config
  # Leaf Modules
    # Modules to mark as leaf modules, only for Zero-stage 3
    # Must be list of strings that can be be used as such getattr(module, leaf_module)
    # where module is one of savanna.model.block or savanna.model.operators.hyena.hyena.

    # This controls the granularity of zero-3 parameter partitioning.  I.e., if ParallelSequenceMixer is
    # set as a leaf module, then the entire ParallelSequenceMixer will be gathered / partitioned as a single unit.
    # ParallelSequenceMixer is the equivalent of an AttentionBlock: input projections, self attention, and output projections.
    # ParallelBlockPipe is the equivalent of a TransformerBlock: AttentionBlock + FFN
    # backbone modules: ParallelBlockPipe
    # block modules:  'ParallelSequenceMixer', 'ParallelGLU', 'ParallelLinear', 'FlexLinear', 'ParallelMLP',
    # hyena_modules: 'ParallelCausalDepthwiseConv1d', 'ParallelComplexModalFilter', 'ParallelHyenaOperator', 'ParallelImplicitFreeformFilter', 'ParallelShortHyenaOperator',
  #NOTE: If a module is specified as a leaf module, all its nested modules will be
  'zero_use_leaf_modules': false,
  'zero_leaf_modules': ["ParallelSequenceMixer", "ParallelGLU"],

  'zero_use_mics': false,
  'zero_optimization':
    {
      'stage': 3,
      'prefetch_bucket_size': 500000000,
      'max_live_parameters': 1000000000,
      'allgather_partitions': True,
      'allgather_bucket_size': 500000000,
      'overlap_comm': True,
      'reduce_scatter': True,
      'reduce_bucket_size': 500000000,
      'contiguous_gradients': True,
      'cpu_offload': false,
      'param_persistence_threshold': 0,
      # "mics_shard_size": 8,
      # "mics_hierarchical_params_gather": false,
    },

  # Batch sizing
  'train_micro_batch_size_per_gpu': 8,
  'gradient_accumulation_steps': 1,

  # Activation checkpointing
  'checkpoint-activations': true,
  'checkpoint-num-layers': 1,

  # Training
  'train-iters': 40,
  'lr-decay-iters': 40,

  'make_vocab_size_divisible_by': 8,
  'num_layers': 50,
  'hidden_size': 8192,
  'num_attention_heads': 64,
  'num_groups_hyena': 8192,
  'num_groups_hyena_medium': 512,
  'num_groups_hyena_short': 512,
  'num_groups_hyena_mlp': 512,
  'operator-config':
    [
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['hyena_short_conv'], 1],
      [['hyena_medium_conv'], 1],
      [['hyena'], 1],
      [['flash_v2'], 1],
    ],

  # These kernels will be also autotuned and activated
  'use_cgcg': false,
  'use_cgcg_short': false,
  'use_cgcg_mlp': false,

  # Tune to target sequence length e.g., 8192
  'seq_length': 8192,
  'max_position_embeddings': 8192,

  'hyena_medium_conv_len': 128, # default is null
  'log_attn_norms': false,
  'pos_emb': 'rotary',
  'rotary_emb_base': 1000000,
  'rotary_pct': 1,
  'prenorm': true,
  'postnorm': false,
  'pre_mlp_norm': true,
  'outer_mlp_norm': false,
  'no_weight_tying': false,
  'gpt_j_residual': false,
  'normalize_hyena_filters': false,
  'short-conv-L': 3,
  'hyena_filter_fast_decay': 0.3,
  'hyena_filter_slow_decay': 1.2,
  'hyena_filter_w': 14,
  'hyena_filter_cls': 'implicit_modal',
  'hyena_medium_filter_cls': 'explicit_single_decay',
  'explicit_filter_decay_preset': 'weak',
  'hyena_filter_order': 16,
  'hyena_filter_wd': 0.,
  'use_fast_heads': false,
  'use_slow_heads': false,
  'use-hyena-filter': true,
  'output_layer_parallelism': 'column',
  'bias_dropout_fusion': false,
  'norm': 'rmsnorm',
  'rms_norm_epsilon': 1.0e-6,
  'identity_mlp': false,
  'activation': 'gelu',
  'mlp_type': 'llama',
  'scaled-upper-triang-masked-softmax-fusion': true,
  'bias-gelu-fusion': false,
  'init_method': 'small_init',
  'output_layer_init_method': 'wang_init',
  'optimizer':
    {
      'type': 'Adam',
      'params': { 'lr': 0.0003, 'betas': [0.9, 0.95], 'eps': 1.0e-8 },
    },
  'min_lr': 0.00003,

  'data-impl': 'mmap',

  'partition-activations': false,
  'synchronize-each-layer': false,
  'gradient_clipping': 1.0,
  'weight-decay': 0.1,
  'hidden-dropout': 0.0,
  'attention-dropout': 0.0,
  'precision': 'bfloat16',
  'bf16': { 'enabled': true },
  'distributed-backend': 'nccl',
  'lr-decay-style': 'cosine',
  'warmup': 0.005,
  'checkpoint-factor': 2500,
  'extra_save_iters': [100],
  'eval-interval': 200,
  'eval-iters': 20,
  'log-interval': 5,
  'steps_per_print': 5,
  'keep-last-n-checkpoints': 100,
  'wall_clock_breakdown': false,

  'tokenizer_type': CharLevelTokenizer,
  'use_fp8_input_projections': true,
  'use_fp8_output_projections': true,
  'use_fp8_mlp_projections': true,
  'use_fp8_norm': true,
  'checkpoint_strict_load': false,
  'make_gated_mlp_multiple_of': 128,
  'materialize_attn_mask': false, # default false, to save memory
  'fast_conv_proj': true,
  'hyena_short_conv_len': 7,
  'to_upper': "normalized_weighted",
  'mask_loss_control_tags': true,
  'lowercase_loss_reweighting': 0.1,
}
"""
