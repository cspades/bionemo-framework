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
from collections import defaultdict
from dataclasses import dataclass, asdict

import nvidia_resiliency_ext.ptl_resiliency as res_module
import torch
import yaml
from lightning.pytorch.callbacks import LearningRateMonitor, RichModelSummary
from lightning.pytorch.loggers import TensorBoardLogger, WandbLogger
from megatron.core.distributed import DistributedDataParallelConfig
from megatron.core.optimizer import OptimizerConfig
from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.llm.gpt.data import PreTrainingDataModule
from nemo.collections.llm.gpt.data.megatron.hyena import Evo2Dataset
from nemo.collections.nlp.modules.common.tokenizer_utils import get_nmt_tokenizer
from nemo.lightning import NeMoLogger
from nemo.lightning.pytorch import callbacks as nl_callbacks
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from nemo.lightning.pytorch.callbacks.flops_callback import FLOPsMeasurementCallback
from nemo.lightning.pytorch.callbacks.megatron_comm_overlap import MegatronCommOverlapCallback
from nemo.lightning.pytorch.optim import CosineAnnealingScheduler
from nemo.lightning.pytorch.optim.megatron import MegatronOptimizerModule
from nemo.lightning.pytorch.strategies.utils import RestoreConfig
from nemo.utils.exp_manager import TimingCallback

from bionemo.evo2.utils.config import Evo2BlendedDatasetConfig
from bionemo.llm.utils.datamodule_utils import infer_global_batch_size


torch._dynamo.config.suppress_errors = True


def parse_args():
    """Parse arguments for Evo2 model training."""
    parser = argparse.ArgumentParser(description="Train a Hyena model using NeMo 2.0.")
    parser.add_argument(
        "-d",
        "--dataset-config",
        type=str,
        required=True,
        help="Path to the blended / weighted training dataset configuration YAML.",
    )
    parser.add_argument("--num-nodes", type=int, default=1, help="Number of nodes to use for training, defaults to 1.")
    parser.add_argument("--devices", type=int, default=1, help="Number of devices to use for training, defaults to 1.")
    parser.add_argument("--seq-length", type=int, default=8192, help="Training sequence length")
    parser.add_argument(
        "--tensor-parallel-size", type=int, default=1, help="Order of tensor parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--pipeline-model-parallel-size", type=int, default=1, help="Order of pipeline parallelism. Defaults to 1."
    )
    parser.add_argument(
        "--context-parallel-size", type=int, default=1, help="Order of context parallelism. Defaults to 1."
    )
    parser.add_argument("--wandb-project", type=str, default="bionemo_evo2", help="Wandb project name")
    parser.add_argument("--wandb-run-id", type=str, default=None, help="Wandb run identifier")
    parser.add_argument("--sequence-parallel", action="store_true", help="Set to enable sequence parallelism.")
    parser.add_argument("--fp8", action="store_true", help="Set to enable FP8")
    parser.add_argument("--micro-batch-size", type=int, default=1, help="Micro-batch size for data-parallel training.")
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=None,
        help="Global batch size for training. If set to None, infer it from the TP, CP, and PP parameters.",
    )
    parser.add_argument(
        "--grad-acc-batches", type=int, default=1, help="Number of batches to accumulate gradients over."
    )
    parser.add_argument("--max-steps", type=int, help="Number of training optimizer update steps.")
    parser.add_argument(
        "--val-check-interval", type=int, help="Number of steps between validation measurements and model checkpoints."
    )
    parser.add_argument("--grad-reduce-in-fp32", action="store_true", default=False, help="Gradient reduce in FP32.")
    parser.add_argument(
        "--no-aligned-megatron-ddp", action="store_true", default=False, help="Do not do aligned gradient updates etc."
    )
    parser.add_argument("--use-megatron-comm-overlap-llama3-8k", action="store_true", default=False)
    parser.add_argument("--align-param-gather", action="store_true", default=False)
    parser.add_argument("--straggler-detection", action="store_true", default=False)
    parser.add_argument(
        "--model-size",
        type=str,
        choices=["7b", "40b", "test"],
        default="7b",
        help="Model size, choose between 7b, 40b, or test (4 layers, less than 1b).",
    )
    parser.add_argument(
        "--experiment-dir", type=str, default=None, help="Directory to write model checkpoints and results to."
    )
    parser.add_argument(
        "--limit-val-batches", type=int, default=20, help="Number of validation steps",
    )
    parser.add_argument(
        "--ckpt-dir",
        type=str,
        default=None,
        help="Directory to restore an initial checkpoint from. Use this for supervised fine-tuning.",
    )
    parser.add_argument(
        "--restore-optimizer-from-ckpt",
        action="store_true",
        help="Restore optimizer state from initial checkpoint. Defaults to False.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Set random seed for training.")
    parser.add_argument("--workers", type=int, default=0, help="Number of workers to use for data loading.")
    parser.add_argument(
        "--gc-interval",
        type=int,
        default=0,
        help="Set to a value > 0 if you want to synchronize garbage collection, will do gc every gc-interval steps.",
    )
    parser.add_argument(
        "--enable-preemption",
        action="store_true",
        default=False,
        help="Enable preemption hooks. If enabled this will save a checkpoint whenver slurm exits.",
    )
    parser.add_argument(
        "--ckpt-async-save",
        action="store_true",
        default=False,
    )
    parser.add_argument(
        "--ckpt-format",
        type=str,
        choices=['torch_dist', 'zarr'],
        default='torch_dist',
        help="Specify checkpoint format to use. Defaults to 'torch_dist', as 'zarr' is deprecated."
    )
    parser.add_argument(
        "--tflops-callback",
        action="store_true",
        help="Enable tflops calculation callback for Hyena / Evo2. Defaults to False."
    )

    # NSYS profiling/tooling arguments
    parser.add_argument(
        "--nsys-profiling",
        action="store_true",
        default=False,
        help="Enable targeted `nsys` profiling on the training loop for a defined step range. To actually get profiling output you must run the whole program with `nsys`. For example: "
        " `nsys profile -s none -o output_report_name -t cuda,nvtx --force-overwrite true --capture-range=cudaProfilerApi --capture-range-end=stop  [regular python command here]`",
    )
    # start, end, rank
    parser.add_argument(
        "--nsys-start-step",
        type=int,
        required=False,
        default=0,
        help="Start nsys profiling after this step.",
    )
    parser.add_argument(
        "--nsys-end-step",
        type=int,
        required=False,
        help="End nsys profiling after this step.",
    )
    # rank as list of integers
    parser.add_argument(
        "--nsys-ranks",
        type=int,
        nargs="+",
        required=False,
        default=[0],
        help="Enable nsys profiling for these ranks.",
    )

    return parser.parse_args()


@dataclass
class TPOverlapCfg:
    pass


@dataclass
class PipelineOverlapCfg(TPOverlapCfg):
    num_sm: int
    cga_size: int
    num_splits: int
    set_sm_margin: bool
    fp8_buf: bool = (False,)
    method: str = "pipeline"


@dataclass
class RingExchangeOverlapCfg(TPOverlapCfg):
    aggregate: bool = False
    method: str = "ring_exchange"
    num_sm: int = 1
    set_sm_margin: bool = False


@dataclass
class BulkOverlapCfg(TPOverlapCfg):
    num_sm: int
    cga_size: int
    set_sm_margin: bool
    method: str = "bulk"


@dataclass
class TransformerLayerTPOverlapCfg:
    qkv_dgrad: TPOverlapCfg
    qkv_wgrad: TPOverlapCfg
    fc1_dgrad: TPOverlapCfg
    fc1_wgrad: TPOverlapCfg
    qkv_fprop: TPOverlapCfg
    proj_dgrad: TPOverlapCfg
    fc1_fprop: TPOverlapCfg
    fc2_dgrad: TPOverlapCfg
    proj_fprop: TPOverlapCfg
    fc2_fprop: TPOverlapCfg


# TODO: Add more configs and create a getter function for expose a single api
# Model configs: H100/70B/TP8/MBS1/SeqLen8K
userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192 = TransformerLayerTPOverlapCfg(
    qkv_dgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_wgrad=BulkOverlapCfg(num_sm=24, cga_size=2, set_sm_margin=False),
    fc1_dgrad=BulkOverlapCfg(num_sm=2, cga_size=2, set_sm_margin=False),
    fc1_wgrad=BulkOverlapCfg(num_sm=4, cga_size=2, set_sm_margin=False),
    qkv_fprop=RingExchangeOverlapCfg(aggregate=False),
    proj_dgrad=RingExchangeOverlapCfg(aggregate=False),
    fc1_fprop=RingExchangeOverlapCfg(aggregate=False),
    fc2_dgrad=RingExchangeOverlapCfg(aggregate=False),
    proj_fprop=PipelineOverlapCfg(num_sm=24, cga_size=2, num_splits=4, set_sm_margin=True),
    fc2_fprop=PipelineOverlapCfg(num_sm=16, cga_size=2, num_splits=4, set_sm_margin=True),
)


def parse_dataset_config(dataset_config_path: str):
    """Parse the blended training datasplit configuration and renormalize data split weights for training Hyena."""
    blended_dataset_config = defaultdict(list)
    weight_sums = defaultdict(float)
    with open(dataset_config_path, "r") as config_file:
        dataset_config_batch = yaml.safe_load(config_file)
        for dataset_config in dataset_config_batch:
            # Validate.
            config_model = Evo2BlendedDatasetConfig(**dataset_config)
            # Integrate the weights for renormalization.
            weight_sums[config_model.dataset_split] += abs(config_model.dataset_weight)
        for dataset_config in dataset_config_batch:
            # Validate.
            config_model = Evo2BlendedDatasetConfig(**dataset_config)
            # Add indexed dataset to split and associate with blended training weight.
            blended_dataset_config[config_model.dataset_split].extend(
                [config_model.dataset_weight / weight_sums[config_model.dataset_split], config_model.dataset_prefix]
            )
    return blended_dataset_config


def main():
    """Main function to run Evo2 training."""
    args = parse_args()

    # Parse dataset configuration.
    blended_dataset_config = parse_dataset_config(args.dataset_config)

    # Instantiate tokenizer.
    tokenizer = get_nmt_tokenizer(
        "byte-level",
    )

    # Infer global batch size.
    global_batch_size = args.global_batch_size
    if global_batch_size is None:
        global_batch_size = infer_global_batch_size(
            micro_batch_size=args.micro_batch_size,
            num_nodes=args.num_nodes,
            devices=args.devices,
            accumulate_grad_batches=args.grad_acc_batches,
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_model_parallel_size,
            context_model_parallel_size=args.context_parallel_size,
        )

    # Instantiate pre-training module.
    data = PreTrainingDataModule(
        paths=blended_dataset_config,
        dataset_cls=Evo2Dataset,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=global_batch_size,
        seed=args.seed,
        num_workers=args.workers,
        tokenizer=tokenizer,
    )

    if args.model_size == "7b":
        evo2_config = llm.Hyena7bConfig()
    elif args.model_size == "40b":
        evo2_config = llm.Hyena40bConfig()
    elif args.model_size == "test":
        evo2_config = llm.HyenaTestConfig()
    else:
        raise ValueError(f"Invalid model size: {args.model_size}")

    evo2_config.seq_length = args.seq_length
    model = llm.GPTModel(evo2_config, tokenizer=data.tokenizer)

    # Setup callbacks.
    checkpoint_callback = ModelCheckpoint(
        every_n_train_steps=args.val_check_interval,
        dirpath=args.experiment_dir,
        save_top_k=5,
        always_save_context=True,
        save_optim_on_train_end=True,
        save_context_on_train_end=True,
    )
    callbacks = [
        checkpoint_callback,
        RichModelSummary(max_depth=4),
        LearningRateMonitor(),
        TimingCallback(),
    ]
    if args.enable_preemption:
        callbacks.append(nl_callbacks.PreemptionCallback())
    if args.tflops_callback:
        # Add callback that logs the tera-FLOPS per second per GPU during training.
        flop_meas_callback = FLOPsMeasurementCallback(
            asdict(evo2_config),
            data,
            "hyena",
        )
        callbacks.append(flop_meas_callback)

    if args.straggler_detection:
        callbacks.append(
            res_module.StragglerDetectionCallback(
                report_time_interval=300,
                calc_relative_gpu_perf=True,
                calc_individual_gpu_perf=True,
                num_gpu_perf_scores_to_print=5,
                gpu_relative_perf_threshold=0.7,
                gpu_individual_perf_threshold=0.7,
                stop_if_detected=True,
                enable_ptl_logging=True,
            )
        )
    if args.use_megatron_comm_overlap_llama3_8k:
        callbacks.append(
            MegatronCommOverlapCallback(
                tp_comm_overlap=True,
                tp_comm_overlap_cfg=userbuffers_bf16_h100_h8192_tp4_mbs1_seqlen8192,
                wgrad_deferral_limit=22, # default from NeMo
                overlap_param_gather_with_optimizer_step=False,  # Currently disabled due to an issue with checkpointing.
                align_param_gather=args.align_param_gather,
            )
        )

    if args.gc_interval > 0:
        callbacks.append(
            nl_callbacks.GarbageCollectionCallback(
                gc_interval_train=args.gc_interval, gc_interval_val=args.gc_interval
            )
        )
    if args.nsys_profiling:
        if args.nsys_end_step is None:
            nsys_end_step = args.max_steps
        else:
            nsys_end_step = args.nsys_end_step
        callbacks.append(
            nl_callbacks.NsysCallback(
                start_step=args.nsys_start_step, end_step=nsys_end_step, ranks=args.nsys_ranks, gen_shape=True
            )
        )

    loggers = []
    wandb_logger = WandbLogger(
        name=(
            f"evo2-size-{args.model_size}-TP{args.tensor_parallel_size}-"
            f"PP{args.pipeline_model_parallel_size}-CP{args.context_parallel_size}"
            f"-GBS{global_batch_size}-MBS{args.micro_batch_size}"
            f"-GRFP32{args.grad_reduce_in_fp32}-ALIGN{not args.no_aligned_megatron_ddp}"
            f"-NODES{args.num_nodes}-FP8{args.fp8}"
        ),
        id=args.wandb_run_id,  # set this to use the same curve name for restarts.
        project="bionemo_evo2",
        save_dir=args.experiment_dir,
    )
    loggers.append(wandb_logger)
    tb_logger = TensorBoardLogger(
        save_dir="dummy",  ## NOTE: this gets overwritten by default
    )
    loggers.append(tb_logger)

    nemo_logger = NeMoLogger(log_dir=args.experiment_dir, wandb=wandb_logger)
    if args.no_aligned_megatron_ddp:
        ddp: str | DistributedDataParallelConfig = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=args.grad_reduce_in_fp32,
            align_param_gather=args.align_param_gather,
        )
    else:
        ddp = DistributedDataParallelConfig(
            check_for_nan_in_grad=True,
            grad_reduce_in_fp32=args.grad_reduce_in_fp32,
            overlap_grad_reduce=True,
            overlap_param_gather=True,
            average_in_collective=True,
            align_param_gather=args.align_param_gather,
            use_distributed_optimizer=True,  # this should inherit from the optimizer config, but just in case...
        )
    # Initialize Megatron Strategy and Trainer.
    strategy = nl.MegatronStrategy(
        ddp=ddp,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        context_parallel_size=args.context_parallel_size,
        pipeline_dtype=torch.bfloat16,
        sequence_parallel=args.sequence_parallel,
        ckpt_load_optimizer=True,
        ckpt_save_optimizer=True,
        ckpt_async_save=args.ckpt_async_save,
        save_ckpt_format=args.ckpt_format,
    )
    trainer = nl.Trainer(
        devices=args.devices,
        num_nodes=args.num_nodes,
        max_steps=args.max_steps,
        accelerator="gpu",
        strategy=strategy,
        logger=loggers,
        callbacks=callbacks,
        log_every_n_steps=1,
        limit_val_batches=args.limit_val_batches,
        num_sanity_val_steps=0,
        use_distributed_sampler=False,
        plugins=nl.MegatronMixedPrecision(
            precision="bf16-mixed",
            params_dtype=torch.bfloat16,
            fp8="hybrid" if args.fp8 else None,
            fp8_amax_history_len=16 if args.fp8 else 1,
            fp8_amax_compute_algo="max" if args.fp8 else "most_recent",
        ),
        val_check_interval=args.val_check_interval,
    )

    # Logger setup
    nemo_logger.setup(
        trainer,
        resume_if_exists=True,
    )

    resume = nl.AutoResume(
        resume_if_exists=True,
        resume_ignore_no_checkpoint=True,
        resume_past_end=False,
        resume_from_directory=args.experiment_dir,
        restore_config=(
            RestoreConfig(
                path=args.ckpt_dir,
                load_model_state=True,
                load_optim_state=args.restore_optimizer_from_ckpt,
            )
            if args.ckpt_dir
            else None
        ),
    )
    resume.setup(trainer, model)

    # Optimizer and scheduler setup
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=0.0003,
        adam_beta1=0.9,
        adam_beta2=0.95,
        use_distributed_optimizer=True,
        bf16=True,
    )
    sched = CosineAnnealingScheduler(
        max_steps=trainer.max_steps,
        warmup_steps=2500,
        min_lr=0.000003,
    )

    opt = MegatronOptimizerModule(opt_config, sched)
    opt.connect(model)

    # Start training
    trainer.fit(model, data)


if __name__ == "__main__":
    main()
