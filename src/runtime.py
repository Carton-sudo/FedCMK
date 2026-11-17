"""Runtime assembly extracted from the research entry; use ../run.py."""

# _get_setups, _random_seeder and main retain their source AST unchanged.

import torch

import torch.optim as optim

import torch.optim.lr_scheduler as lr_scheduler

import algorithms

from train_tools import *

from utils import *

import numpy as np

import argparse

import warnings

import wandb

import random

import pprint

import os

import contextlib

import json

import sys

import time

ALGO = {
    "fedavg": algorithms.fedavg.Server,
    "fedprox": algorithms.fedprox.Server,
    "moon": algorithms.moon.Server,
    "fedntd": algorithms.fedntd.Server,
    "fedlmd": algorithms.fedlmd.Server,
    "fedcmk": algorithms.fedcmk.Server,
}

SCHEDULER = {
    "step": lr_scheduler.StepLR,
    "multistep": lr_scheduler.MultiStepLR,
    "cosine": lr_scheduler.CosineAnnealingLR,
}

def _get_setups(args):
    """Get train configuration"""

    # Fix randomness for data distribution
    partition_seed = int(getattr(args.data_setups.partition, "seed", 19940817))
    np.random.seed(partition_seed)
    random.seed(partition_seed)

    # 1 数据 Distribute the data to clients
    algo_params = args.train_setups.algo.params
    local_metrics_enabled = getattr(
        algo_params, "local_metrics_enabled", True
    )
    data_init_start = time.perf_counter()
    data_distributed = data_distributer(
        **args.data_setups,
        local_metrics_enabled=local_metrics_enabled,
    )
    data_init_sec = time.perf_counter() - data_init_start

    # Fix randomness for experiment
    _random_seeder(args.train_setups.seed)
    # 2 实例化 神经网络
    model = create_models(
        args.train_setups.model.name,
        args.data_setups.dataset_name,
        **args.train_setups.model.params,
    )

    # 3 优化器设定 Optimization setups
    #{
    optimizer = optim.SGD(model.parameters(), **args.train_setups.optimizer.params)
    scheduler = None # 这个是学习率的衰减策略
    #   }

    if args.train_setups.scheduler.enabled:
        scheduler = SCHEDULER[args.train_setups.scheduler.name](
            optimizer, **args.train_setups.scheduler.params
        )

    # Algorith-specific global server container
    logging_setups = dict(args.train_setups.get("logging", {}))
    partition = args.data_setups.partition
    if partition.method == "sharding":
        partition_name = f"sharding_s{partition.shard_per_user}"
    elif partition.method == "lda":
        partition_name = f"lda_alpha{partition.alpha}"
    else:
        partition_name = str(partition.method)
    run_metadata = {
        "algorithm": args.train_setups.algo.name,
        "dataset": args.data_setups.dataset_name,
        "partition": partition_name,
        "partition_seed": partition_seed,
        "partition_hash": data_distributed.get("partition_hash"),
        "partition_index_path": data_distributed.get("partition_index_path"),
        "seed": int(args.train_setups.seed),
        "data_init_sec": data_init_sec,
    }
    server = ALGO[args.train_setups.algo.name](
        algo_params,
        model,
        data_distributed,
        optimizer,
        scheduler,
        **args.train_setups.scenario,
        **logging_setups,
        run_metadata=run_metadata,
    )

    return server

def _random_seeder(seed):
    """Fix randomness"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args):

    """Assemble the server, run training and save configured model artifacts."""

    # Load the configuration
    server = _get_setups(args)

    # Conduct FL
    server.run()

    # Keep model persistence independent from metric persistence.  Formal
    # metric-only campaigns can disable all model-bearing artifacts without
    # changing training, evaluation, or round_metrics.csv.
    if server.save_model_artifacts:
        model_dir = (
            str(server.output_dir)
            if server.output_dir is not None
            else wandb.run.dir
        )
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "model.pth")
        torch.save(server.export_model_state_dict(), model_path)
        wandb.save(model_path)
    else:
        print(
            "[Artifact Policy] Model checkpoints and model.pth are disabled; "
            "metrics and provenance files remain enabled."
        )
