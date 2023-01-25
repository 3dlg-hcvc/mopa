#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import sys
import argparse
import random
import numpy as np
import torch
from habitat_baselines.common.baseline_registry import baseline_registry
from baselines.rl.ppo.ppo_trainer_hier import HierOnTrainer
from baselines.rl.ppo.ppo_trainer_sem_map import SemMapOnTrainer
from baselines.rl.ppo.ppo_trainer_sem_map_real import SemMapOnRealTrainer
from baselines.rl.ppo.ppo_trainer_pred_sem_map import PredSemMapOnTrainer
from baselines.rl.ppo.ppo_trainer_pred_sem_map_w_real_obj import PredSemMapRealOnTrainer
from baselines.rl.ppo.ppo_trainer_ora_map_w_path_planner import MapWithPathPlannerOnTrainer
from baselines.rl.ppo.ppo_trainer_ora_map_w_fast_marching import MapWithFMMOnTrainer
from baselines.rl.ppo.ppo_trainer_sem_map_shortest_pp import ShortestPathPlannerTrainer
from baselines.rl.ppo.ppo_trainer_sem_map_shortest_pp_map import ShortestPathPlannerMapTrainer
from baselines.rl.ppo.ppo_trainer_pred_sem_map_rednet import PredSemMapRedNetOnTrainer
from baselines.rl.ppo.ppo_trainer_sem_map_frontier import SemMapOnFrontierTrainer
from baselines.config.default import get_config
from baselines.nonlearning_agents import (
    evaluate_agent,
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-type",
        choices=["train", "eval"],
        required=True,
        help="run type of the experiment (train or eval)",
    )
    parser.add_argument(
        "--exp-config",
        type=str,
        required=True,
        help="path to config yaml containing info about experiment",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, opts=None) -> None:
    r"""Runs experiment given mode and config

    Args:
        exp_config: path to config file.
        run_type: "train" or "eval.
        opts: list of strings of additional config options.

    Returns:
        None.
    """
    config = get_config(exp_config, opts)
    random.seed(config.TASK_CONFIG.SEED)
    np.random.seed(config.TASK_CONFIG.SEED)
    torch.manual_seed(config.TASK_CONFIG.SEED)
    
    if run_type == "eval" and config.EVAL.EVAL_NONLEARNING:
        evaluate_agent(config)
        return
    
    trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
    config.defrost()
    config.TASK_CONFIG.TRAINER_NAME = config.TRAINER_NAME
    config.freeze()
    
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()

if __name__ == "__main__":
    main()
