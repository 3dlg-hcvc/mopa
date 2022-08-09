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
from baselines.rl.ppo.ppo_trainer import PPOTrainerNO
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
        "--agent-type",
        choices=["no-map", "oracle", "oracle-ego", "proj-neural", "obj-recog", "semantic", "ora-obj-vis"],
        required=True,
        help="agent type: oracle, oracleego, projneural, objrecog, semantic, ora-obj-vis",
    )

    parser.add_argument(
        "opts",
        default=None,
        nargs=argparse.REMAINDER,
        help="Modify config options from command line",
    )

    args = parser.parse_args()
    run_exp(**vars(args))


def run_exp(exp_config: str, run_type: str, agent_type: str, opts=None) -> None:
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
    
    if "TRAINER_NAME" in config and config.TRAINER_NAME not in ["ppo"]:
        trainer_init = baseline_registry.get_trainer(config.TRAINER_NAME)
        config.defrost()
        config.TASK_CONFIG.TRAINER_NAME = config.TRAINER_NAME
        config.RL.PPO.hidden_size = 512
        if config.TRAINER_NAME == "semantic":
                config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
        config.freeze()
    else:
        config.defrost()
        config.TRAINER_NAME = agent_type
        config.TASK_CONFIG.TRAINER_NAME = agent_type
        config.freeze()

        if agent_type in ["oracle", "oracle-ego", "no-map", "ora-obj-vis"]:
            #trainer_init = baseline_registry.get_trainer("oracle")
            trainer_init = baseline_registry.get_trainer("oracle-map")
            config.defrost()
            #config.RL.PPO.hidden_size = 512 if agent_type=="no-map" else 768 --- set this in the config
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MIN_DEPTH = 0.5
            config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.MAX_DEPTH = 5.0
            config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT = 1.5
            if agent_type == "oracle-ego":
                config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
            config.freeze()
        elif agent_type in ["obj-recog"]:
            trainer_init = baseline_registry.get_trainer("obj-recog")
            config.defrost()
            config.RL.PPO.hidden_size = 512
            config.freeze()
        else:
            trainer_init = baseline_registry.get_trainer("non-oracle")
            config.defrost()
            #config.RL.PPO.hidden_size = 512
            if agent_type == "semantic":
                config.TASK_CONFIG.TASK.MEASUREMENTS.append('FOW_MAP')
            config.freeze()
        
    assert trainer_init is not None, f"{config.TRAINER_NAME} is not supported"
    trainer = trainer_init(config)

    if run_type == "train":
        trainer.train()
    elif run_type == "eval":
        trainer.eval()

if __name__ == "__main__":
    main()

    #MIN_DEPTH: 0.5
    #MAX_DEPTH: 5.0
