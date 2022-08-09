#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from asyncio.log import logger
import numpy as np
import pytest

from habitat.core.spaces import ActionSpace, EmptySpace
from habitat.tasks.nav.nav import IntegratedPointGoalGPSAndCompassSensor

torch = pytest.importorskip("torch")
habitat_baselines = pytest.importorskip("habitat_baselines")

import gym
from torch import distributed as distrib
from torch import nn

from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.config.default import get_config
from habitat_baselines.rl.ddppo.algo import DDPPO
from habitat_baselines.rl.ppo.policy import PointNavBaselinePolicy

from baselines.rl.ppo import (
    PPONonOracle, BaselinePolicyNonOracle
)
from baselines.common.rollout_storage import (
     RolloutStorageNonOracle
)


def _worker_fn(
    world_rank: int, world_size: int, port: int, unused_params: bool
):
    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    
    config = get_config("baselines/config/ppo_multinav.yaml")
    obs_space = gym.spaces.Dict(
        {
            "agent_position": gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(3,),
                dtype=np.float32,
            ),
            "compass": gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(1,),
                dtype=np.float32,
            ),
            "depth": gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(256, 256, 1),
                dtype=np.float32,
            ),
            "gps": gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(2,),
                dtype=np.float32,
            ),
            "heading": gym.spaces.Box(
                low=np.finfo(np.float32).min,
                high=np.finfo(np.float32).max,
                shape=(1,),
                dtype=np.float32,
            ),
            "multiobjectgoal": gym.spaces.Box(
                low=0,
                high=7,
                shape=(1,),
                dtype=np.int64,
            ),
            "rgb": gym.spaces.Box(
                low=0,
                high=255,
                shape=(256, 256, 3),
                dtype=np.uint8,
            )
        }
    )
    action_space = ActionSpace({"FOUND": EmptySpace(), "MOVE_FORWARD": EmptySpace(), "TURN_LEFT": EmptySpace(), "TURN_RIGHT": EmptySpace()})
    ppo_cfg = config.RL.PPO
    actor_critic = BaselinePolicyNonOracle(
            batch_size=config.NUM_ENVIRONMENTS,
            observation_space=obs_space,
            action_space=action_space,
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            device=device,
            object_category_embedding_size=config.RL.OBJECT_CATEGORY_EMBEDDING_SIZE,
            previous_action_embedding_size=config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
            use_previous_action=config.RL.PREVIOUS_ACTION,
            egocentric_map_size=config.RL.MAPS.egocentric_map_size,
            global_map_size=config.RL.MAPS.global_map_size,
            global_map_depth=config.RL.MAPS.global_map_depth,
            coordinate_min=config.RL.MAPS.coordinate_min,
            coordinate_max=config.RL.MAPS.coordinate_max
        )
    # This use adds some arbitrary parameters that aren't part of the computation
    # graph, so they will mess up DDP if they aren't correctly ignored by it
    if unused_params:
        actor_critic.unused = nn.Linear(64, 64)

    actor_critic.to(device=device)
    agent = PPONonOracle(
            actor_critic=actor_critic,
            clip_param=ppo_cfg.clip_param,
            ppo_epoch=ppo_cfg.ppo_epoch,
            num_mini_batch=ppo_cfg.num_mini_batch,
            value_loss_coef=ppo_cfg.value_loss_coef,
            entropy_coef=ppo_cfg.entropy_coef,
            lr=ppo_cfg.lr,
            eps=ppo_cfg.eps,
            max_grad_norm=ppo_cfg.max_grad_norm,
            use_normalized_advantage=ppo_cfg.use_normalized_advantage,
        )
    rollouts = RolloutStorageNonOracle(
            ppo_cfg.num_steps,
            config.NUM_ENVIRONMENTS,
            obs_space,
            action_space,
            ppo_cfg.hidden_size,
            config.RL.MAPS.global_map_size,
            config.RL.MAPS.global_map_depth,
        )
    rollouts.to(device)

    for k, v in rollouts.observations.items():
        if k == "multiobjectgoal":
            rollouts.observations[k] = torch.randint_like(v, low=0, high=7, dtype=torch.int64)
        else:
            rollouts.observations[k] = torch.randn_like(v)

    # Add two steps so batching works
    rollouts.advance_rollout()
    #rollouts.advance_rollout()

    # Get a single batch
    batch = next(rollouts.recurrent_generator(agent.get_advantages(rollouts), ppo_cfg.num_mini_batch))
    (
        obs_batch,
        recurrent_hidden_states_batch,
        global_map_batch,
        actions_batch,
        prev_actions_batch,
        value_preds_batch,
        return_batch,
        masks_batch,
        old_action_log_probs_batch,
        adv_targ,
    ) = batch

    # Call eval actions through the internal wrapper that is used in
    # agent.update
    value, action_log_probs, dist_entropy, _ = agent._evaluate_actions(
        obs_batch,
        recurrent_hidden_states_batch,
        global_map_batch,
        prev_actions_batch,
        masks_batch,
        actions_batch,
    )
    # Backprop on things
    (value.mean() + action_log_probs.mean() + dist_entropy.mean()).backward()

    # Make sure all ranks have very similar parameters
    for param in actor_critic.parameters():
        if param.grad is not None:
            grads = [param.grad.detach().clone() for _ in range(world_size)]
            #distrib.all_gather(grads, grads[world_rank])

            for i in range(world_size):
                assert torch.isclose(grads[i], grads[world_rank]).all()


@pytest.mark.parametrize("unused_params", [True, False])
def test_ddppo_reduce(unused_params: bool):
    world_size = 2
    torch.multiprocessing.spawn(
        _worker_fn,
        args=(world_size, 8748 + int(unused_params), unused_params),
        nprocs=world_size,
    )
