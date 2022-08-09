#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import json
import os
import time
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import random
from einops import rearrange
import math
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch_scatter
import tqdm
from torch.optim.lr_scheduler import LambdaLR
from gym import spaces

from habitat import Config, logger, VectorEnv
from habitat.utils.visualizations.utils import append_text_to_image
from baselines.common.viz_utils import observations_to_image
from baselines.common.base_trainer import (
    BaseRLTrainerNonOracle)
from habitat_baselines.common.baseline_registry import baseline_registry
from baselines.common.env_utils import construct_envs
from habitat.core.environments import get_env_class
from baselines.common.rollout_storage import (
     RolloutStorageNonOracle)
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer,
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
)
from baselines.rl.ppo import (
    PPONonOracle, BaselinePolicyNonOracle
    )
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.common.tensorboard_utils import TensorboardWriter

from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    get_distrib_size,
    init_distrib_slurm,
    is_slurm_batch_job,
    rank0_only,
    requeue_job,
    load_resume_state,
    save_resume_state
)


@baseline_registry.register_trainer(name="non-oracle")
class PPOTrainerNO(BaseRLTrainerNonOracle):
    r"""Trainer class for PPO algorithm
    Paper: https://arxiv.org/abs/1707.06347.
    """
    supported_tasks = ["Nav-v0"]

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        if config is not None:
            logger.info(f"config: {config}")

        self._static_encoder = False
        self._encoder = None

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        self.actor_critic = BaselinePolicyNonOracle(
            batch_size=self.config.NUM_ENVIRONMENTS,
            observation_space=self.envs.observation_spaces[0],
            action_space=self.envs.action_spaces[0],
            hidden_size=ppo_cfg.hidden_size,
            goal_sensor_uuid=self.config.TASK_CONFIG.TASK.GOAL_SENSOR_UUID,
            device=self.device,
            object_category_embedding_size=self.config.RL.OBJECT_CATEGORY_EMBEDDING_SIZE,
            previous_action_embedding_size=self.config.RL.PREVIOUS_ACTION_EMBEDDING_SIZE,
            use_previous_action=self.config.RL.PREVIOUS_ACTION,
            egocentric_map_size=self.config.RL.MAPS.egocentric_map_size,
            global_map_size=self.config.RL.MAPS.global_map_size,
            global_map_depth=self.config.RL.MAPS.global_map_depth,
            coordinate_min=self.config.RL.MAPS.coordinate_min,
            coordinate_max=self.config.RL.MAPS.coordinate_max
        )
        self.actor_critic.to(self.device)

        self.agent = PPONonOracle(
            actor_critic=self.actor_critic,
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

    def save_checkpoint(
        self, file_name: str, extra_state: Optional[Dict] = None
    ) -> None:
        r"""Save checkpoint with specified name.

        Args:
            file_name: file name for checkpoint

        Returns:
            None
        """
        checkpoint = {
            "state_dict": self.agent.state_dict(),
            "config": self.config,
        }
        if extra_state is not None:
            checkpoint["extra_state"] = extra_state

        torch.save(
            checkpoint, os.path.join(self.config.CHECKPOINT_FOLDER, file_name)
        )

    def load_checkpoint(self, checkpoint_path: str, *args, **kwargs) -> Dict:
        r"""Load checkpoint of specified path as a dict.

        Args:
            checkpoint_path: path of target checkpoint
            *args: additional positional args
            **kwargs: additional keyword args

        Returns:
            dict containing checkpoint info
        """
        return torch.load(checkpoint_path, *args, **kwargs)

    METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "raw_metrics"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if (k + "." + subk) not in cls.METRICS_BLACKLIST
                    }
                )
            # Things that are scalar-like will have an np.size of 1.
            # Strings also have an np.size of 1, so explicitly ban those
            elif np.size(v) == 1 and not isinstance(v, str):
                result[k] = float(v)

        return result

    @classmethod
    def _extract_scalars_from_infos(
        cls, infos: List[Dict[str, Any]]
    ) -> Dict[str, List[float]]:

        results = defaultdict(list)
        for i in range(len(infos)):
            for k, v in cls._extract_scalars_from_info(infos[i]).items():
                results[k].append(v)

        return results

    def percent_done(self) -> float:
        if self.config.NUM_UPDATES != -1:
            return self.num_updates_done / self.config.NUM_UPDATES
        else:
            return self.num_steps_done / self.config.TOTAL_NUM_STEPS

    def is_done(self) -> bool:
        return self.percent_done() >= 1.0

    def _collect_rollout_step(
        self, rollouts, current_episode_reward, running_episode_stats
    ):
        pth_time = 0.0
        env_time = 0.0

        t_sample_action = time.time()
        # sample actions
        with torch.no_grad():
            step_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }

            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
                global_map,
            ) = self.actor_critic.act(
                step_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.global_map[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            )

        pth_time += time.time() - t_sample_action

        t_step_env = time.time()

        outputs = self.envs.step([a[0].item() for a in actions])
        observations, rewards, dones, infos = [list(x) for x in zip(*outputs)]

        env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(observations, device=self.device)
        rewards = torch.tensor(
            rewards, dtype=torch.float, device=current_episode_reward.device
        )
        rewards = rewards.unsqueeze(1)

        masks = torch.tensor(
            [[0.0] if done else [1.0] for done in dones],
            dtype=torch.float,
            device=current_episode_reward.device,
        )

        current_episode_reward += rewards
        running_episode_stats["reward"] += (1 - masks) * current_episode_reward
        running_episode_stats["count"] += 1 - masks
        for k, v in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v, dtype=torch.float, device=current_episode_reward.device
            ).unsqueeze(1)
            if k not in running_episode_stats:
                running_episode_stats[k] = torch.zeros_like(
                    running_episode_stats["count"]
                )

            running_episode_stats[k] += (1 - masks) * v

        current_episode_reward *= masks

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        rollouts.insert(
            batch,
            recurrent_hidden_states,
            global_map,
            actions,
            actions_log_probs,
            values,
            rewards,
            masks,
        )

        pth_time += time.time() - t_update_stats

        return pth_time, env_time, self.envs.num_envs

    def _update_agent(self, ppo_cfg, rollouts):
        t_update_model = time.time()
        with torch.no_grad():
            last_observation = {
                k: v[rollouts.step] for k, v in rollouts.observations.items()
            }
            next_value = self.actor_critic.get_value(
                last_observation,
                rollouts.recurrent_hidden_states[rollouts.step],
                rollouts.global_map[rollouts.step],
                rollouts.prev_actions[rollouts.step],
                rollouts.masks[rollouts.step],
            ).detach()

        rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        value_loss, action_loss, dist_entropy = self.agent.update(rollouts)

        rollouts.after_update()

        return (
            time.time() - t_update_model,
            value_loss,
            action_loss,
            dist_entropy,
        )

    def train(self) -> None:
        r"""Main method for training PPO.

        Returns:
            None
        """

        self.envs = construct_envs(
            self.config, get_env_class(self.config.ENV_NAME)
        )

        ppo_cfg = self.config.RL.PPO
        self.device = (
            torch.device("cuda", self.config.TORCH_GPU_ID)
            if torch.cuda.is_available()
            else torch.device("cpu")
        )
        if not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)
        self._setup_actor_critic_agent(ppo_cfg)
        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        rollouts = RolloutStorageNonOracle(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            self.envs.observation_spaces[0],
            self.envs.action_spaces[0],
            ppo_cfg.hidden_size,
            self.config.RL.MAPS.global_map_size,
            self.config.RL.MAPS.global_map_depth,
        )
        rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        for sensor in rollouts.observations:
            rollouts.observations[sensor][0].copy_(batch[sensor])

        # batch and observations may contain shared PyTorch CUDA
        # tensors.  We must explicitly clear them here otherwise
        # they will be kept in memory for the entire duration of training!
        batch = None
        observations = None

        current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        t_start = time.time()
        env_time = 0
        pth_time = 0
        prev_time = 0
        count_steps = 0
        count_checkpoints = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: linear_decay(x, self.config.NUM_UPDATES),
        )
        
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            env_time = requeue_stats["env_time"]
            pth_time = requeue_stats["pth_time"]
            count_steps = requeue_stats["count_steps"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            _last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            running_episode_stats = requeue_stats["running_episode_stats"]
            window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )
        add_signal_handlers()

        with TensorboardWriter(
            self.config.TENSORBOARD_DIR, flush_secs=self.flush_secs
        ) as writer:
            for update in range(self.num_updates_done, self.config.NUM_UPDATES):
                self.num_updates_done = update
                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * linear_decay(
                        update, self.config.NUM_UPDATES
                    )
                    
                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=env_time,
                        pth_time=pth_time,
                        count_checkpoints=count_checkpoints,
                        count_steps=count_steps,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - t_start) + prev_time,
                        running_episode_stats=running_episode_stats,
                        window_episode_stats=dict(window_episode_stats),
                    )

                    save_resume_state(
                        dict(
                            state_dict=self.agent.state_dict(),
                            optim_state=self.agent.optimizer.state_dict(),
                            lr_sched_state=lr_scheduler.state_dict(),
                            config=self.config,
                            requeue_stats=requeue_stats,
                        ),
                        self.config,
                    )

                if EXIT.is_set():
                    profiling_wrapper.range_pop()  # train update

                    self.envs.close()

                    requeue_job()

                    return

                for step in range(ppo_cfg.num_steps):
                    (
                        delta_pth_time,
                        delta_env_time,
                        delta_steps,
                    ) = self._collect_rollout_step(
                        rollouts, current_episode_reward, running_episode_stats
                    )
                    pth_time += delta_pth_time
                    env_time += delta_env_time
                    count_steps += delta_steps

                (
                    delta_pth_time,
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent(ppo_cfg, rollouts)
                pth_time += delta_pth_time
                
                for k, v in running_episode_stats.items():
                    window_episode_stats[k].append(v.clone())

                deltas = {
                    k: (
                        (v[-1] - v[0]).sum().item()
                        if len(v) > 1
                        else v[0].sum().item()
                    )
                    for k, v in window_episode_stats.items()
                }
                deltas["count"] = max(deltas["count"], 1.0)

                writer.add_scalar(
                    "train/reward", deltas["reward"] / deltas["count"], count_steps
                )

                writer.add_scalar(
                    "train/learning_rate", lr_scheduler._last_lr[0], count_steps
                )

                total_actions = rollouts.actions.shape[0] * rollouts.actions.shape[1]
                total_found_actions = int(torch.sum(rollouts.actions == 0).cpu().numpy())
                total_forward_actions = int(torch.sum(rollouts.actions == 1).cpu().numpy())
                total_left_actions = int(torch.sum(rollouts.actions == 2).cpu().numpy())
                total_right_actions = int(torch.sum(rollouts.actions == 3).cpu().numpy())
                total_look_up_actions = int(torch.sum(rollouts.actions == 4).cpu().numpy())
                total_look_down_actions = int(torch.sum(rollouts.actions == 5).cpu().numpy())
                assert total_actions == (total_found_actions + total_forward_actions + 
                    total_left_actions + total_right_actions + total_look_up_actions + 
                    total_look_down_actions
                )
                # Check later why this assertion is not true
                # total_actions = (total_stop_actions + total_forward_actions + 
                #     total_left_actions + total_right_actions + total_look_up_actions + 
                #     total_look_down_actions
                # )
                writer.add_histogram(
                    "map_encoder_cnn_0", self.actor_critic.net.map_encoder.cnn[0].weight, count_steps
                )
                writer.add_histogram(
                    "map_encoder_cnn_1", self.actor_critic.net.map_encoder.cnn[2].weight, count_steps
                )
                writer.add_histogram(
                    "map_encoder_cnn_2", self.actor_critic.net.map_encoder.cnn[4].weight, count_steps
                )
                writer.add_histogram(
                    "map_encoder_linear", self.actor_critic.net.map_encoder.cnn[6].weight, count_steps
                )
                
                
                writer.add_histogram(
                    "visual_encoder_cnn_0", self.actor_critic.net.visual_encoder.cnn[0].weight, count_steps
                )
                writer.add_histogram(
                    "visual_encoder_cnn_1", self.actor_critic.net.visual_encoder.cnn[2].weight, count_steps
                )
                writer.add_histogram(
                    "visual_encoder_cnn_2", self.actor_critic.net.visual_encoder.cnn[4].weight, count_steps
                )
                writer.add_histogram(
                    "visual_encoder_linear", self.actor_critic.net.image_features_linear.weight, count_steps
                )
                writer.add_scalar(
                    "train/found_action_prob", total_found_actions/total_actions, count_steps
                )
                writer.add_scalar(
                    "train/forward_action_prob", total_forward_actions/total_actions, count_steps
                )
                writer.add_scalar(
                    "train/left_action_prob", total_left_actions/total_actions, count_steps
                )
                writer.add_scalar(
                    "train/right_action_prob", total_right_actions/total_actions, count_steps
                )
                writer.add_scalar(
                    "train/look_up_action_prob", total_look_up_actions/total_actions, count_steps
                )
                writer.add_scalar(
                    "train/look_down_action_prob", total_look_down_actions/total_actions, count_steps
                )

                metrics = {
                    k: v / deltas["count"]
                    for k, v in deltas.items()
                    if k not in {"reward", "count"}
                }

                for k, v in metrics.items():
                    writer.add_scalar(f"metrics/{k}", v, count_steps)
                    
                
                writer.add_scalar("train/losses_value", value_loss, count_steps)
                writer.add_scalar("train/losses_policy", action_loss, count_steps)
                writer.add_scalar("train/dist_entropy", dist_entropy, count_steps)


                # log stats
                if update > 0 and update % self.config.LOG_INTERVAL == 0:
                    logger.info(
                        "update: {}\tfps: {:.3f}\t".format(
                            update, count_steps / (time.time() - t_start)
                        )
                    )

                    logger.info(
                        "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                        "frames: {}".format(
                            update, env_time, pth_time, count_steps
                        )
                    )

                    logger.info(
                        "Average window size: {}  {}".format(
                            len(window_episode_stats["count"]),
                            "  ".join(
                                "{}: {:.3f}".format(k, v / deltas["count"])
                                for k, v in deltas.items()
                                if k != "count"
                            ),
                        )
                    )

                # checkpoint model
                if update % self.config.CHECKPOINT_INTERVAL == 0:
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth", 
                        dict(
                            step=count_steps,
                            wall_time=(time.time() - t_start) + prev_time,
                        )
                    )
                    count_checkpoints += 1

            self.envs.close()

    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        r"""Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object for logging to tensorboard
            checkpoint_index: index of cur checkpoint for logging

        Returns:
            None
        """
        # Map location CPU is almost always better than mapping to a CUDA device.
        ckpt_dict = self.load_checkpoint(checkpoint_path, map_location="cpu")

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO
        map_config = config.RL.MAPS

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False # shuffling turned off
        config.freeze()

        # if len(self.config.VIDEO_OPTION) > 0 and np.random.uniform(0, 1) <= self.config.VIDEO_PROB:
        if len(self.config.VIDEO_OPTION) > 0:
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        logger.info(f"env config: {config}")
        self.envs = construct_envs(config, get_env_class(config.ENV_NAME))
        self._setup_actor_critic_agent(ppo_cfg)

        self.agent.load_state_dict(ckpt_dict["state_dict"])
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(observations, device=self.device)

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device=self.device
        )

        test_recurrent_hidden_states = torch.zeros(
            self.actor_critic.net.num_recurrent_layers,
            self.config.NUM_ENVIRONMENTS,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        test_global_map = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.config.RL.MAPS.global_map_size,
            self.config.RL.MAPS.global_map_size,
            self.config.RL.MAPS.global_map_depth,
        )
        test_global_map_visualization = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.config.RL.MAPS.global_map_size,
            self.config.RL.MAPS.global_map_size,
            3,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS, 1, device=self.device, dtype=torch.long
        )
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS, 1, device=self.device
        )
        stats_episodes = dict()  # dict of dicts that stores stats per episode
        raw_metrics_episodes = dict()

        all_actions = {}
        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        pbar = tqdm.tqdm(total=self.config.TEST_EPISODE_COUNT)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < self.config.TEST_EPISODE_COUNT
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()

            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                    test_global_map,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    test_global_map,
                    prev_actions,
                    not_done_masks,
                    deterministic=False,
                )

                prev_actions.copy_(actions)

            outputs = self.envs.step([a[0].item() for a in actions])

            observations, rewards, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(observations, device=self.device)

            not_done_masks = torch.tensor(
                [[0.0] if done else [1.0] for done in dones],
                dtype=torch.float,
                device=self.device,
            )

            # Reset global map
            #test_global_map_visualization = not_done_masks.unsqueeze(2).unsqueeze(3).cpu() * test_global_map_visualization

            rewards = torch.tensor(
                rewards, dtype=torch.float, device=self.device
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if current_episodes[i].episode_id not in all_actions:
                    all_actions[current_episodes[i].episode_id] = []
                all_actions[current_episodes[i].episode_id].append(actions[i].item())
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not_done_masks[i].item() == 0:
                    pbar.update()
                    episode_stats = dict()
                    episode_stats["reward"] = current_episode_reward[i].item()
                    episode_stats.update(
                        self._extract_scalars_from_info(infos[i])
                    )
                    current_episode_reward[i] = 0
                    # use scene_id + episode_id as unique id for storing stats
                    stats_episodes[
                        (
                            current_episodes[i].scene_id,
                            current_episodes[i].episode_id,
                        )
                    ] = episode_stats

                    if 'RAW_METRICS' in config.TASK_CONFIG.TASK.MEASUREMENTS:
                        raw_metrics_episodes[
                            current_episodes[i].scene_id + '.' + 
                            current_episodes[i].episode_id
                        ] = infos[i]["raw_metrics"]


                    if len(self.config.VIDEO_OPTION) > 0:
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=os.path.basename(current_episodes[i].scene_id) + '_' + current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            tb_writer=writer,
                        )

                        rgb_frames[i] = []

                # episode continues
                elif len(self.config.VIDEO_OPTION) > 0:
                    grid_x, grid_y = to_grid(
                        map_config.coordinate_min,
                        map_config.coordinate_max,
                        map_config.global_map_size,
                        observations[i]['gps']
                    )
                    projection1 = draw_projection(observations[i]['rgb'], observations[i]['depth'] * 10, map_config.egocentric_map_size, map_config.global_map_size, map_config.coordinate_min, map_config.coordinate_max)
                    projection1 = projection1.squeeze(0).squeeze(0).permute(1, 2, 0)

                    projection1 = rotate_tensor(projection1.permute(2, 0, 1).unsqueeze(0), torch.tensor(-(observations[i]["compass"])).unsqueeze(0))
                    projection1 = projection1.squeeze(0).permute(1, 2, 0)

                    s = map_config.egocentric_map_size
                    temp = torch.max(test_global_map_visualization[i][grid_x - math.floor(s/2):grid_x + math.ceil(s/2),grid_y - math.floor(s/2):grid_y + math.ceil(s/2),:], projection1)
                    test_global_map_visualization[i][grid_x - math.floor(s/2):grid_x + math.ceil(s/2),grid_y - math.floor(s/2):grid_y + math.ceil(s/2),:] = temp
                    global_map1 = rotate_tensor(test_global_map_visualization[i][grid_x - math.floor(51/2):grid_x + math.ceil(51/2),grid_y - math.floor(51/2):grid_y + math.ceil(51/2),:].permute(2, 1, 0).unsqueeze(0), torch.tensor(-(observations[i]["compass"])).unsqueeze(0)).squeeze(0).permute(1, 2, 0).numpy()
                    egocentric_map = torch.sum(test_global_map[i, grid_x - math.floor(51/2):grid_x+math.ceil(51/2), grid_y - math.floor(51/2):grid_y + math.ceil(51/2),:], dim=2)

                    frame = observations_to_image(observations[i], projection1.data.numpy(), egocentric_map.data.cpu().numpy(), global_map1, infos[i], actions[i].cpu().numpy())
                    
                    if self.config.VIDEO_RENDER_ALL_INFO:
                        _m = self._extract_scalars_from_info(infos[i])
                        _m["reward"] = current_episode_reward[i].item()
                        frame = overlay_frame(frame, _m)
                            
                    rgb_frames[i].append(frame)

            (
                self.envs,
                test_recurrent_hidden_states,
                test_global_map,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                test_global_map,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = dict()
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum([v[stat_key] for v in stats_episodes.values()])
                / num_episodes
            )

        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar("eval/average_reward", aggregated_stats["reward"],
            step_id,
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        
        for k, v in metrics.items():
            writer.add_scalar(f"eval/{k}", v, step_id)

        ##Dump metrics JSON
        if 'RAW_METRICS' in config.TASK_CONFIG.TASK.MEASUREMENTS:
            if not os.path.exists(config.TENSORBOARD_DIR_EVAL +'/metrics'):
                os.mkdir(config.TENSORBOARD_DIR_EVAL +'/metrics')
            with open(config.TENSORBOARD_DIR_EVAL +'/metrics/' + checkpoint_path.split('/')[-1] + '.json', 'w') as fp:
                json.dump(raw_metrics_episodes, fp)
            with open(config.TENSORBOARD_DIR_EVAL +'/metrics/' + checkpoint_path.split('/')[-1] + '_overall.json', 'w') as fp:
                json.dump(aggregated_stats, fp)
            with open(config.TENSORBOARD_DIR_EVAL +'/metrics/' + checkpoint_path.split('/')[-1] + '_actions.json', 'w') as fp:
                json.dump({"actions":all_actions}, fp)

        self.envs.close()


def to_grid(coordinate_min, coordinate_max, global_map_size, position):
    grid_size = (coordinate_max - coordinate_min) / global_map_size
    grid_x = ((coordinate_max - position[0]) / grid_size).round()
    grid_y = ((position[1] - coordinate_min) / grid_size).round()
    return int(grid_x), int(grid_y)


def draw_projection(image, depth, s, global_map_size, coordinate_min, coordinate_max):
    image = torch.tensor(image).permute(2, 0, 1).unsqueeze(0)
    depth = torch.tensor(depth).permute(2, 0, 1).unsqueeze(0)
    spatial_locs, valid_inputs = _compute_spatial_locs(depth, s, global_map_size, coordinate_min, coordinate_max)
    x_gp1 = _project_to_ground_plane(image, spatial_locs, valid_inputs, s)
    
    return x_gp1


def _project_to_ground_plane(img_feats, spatial_locs, valid_inputs, s):
    outh, outw = (s, s)
    bs, f, HbyK, WbyK = img_feats.shape
    device = img_feats.device
    eps=-1e16
    K = 1

    # Sub-sample spatial_locs, valid_inputs according to img_feats resolution.
    idxes_ss = ((torch.arange(0, HbyK, 1)*K).long().to(device), \
                (torch.arange(0, WbyK, 1)*K).long().to(device))

    spatial_locs_ss = spatial_locs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 2, HbyK, WbyK)
    valid_inputs_ss = valid_inputs[:, :, idxes_ss[0][:, None], idxes_ss[1]] # (bs, 1, HbyK, WbyK)
    valid_inputs_ss = valid_inputs_ss.squeeze(1) # (bs, HbyK, WbyK)
    invalid_inputs_ss = ~valid_inputs_ss

    # Filter out invalid spatial locations
    invalid_spatial_locs = (spatial_locs_ss[:, 1] >= outh) | (spatial_locs_ss[:, 1] < 0 ) | \
                        (spatial_locs_ss[:, 0] >= outw) | (spatial_locs_ss[:, 0] < 0 ) # (bs, H, W)

    invalid_writes = invalid_spatial_locs | invalid_inputs_ss

    # Set the idxes for all invalid locations to (0, 0)
    spatial_locs_ss[:, 0][invalid_writes] = 0
    spatial_locs_ss[:, 1][invalid_writes] = 0

    # Weird hack to account for max-pooling negative feature values
    invalid_writes_f = rearrange(invalid_writes, 'b h w -> b () h w').float()
    img_feats_masked = img_feats * (1 - invalid_writes_f) + eps * invalid_writes_f
    img_feats_masked = rearrange(img_feats_masked, 'b e h w -> b e (h w)')

    # Linearize ground-plane indices (linear idx = y * W + x)
    linear_locs_ss = spatial_locs_ss[:, 1] * outw + spatial_locs_ss[:, 0] # (bs, H, W)
    linear_locs_ss = rearrange(linear_locs_ss, 'b h w -> b () (h w)')
    linear_locs_ss = linear_locs_ss.expand(-1, f, -1) # .contiguous()

    proj_feats, _ = torch_scatter.scatter_max(
                        img_feats_masked,
                        linear_locs_ss,
                        dim=2,
                        dim_size=outh*outw,
                    )
    proj_feats = rearrange(proj_feats, 'b e (h w) -> b e h w', h=outh)

    # Replace invalid features with zeros
    eps_mask = (proj_feats == eps).float()
    proj_feats = proj_feats * (1 - eps_mask) + eps_mask * (proj_feats - eps)

    return proj_feats


def _compute_spatial_locs(depth_inputs, s, global_map_size, coordinate_min, coordinate_max):
    bs, _, imh, imw = depth_inputs.shape
    local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
    cx, cy = 256./2., 256./2.
    fx = fy =  (256. / 2.) / np.tan(np.deg2rad(79. / 2.))

    #2D image coordinates
    x    = rearrange(torch.arange(0, imw), 'w -> () () () w')
    y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()')
    xx   = (x - cx) / fx
    yy   = (y - cy) / fy

    # 3D real-world coordinates (in meters)
    Z            = depth_inputs
    X            = xx * Z
    Y            = yy * Z
    # valid_inputs = (depth_inputs != 0) & ((Y < 1) & (Y > -1))
    valid_inputs = (depth_inputs != 0) & ((Y > -0.5) & (Y < 1))

    # 2D ground projection coordinates (in meters)
    # Note: map_scale - dimension of each grid in meters
    # - depth/scale + (s-1)/2 since image convention is image y downward
    # and agent is facing upwards.
    x_gp            = ( (X / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)
    y_gp            = (-(Z / local_scale) + (s-1)/2).round().long() # (bs, 1, imh, imw)

    return torch.cat([x_gp, y_gp], dim=1), valid_inputs


def rotate_tensor(x_gp, heading):
    sin_t = torch.sin(heading.squeeze(1))
    cos_t = torch.cos(heading.squeeze(1))
    A = torch.zeros(x_gp.size(0), 2, 3)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = sin_t
    A[:, 1, 0] = -sin_t
    A[:, 1, 1] = cos_t

    grid = F.affine_grid(A, x_gp.size(), align_corners=False)
    rotated_x_gp = F.grid_sample(x_gp, grid, align_corners=False)
    return rotated_x_gp

