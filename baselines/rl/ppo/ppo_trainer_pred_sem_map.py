#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import contextlib
import os
import time
import math
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional, Tuple, Union
from gym import spaces

import random
import numpy as np
from numpy import float32, ndarray, uint8
from torch import Tensor
import torch
from torch import nn
from torchvision import ops
import tqdm
from torch.optim.lr_scheduler import LambdaLR

from habitat import Config, logger, VectorEnv
from baselines.common.viz_utils import observations_to_image, save_map_image
from habitat_baselines.common.baseline_registry import baseline_registry
from baselines.common.env_utils import construct_envs
from habitat.core.environments import get_env_class
from habitat_baselines.common.tensorboard_utils import (
    TensorboardWriter,
    get_writer
)
from habitat_baselines.utils.common import (
    batch_obs,
    generate_video,
    linear_decay,
    action_array_to_dict,
    get_num_actions,
    is_continuous_action_space,
    ObservationBatchingCache,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat.tasks.utils import cartesian_to_polar

from habitat.utils import profiling_wrapper
from habitat.utils.render_wrapper import overlay_frame
from habitat_baselines.rl.ddppo.ddp_utils import (
    EXIT,
    add_signal_handlers,
    rank0_only,
    requeue_job,
    load_resume_state,
    save_resume_state,
    init_distrib_slurm,
    is_slurm_batch_job,
    get_distrib_size,
)
from habitat_baselines.common.rollout_storage import RolloutStorage
from habitat_baselines.rl.ddppo.algo import DDPPO
from baselines.rl.ppo.ppo import PPO
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
    apply_obs_transforms_obs_space,
    get_active_obs_transforms,
)
from habitat_baselines.common.base_trainer import BaseRLTrainer
from baselines.rl.ppo.policy import HierNetPolicy
from baselines.rl.ddppo.policy import (  # noqa: F401.
    PointNavResNetPolicy,
)
from multion import maps as multion_maps
from baselines.common.utils import (
    draw_projection,
    rotate_tensor,
    get_grid,
)
import torch.nn.functional as F
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from baselines.rl.models.projection import Projection, RotateTensor, get_grid
import baselines.common.depth_utils as du
import baselines.common.rotation_utils as ru
#from habitat.utils.visualizations import fog_of_war
import baselines.common.fog_of_war as fog_of_war
from baselines.common.object_detector_cyl import ObjectDetector

@baseline_registry.register_trainer(name="predsemmap")
class PredSemMapOnTrainer(BaseRLTrainer):
    r"""Trainer class for predicted semantic map
    """
    supported_tasks = ["Nav-v0"]

    SHORT_ROLLOUT_THRESHOLD: float = 0.25
    _is_distributed: bool
    _obs_batching_cache: ObservationBatchingCache
    envs: VectorEnv
    agent: PPO
    actor_critic: HierNetPolicy

    def __init__(self, config=None):
        super().__init__(config)
        self.actor_critic = None
        self.agent = None
        self.envs = None
        self.obs_transforms = []

        self._static_encoder = False
        self._encoder = None
        self._obs_space = None

        # Distributed if the world size would be
        # greater than 1
        self._is_distributed = get_distrib_size()[2] > 1
        self._obs_batching_cache = ObservationBatchingCache()

        self.using_velocity_ctrl = (
            self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
        ) == ["VELOCITY_CONTROL"]

    @property
    def obs_space(self):
        if self._obs_space is None and self.envs is not None:
            self._obs_space = self.envs.observation_spaces[0]

        return self._obs_space

    @obs_space.setter
    def obs_space(self, new_obs_space):
        self._obs_space = new_obs_space

    def _all_reduce(self, t: torch.Tensor) -> torch.Tensor:
        r"""All reduce helper method that moves things to the correct
        device and only runs if distributed
        """
        if not self._is_distributed:
            return t

        orig_device = t.device
        t = t.to(device=self.device)
        torch.distributed.all_reduce(t)

        return t.to(device=orig_device)

    def _setup_actor_critic_agent(self, ppo_cfg: Config) -> None:
        r"""Sets up actor critic and agent for PPO.

        Args:
            ppo_cfg: config node with relevant params

        Returns:
            None
        """
        logger.add_filehandler(self.config.LOG_FILE)

        policy = baseline_registry.get_policy(self.config.RL.POLICY.name)
        observation_space = self.obs_space
        self.obs_transforms = get_active_obs_transforms(self.config)
        observation_space = apply_obs_transforms_obs_space(
            observation_space, self.obs_transforms
        )

        self.actor_critic = policy.from_config(
            self.config, observation_space, self.policy_action_space
        )
        self.obs_space = observation_space
        self.actor_critic.to(self.device)

        if (
            self.config.RL.DDPPO.pretrained_encoder
            or self.config.RL.DDPPO.pretrained
        ):
            pretrained_state = torch.load(
                self.config.RL.DDPPO.pretrained_weights, map_location="cpu"
            )

        if self.config.RL.DDPPO.pretrained:
            self.actor_critic.load_state_dict(
                {  # type: ignore
                    k[len("actor_critic.") :]: v
                    for k, v in pretrained_state["state_dict"].items()
                }
            )
        elif self.config.RL.DDPPO.pretrained_encoder:
            prefix = "actor_critic.net.visual_encoder."
            self.actor_critic.net.visual_encoder.load_state_dict(
                {
                    k[len(prefix) :]: v
                    for k, v in pretrained_state["state_dict"].items()
                    if k.startswith(prefix)
                }
            )

        if not self.config.RL.DDPPO.train_encoder:
            self._static_encoder = True
            for param in self.actor_critic.net.visual_encoder.parameters():
                param.requires_grad_(False)

        if self.config.RL.DDPPO.reset_critic:
            nn.init.orthogonal_(self.actor_critic.critic.fc.weight)
            nn.init.constant_(self.actor_critic.critic.fc.bias, 0)

        self.agent = (DDPPO if self._is_distributed else PPO)(
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

    def _init_envs(self, config=None):
        if config is None:
            config = self.config

        self.envs = construct_envs(
            config,
            get_env_class(config.ENV_NAME),
            workers_ignore_signals=is_slurm_batch_job(),
        )

    def _init_train(self):
        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.config: Config = resume_state["config"]
            self.using_velocity_ctrl = (
                self.config.TASK_CONFIG.TASK.POSSIBLE_ACTIONS
            ) == ["VELOCITY_CONTROL"]

        if self.config.RL.DDPPO.force_distributed:
            self._is_distributed = True

        if is_slurm_batch_job():
            add_signal_handlers()

        if self._is_distributed:
            local_rank, tcp_store = init_distrib_slurm(
                self.config.RL.DDPPO.distrib_backend
            )
            if rank0_only():
                logger.info(
                    "Initialized DD-PPO with {} workers".format(
                        torch.distributed.get_world_size()
                    )
                )

            self.config.defrost()
            self.config.TORCH_GPU_ID = local_rank
            self.config.SIMULATOR_GPU_ID = local_rank
            # Multiply by the number of simulators to make sure they also get unique seeds
            self.config.TASK_CONFIG.SEED += (
                torch.distributed.get_rank() * self.config.NUM_ENVIRONMENTS
            )
            self.config.freeze()

            random.seed(self.config.TASK_CONFIG.SEED)
            np.random.seed(self.config.TASK_CONFIG.SEED)
            torch.manual_seed(self.config.TASK_CONFIG.SEED)
            self.num_rollouts_done_store = torch.distributed.PrefixStore(
                "rollout_tracker", tcp_store
            )
            self.num_rollouts_done_store.set("num_done", "0")

        if rank0_only() and self.config.VERBOSE:
            logger.info(f"config: {self.config}")

        profiling_wrapper.configure(
            capture_start_step=self.config.PROFILING.CAPTURE_START_STEP,
            num_steps_to_capture=self.config.PROFILING.NUM_STEPS_TO_CAPTURE,
        )

        self._init_envs()

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume ALL actions are NOT discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = None
                discrete_actions = True

        ppo_cfg = self.config.RL.PPO
        if torch.cuda.is_available():
            self.device = torch.device("cuda", self.config.TORCH_GPU_ID)
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        if rank0_only() and not os.path.isdir(self.config.CHECKPOINT_FOLDER):
            os.makedirs(self.config.CHECKPOINT_FOLDER)

        self._setup_actor_critic_agent(ppo_cfg)
        if self._is_distributed:
            self.agent.init_distributed(find_unused_params=True)  # type: ignore

        logger.info(
            "agent number of parameters: {}".format(
                sum(param.numel() for param in self.agent.parameters())
            )
        )

        obs_space = self.obs_space
        if self._static_encoder:
            self._encoder = self.actor_critic.net.visual_encoder
            obs_space = spaces.Dict(
                {
                    "visual_features": spaces.Box(
                        low=np.finfo(np.float32).min,
                        high=np.finfo(np.float32).max,
                        shape=self._encoder.output_shape,
                        dtype=np.float32,
                    ),
                    **obs_space.spaces,
                }
            )

        self._nbuffers = 2 if ppo_cfg.use_double_buffered_sampler else 1

        self.rollouts = RolloutStorage(
            ppo_cfg.num_steps,
            self.envs.num_envs,
            obs_space,
            self.policy_action_space,
            ppo_cfg.hidden_size,
            num_recurrent_layers=self.actor_critic.net.num_recurrent_layers,
            is_double_buffered=ppo_cfg.use_double_buffered_sampler,
            action_shape=action_shape,
            discrete_actions=discrete_actions,
        )
        self.rollouts.to(self.device)

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.buffers["observations"][0] = batch  # type: ignore

        self.current_episode_reward = torch.zeros(self.envs.num_envs, 1)
        self.running_episode_stats = dict(
            count=torch.zeros(self.envs.num_envs, 1),
            reward=torch.zeros(self.envs.num_envs, 1),
        )
        self.window_episode_stats = defaultdict(
            lambda: deque(maxlen=ppo_cfg.reward_window_size)
        )

        self.env_time = 0.0
        self.pth_time = 0.0
        self.t_start = time.time()

    @rank0_only
    @profiling_wrapper.RangeContext("save_checkpoint")
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

    METRICS_BLACKLIST = {"top_down_map", "collisions", "collisions.is_collision", "raw_metrics"}

    @classmethod
    def _extract_scalars_from_info(
        cls, info: Dict[str, Any]
    ) -> Dict[str, float]:
        result = {}
        for k, v in info.items():
            if not isinstance(k, str) or k in cls.METRICS_BLACKLIST:
                continue

            if isinstance(v, dict):
                result.update(
                    {
                        k + "." + subk: subv
                        for subk, subv in cls._extract_scalars_from_info(
                            v
                        ).items()
                        if isinstance(subk, str)
                        and k + "." + subk not in cls.METRICS_BLACKLIST
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

    def _compute_actions_and_step_envs(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_sample_action = time.time()

        # sample actions
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idxs[buffer_index],
                env_slice,
            ]

            profiling_wrapper.range_push("compute actions")
            (
                values,
                actions,
                actions_log_probs,
                recurrent_hidden_states,
            ) = self.actor_critic.act(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        # NB: Move actions to CPU.  If CUDA tensors are
        # sent in to env.step(), that will create CUDA contexts
        # in the subprocesses.
        # For backwards compatibility, we also call .item() to convert to
        # an int
        actions = actions.to(device="cpu")
        self.pth_time += time.time() - t_sample_action

        profiling_wrapper.range_pop()  # compute actions

        t_step_env = time.time()

        for index_env, act in zip(
            range(env_slice.start, env_slice.stop), actions.unbind(0)
        ):
            if act.shape[0] > 1:
                step_action = action_array_to_dict(
                    self.policy_action_space, act
                )
            else:
                step_action = act.item()
            self.envs.async_step_at(index_env, step_action)

        self.env_time += time.time() - t_step_env

        self.rollouts.insert(
            next_recurrent_hidden_states=recurrent_hidden_states,
            actions=actions,
            action_log_probs=actions_log_probs,
            value_preds=values,
            buffer_index=buffer_index,
        )

    def _collect_environment_result(self, buffer_index: int = 0):
        num_envs = self.envs.num_envs
        env_slice = slice(
            int(buffer_index * num_envs / self._nbuffers),
            int((buffer_index + 1) * num_envs / self._nbuffers),
        )

        t_step_env = time.time()
        outputs = [
            self.envs.wait_step_at(index_env)
            for index_env in range(env_slice.start, env_slice.stop)
        ]

        observations, rewards_l, dones, infos = [
            list(x) for x in zip(*outputs)
        ]

        self.env_time += time.time() - t_step_env

        t_update_stats = time.time()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        rewards = torch.tensor(
            rewards_l,
            dtype=torch.float,
            device=self.current_episode_reward.device,
        )
        rewards = rewards.unsqueeze(1)

        not_done_masks = torch.tensor(
            [[not done] for done in dones],
            dtype=torch.bool,
            device=self.current_episode_reward.device,
        )
        done_masks = torch.logical_not(not_done_masks)

        self.current_episode_reward[env_slice] += rewards
        current_ep_reward = self.current_episode_reward[env_slice]
        self.running_episode_stats["reward"][env_slice] += current_ep_reward.where(done_masks, current_ep_reward.new_zeros(()))  # type: ignore
        self.running_episode_stats["count"][env_slice] += done_masks.float()  # type: ignore
        for k, v_k in self._extract_scalars_from_infos(infos).items():
            v = torch.tensor(
                v_k,
                dtype=torch.float,
                device=self.current_episode_reward.device,
            ).unsqueeze(1)
            if k not in self.running_episode_stats:
                self.running_episode_stats[k] = torch.zeros_like(
                    self.running_episode_stats["count"]
                )

            self.running_episode_stats[k][env_slice] += v.where(done_masks, v.new_zeros(()))  # type: ignore

        self.current_episode_reward[env_slice].masked_fill_(done_masks, 0.0)

        if self._static_encoder:
            with torch.no_grad():
                batch["visual_features"] = self._encoder(batch)

        self.rollouts.insert(
            next_observations=batch,
            rewards=rewards,
            next_masks=not_done_masks,
            buffer_index=buffer_index,
        )

        self.rollouts.advance_rollout(buffer_index)

        self.pth_time += time.time() - t_update_stats

        return env_slice.stop - env_slice.start

    @profiling_wrapper.RangeContext("_collect_rollout_step")
    def _collect_rollout_step(self):
        self._compute_actions_and_step_envs()
        return self._collect_environment_result()

    @profiling_wrapper.RangeContext("_update_agent")
    def _update_agent(self):
        ppo_cfg = self.config.RL.PPO
        t_update_model = time.time()
        with torch.no_grad():
            step_batch = self.rollouts.buffers[
                self.rollouts.current_rollout_step_idx
            ]

            next_value = self.actor_critic.get_value(
                step_batch["observations"],
                step_batch["recurrent_hidden_states"],
                step_batch["prev_actions"],
                step_batch["masks"],
            )

        self.rollouts.compute_returns(
            next_value, ppo_cfg.use_gae, ppo_cfg.gamma, ppo_cfg.tau
        )

        self.agent.train()

        value_loss, action_loss, dist_entropy = self.agent.update(
            self.rollouts
        )

        self.rollouts.after_update()
        self.pth_time += time.time() - t_update_model

        return (
            value_loss,
            action_loss,
            dist_entropy,
        )

    def _coalesce_post_step(
        self, losses: Dict[str, float], count_steps_delta: int
    ) -> Dict[str, float]:
        stats_ordering = sorted(self.running_episode_stats.keys())
        stats = torch.stack(
            [self.running_episode_stats[k] for k in stats_ordering], 0
        )

        stats = self._all_reduce(stats)

        for i, k in enumerate(stats_ordering):
            self.window_episode_stats[k].append(stats[i])

        if self._is_distributed:
            loss_name_ordering = sorted(losses.keys())
            stats = torch.tensor(
                [losses[k] for k in loss_name_ordering] + [count_steps_delta],
                device="cpu",
                dtype=torch.float32,
            )
            stats = self._all_reduce(stats)
            count_steps_delta = int(stats[-1].item())
            stats /= torch.distributed.get_world_size()

            losses = {
                k: stats[i].item() for i, k in enumerate(loss_name_ordering)
            }

        if self._is_distributed and rank0_only():
            self.num_rollouts_done_store.set("num_done", "0")

        self.num_steps_done += count_steps_delta

        return losses

    @rank0_only
    def _training_log(
        self, writer, losses: Dict[str, float], prev_time: int = 0
    ):
        deltas = {
            k: (
                (v[-1] - v[0]).sum().item()
                if len(v) > 1
                else v[0].sum().item()
            )
            for k, v in self.window_episode_stats.items()
        }
        deltas["count"] = max(deltas["count"], 1.0)

        writer.add_scalar(
            "reward",
            deltas["reward"] / deltas["count"],
            self.num_steps_done,
        )

        # Check to see if there are any metrics
        # that haven't been logged yet
        metrics = {
            k: v / deltas["count"]
            for k, v in deltas.items()
            if k not in {"reward", "count"}
        }

        for k, v in metrics.items():
            writer.add_scalar(f"metrics/{k}", v, self.num_steps_done)
        for k, v in losses.items():
            writer.add_scalar(f"losses/{k}", v, self.num_steps_done)

        fps = self.num_steps_done / ((time.time() - self.t_start) + prev_time)
        writer.add_scalar("metrics/fps", fps, self.num_steps_done)

        # log stats
        if self.num_updates_done % self.config.LOG_INTERVAL == 0:
            logger.info(
                "update: {}\tfps: {:.3f}\t".format(
                    self.num_updates_done,
                    fps,
                )
            )

            logger.info(
                "update: {}\tenv-time: {:.3f}s\tpth-time: {:.3f}s\t"
                "frames: {}".format(
                    self.num_updates_done,
                    self.env_time,
                    self.pth_time,
                    self.num_steps_done,
                )
            )

            logger.info(
                "Average window size: {}  {}".format(
                    len(self.window_episode_stats["count"]),
                    "  ".join(
                        "{}: {:.3f}".format(k, v / deltas["count"])
                        for k, v in deltas.items()
                        if k != "count"
                    ),
                )
            )

    def should_end_early(self, rollout_step) -> bool:
        if not self._is_distributed:
            return False
        # This is where the preemption of workers happens.  If a
        # worker detects it will be a straggler, it preempts itself!
        return (
            rollout_step
            >= self.config.RL.PPO.num_steps * self.SHORT_ROLLOUT_THRESHOLD
        ) and int(self.num_rollouts_done_store.get("num_done")) >= (
            self.config.RL.DDPPO.sync_frac * torch.distributed.get_world_size()
        )

    @profiling_wrapper.RangeContext("train")
    def train(self) -> None:
        r"""Main method for training DD/PPO.

        Returns:
            None
        """

        self._init_train()

        count_checkpoints = 0
        prev_time = 0

        lr_scheduler = LambdaLR(
            optimizer=self.agent.optimizer,
            lr_lambda=lambda x: 1 - self.percent_done(),
        )

        resume_state = load_resume_state(self.config)
        if resume_state is not None:
            self.agent.load_state_dict(resume_state["state_dict"])
            self.agent.optimizer.load_state_dict(resume_state["optim_state"])
            lr_scheduler.load_state_dict(resume_state["lr_sched_state"])

            requeue_stats = resume_state["requeue_stats"]
            self.env_time = requeue_stats["env_time"]
            self.pth_time = requeue_stats["pth_time"]
            self.num_steps_done = requeue_stats["num_steps_done"]
            self.num_updates_done = requeue_stats["num_updates_done"]
            self._last_checkpoint_percent = requeue_stats[
                "_last_checkpoint_percent"
            ]
            count_checkpoints = requeue_stats["count_checkpoints"]
            prev_time = requeue_stats["prev_time"]

            self.running_episode_stats = requeue_stats["running_episode_stats"]
            self.window_episode_stats.update(
                requeue_stats["window_episode_stats"]
            )

        ppo_cfg = self.config.RL.PPO

        with (
            get_writer(self.config, flush_secs=self.flush_secs)
            if rank0_only()
            else contextlib.suppress()
        ) as writer:
            while not self.is_done():
                profiling_wrapper.on_start_step()
                profiling_wrapper.range_push("train update")

                if ppo_cfg.use_linear_clip_decay:
                    self.agent.clip_param = ppo_cfg.clip_param * (
                        1 - self.percent_done()
                    )

                if rank0_only() and self._should_save_resume_state():
                    requeue_stats = dict(
                        env_time=self.env_time,
                        pth_time=self.pth_time,
                        count_checkpoints=count_checkpoints,
                        num_steps_done=self.num_steps_done,
                        num_updates_done=self.num_updates_done,
                        _last_checkpoint_percent=self._last_checkpoint_percent,
                        prev_time=(time.time() - self.t_start) + prev_time,
                        running_episode_stats=self.running_episode_stats,
                        window_episode_stats=dict(self.window_episode_stats),
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

                self.agent.eval()
                count_steps_delta = 0
                profiling_wrapper.range_push("rollouts loop")

                profiling_wrapper.range_push("_collect_rollout_step")
                for buffer_index in range(self._nbuffers):
                    self._compute_actions_and_step_envs(buffer_index)

                for step in range(ppo_cfg.num_steps):
                    is_last_step = (
                        self.should_end_early(step + 1)
                        or (step + 1) == ppo_cfg.num_steps
                    )

                    for buffer_index in range(self._nbuffers):
                        count_steps_delta += self._collect_environment_result(
                            buffer_index
                        )

                        if (buffer_index + 1) == self._nbuffers:
                            profiling_wrapper.range_pop()  # _collect_rollout_step

                        if not is_last_step:
                            if (buffer_index + 1) == self._nbuffers:
                                profiling_wrapper.range_push(
                                    "_collect_rollout_step"
                                )

                            self._compute_actions_and_step_envs(buffer_index)

                    if is_last_step:
                        break

                profiling_wrapper.range_pop()  # rollouts loop

                if self._is_distributed:
                    self.num_rollouts_done_store.add("num_done", 1)

                (
                    value_loss,
                    action_loss,
                    dist_entropy,
                ) = self._update_agent()

                if ppo_cfg.use_linear_lr_decay:
                    lr_scheduler.step()  # type: ignore

                self.num_updates_done += 1
                losses = self._coalesce_post_step(
                    dict(
                        value_loss=value_loss,
                        action_loss=action_loss,
                        entropy=dist_entropy,
                    ),
                    count_steps_delta,
                )

                self._training_log(writer, losses, prev_time)

                # checkpoint model
                if rank0_only() and self.should_checkpoint():
                    self.save_checkpoint(
                        f"ckpt.{count_checkpoints}.pth",
                        dict(
                            step=self.num_steps_done,
                            wall_time=(time.time() - self.t_start) + prev_time,
                        ),
                    )
                    count_checkpoints += 1

                profiling_wrapper.range_pop()  # train update

            self.envs.close()


    def _pause_envs(
        self,
        envs_to_pause: List[int],
        envs: VectorEnv,
        test_recurrent_hidden_states: Tensor,
        not_done_masks: Tensor,
        current_episode_reward: Tensor,
        prev_actions: Tensor,
        batch: Dict[str, Tensor],
        rgb_frames: Union[List[List[Any]], List[List[ndarray]]],
        goal_observations: Tensor,
        global_object_map: Tensor,
        is_goal: Tensor,
        grid_map: Tensor,
        object_maps: Tensor,
        #pred_ious: Tensor,
    ) -> Tuple[
        VectorEnv,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Dict[str, Tensor],
        List[List[Any]],
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        #Tensor,
    ]:
        # pausing self.envs with no new episode
        if len(envs_to_pause) > 0:
            state_index = list(range(envs.num_envs))
            for idx in reversed(envs_to_pause):
                state_index.pop(idx)
                envs.pause_at(idx)

            # indexing along the batch dimensions
            test_recurrent_hidden_states = test_recurrent_hidden_states[
                state_index
            ]
            not_done_masks = not_done_masks[state_index]
            current_episode_reward = current_episode_reward[state_index]
            prev_actions = prev_actions[state_index]
            goal_observations = goal_observations[state_index]
            global_object_map = global_object_map[state_index]
            is_goal = is_goal[state_index]
            grid_map = grid_map[state_index]
            object_maps = object_maps[state_index]
            #pred_ious = pred_ious[state_index]

            for k, v in batch.items():
                batch[k] = v[state_index]

            rgb_frames = [rgb_frames[i] for i in state_index]

        return (
            envs,
            test_recurrent_hidden_states,
            not_done_masks,
            current_episode_reward,
            prev_actions,
            batch,
            rgb_frames,
            goal_observations,
            global_object_map,
            is_goal,
            grid_map,
            object_maps,
            #pred_ious
        )

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
        if self._is_distributed:
            raise RuntimeError("Evaluation does not support distributed mode")

        # Map location CPU is almost always better than mapping to a CUDA device.
        if self.config.EVAL.SHOULD_LOAD_CKPT:
            ckpt_dict = self.load_checkpoint(
                checkpoint_path, map_location="cpu"
            )
        else:
            ckpt_dict = {}

        if self.config.EVAL.USE_CKPT_CONFIG:
            config = self._setup_eval_config(ckpt_dict["config"])
        else:
            config = self.config.clone()

        ppo_cfg = config.RL.PPO

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = config.EVAL.SPLIT
        config.freeze()

        if (
            len(self.config.VIDEO_OPTION) > 0
            and self.config.VIDEO_RENDER_TOP_DOWN
        ):
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()

        if self.config.RL.POLICY.EXPLORATION_STRATEGY == "stubborn":
            config.defrost()
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
            config.freeze()
            
        if config.VERBOSE:
            logger.info(f"env config: {config}")

        self._init_envs(config)

        action_space = self.envs.action_spaces[0]
        if self.using_velocity_ctrl:
            # For navigation using a continuous action space for a task that
            # may be asking for discrete actions
            self.policy_action_space = action_space["VELOCITY_CONTROL"]
            action_shape = (2,)
            discrete_actions = False
        else:
            self.policy_action_space = action_space
            if is_continuous_action_space(action_space):
                # Assume NONE of the actions are discrete
                action_shape = (get_num_actions(action_space),)
                discrete_actions = False
            else:
                # For discrete pointnav
                action_shape = (1,)
                discrete_actions = True

        self._setup_actor_critic_agent(ppo_cfg)

        if self.agent.actor_critic.should_load_agent_state:
            self.agent.load_state_dict(ckpt_dict["state_dict"], strict=False)
        self.actor_critic = self.agent.actor_critic

        observations = self.envs.reset()
        batch = batch_obs(
            observations, device=self.device, cache=self._obs_batching_cache
        )
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

        current_episode_reward = torch.zeros(
            self.envs.num_envs, 1, device="cpu"
        )

        test_recurrent_hidden_states = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.actor_critic.num_recurrent_layers,
            ppo_cfg.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            *action_shape,
            device=self.device,
            dtype=torch.long if discrete_actions else torch.float,
        )
        if self.config.RL.SEM_MAP_POLICY.use_world_loc:
            goal_world_coordinates = torch.zeros(
                self.config.NUM_ENVIRONMENTS,
                3,  # 3D coordinates 
                device=self.device,
                dtype=torch.float,
            )
        else:
            goal_world_coordinates = torch.zeros(
                self.config.NUM_ENVIRONMENTS,
                2,  # 2D episodic coordinates 
                device=self.device,
                dtype=torch.float,
            )
        goal_observations = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            2,  # polar coordinates
            device=self.device,
            dtype=torch.float,
        )
        goal_grid = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            2,
            device=self.device,
            dtype=torch.float
        )
        is_goal = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.long,
        )
        steps_towards_short_term_goal = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.long,
        )
        collision_threshold_steps = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.long,
        )
        global_object_map = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.config.RL.SEM_MAP_POLICY.global_map_size,
            self.config.RL.SEM_MAP_POLICY.global_map_size,
            self.config.RL.SEM_MAP_POLICY.MAP_CHANNELS,
            device=self.device,
            dtype=torch.float,
        )
        
        not_done_masks = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            1,
            device=self.device,
            dtype=torch.bool,
        )
        stubborn_goal_queues = [[] for _ in range(self.config.NUM_ENVIRONMENTS)]
        pred_ious = [[] for _ in range(self.config.NUM_ENVIRONMENTS)]
        
        ##Depth
        self.camera = du.get_camera_matrix(
                        self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT, 
                        self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 
                        self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV)
        self.elevation = 0. #np.rad2deg(env_config.SIMULATOR.DEPTH_SENSOR.ORIENTATION[0])
        self.camera_height = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        # Init map related location info
        self.map_resolution = self.config.RL.SEM_MAP_POLICY.map_resolution
        self.meters_covered = self.config.RL.SEM_MAP_POLICY.meters_covered
        self.map_grid_size = np.round(self.meters_covered / self.map_resolution).astype(int)
        self.map_center = np.array([np.round(self.map_grid_size/2.).astype(int), np.round(self.map_grid_size/2.).astype(int)])
        self.grid_map = np.zeros([self.config.NUM_ENVIRONMENTS, self.map_grid_size, self.map_grid_size, 2], dtype=uint8)
        #self.grid_map[:,:,0].fill(0.5)
        #self.fog_of_war_mask = np.zeros([self.map_grid_size, self.map_grid_size], dtype=uint8)
        # Init 2D map slicing height info
        # 2D map will display heights between: 
        #   [elevation+slice_range_below, elevation+slice_range_above]
        self.slice_range_below = -1 # Should be 0 or negative
        self.slice_range_above = 10.5 # Should be 0 or positive
        self.z_bins = [0.5, 1.5]
        
        ##
        
        self.object_maps = torch.zeros(
            self.config.NUM_ENVIRONMENTS,
            self.map_grid_size,
            self.map_grid_size,
            self.config.RL.SEM_MAP_POLICY.MAP_CHANNELS,
            device=self.device,
            dtype=torch.float,
        )
        
        # Assume map size with configurable params
        oracle_map_size = torch.zeros(self.envs.num_envs, 3, 2)
        oracle_map_size[:, 0, :] = torch.tensor([self.map_grid_size,
                                                self.map_grid_size], device=self.device)
        oracle_map_size[:, 1, :] = torch.tensor([self.config.RL.SEM_MAP_POLICY.coordinate_min, 
                    self.config.RL.SEM_MAP_POLICY.coordinate_min], device=self.device)
        # 
        
        self._detector = ObjectDetector()
        
        stats_episodes: Dict[
            Any, Any
        ] = {}  # dict of dicts that stores stats per episode

        rgb_frames = [
            [] for _ in range(self.config.NUM_ENVIRONMENTS)
        ]  # type: List[List[np.ndarray]]
        if len(self.config.VIDEO_OPTION) > 0:
            os.makedirs(self.config.VIDEO_DIR, exist_ok=True)

        number_of_eval_episodes = self.config.TEST_EPISODE_COUNT
        if number_of_eval_episodes == -1:
            number_of_eval_episodes = sum(self.envs.number_of_episodes)
        else:
            total_num_eps = sum(self.envs.number_of_episodes)
            if total_num_eps < number_of_eval_episodes:
                logger.warn(
                    f"Config specified {number_of_eval_episodes} eval episodes"
                    ", dataset only has {total_num_eps}."
                )
                logger.warn(f"Evaluating with {total_num_eps} instead.")
                number_of_eval_episodes = total_num_eps

        pbar = tqdm.tqdm(total=number_of_eval_episodes)
        self.actor_critic.eval()
        while (
            len(stats_episodes) < number_of_eval_episodes
            and self.envs.num_envs > 0
        ):
            current_episodes = self.envs.current_episodes()
            
            next_goal_category = batch['multiobjectgoal']
            
            # 1) Get map with all objects
            self.object_maps, agent_locs, semantic, _pred_ious = self.build_map(batch, self.object_maps, self.grid_map)
                
            for i in range(self.envs.num_envs):
                pred_ious[i].append(_pred_ious[i].item())
                # mark agent
                self.object_maps[i, :, :, 2] = 0
                #self.object_maps[i, agent_locs[i, 0], agent_locs[i, 1], 2] = 10
                self.object_maps[
                    i,
                    int(max(0, agent_locs[i, 0] - self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)):
                        int(min(self.map_grid_size, agent_locs[i, 0] + self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)),
                    int(max(0, agent_locs[i, 1] - self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)):
                        int(min(self.map_grid_size, agent_locs[i, 1] + self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)),
                    2] = 10
                
                # Perform steps 2-4 only when the agent is starting or a goal is reached
                if prev_actions[i].item() == 0:
                    
                    steps_towards_short_term_goal[i] = 0    # reset step counter
                    collision_threshold_steps[i] = 0        # reset collision counter
                    
                    # 2) Find next goal position on the map from map and goal category
                    if self.config.RL.SEM_MAP_POLICY.use_oracle_map:
                        goal_grid_loc = ((self.object_maps[i, :, :, 1] - self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_ind_offset)==next_goal_category[i].item()).nonzero(as_tuple=False)
                    else:
                        goal_grid_loc = (self.object_maps[i, :, :, 1]==next_goal_category[i].item()).nonzero(as_tuple=False)
                    
                    # 2.i) If the goal is not visible, then the agent explores
                    if len(goal_grid_loc) == 0:
                        # get agent location on the map
                        agent_grid_loc = self.object_maps[i, :, :, 2].nonzero(as_tuple=False)
                        if len(agent_grid_loc) > 0:
                            agent_grid_loc = agent_grid_loc[0]
                        else:
                            agent_grid_loc = [0, 0]
                        
                        if self.config.RL.POLICY.EXPLORATION_STRATEGY == "stubborn":
                            # Stubborn Exploration strategy
                            # https://github.com/Improbable-AI/Stubborn
                            if self.config.RL.POLICY.USE_LOCAL_MAP_FOR_STUBBORN:
                                local_size = self.config.RL.POLICY.local_map_size
                                
                                if len(stubborn_goal_queues[i]) == 0:
                                    stubborn_goal_queues[i].append(1)
                                    stubborn_goal_queues[i].append(2)
                                    stubborn_goal_queues[i].append(3)
                                    stubborn_goal_queues[i].append(4)
                                    
                                _goal_dir = stubborn_goal_queues[i].pop(0)
                                if _goal_dir == 1:
                                    goal_grid_loc = torch.tensor([min(agent_grid_loc[0]+local_size,oracle_map_size[i][0][0]), min(agent_grid_loc[1]+local_size, oracle_map_size[i][0][1])], device=self.device)
                                elif _goal_dir == 2:
                                    goal_grid_loc = torch.tensor([min(agent_grid_loc[0]+local_size, oracle_map_size[i][0][0]), max(agent_grid_loc[1]-local_size, 0)], device=self.device)
                                elif _goal_dir == 3:
                                    goal_grid_loc = torch.tensor([max(agent_grid_loc[0]-local_size, 0), max(agent_grid_loc[1]-local_size, 0)], device=self.device)
                                else:
                                    goal_grid_loc = torch.tensor([max(agent_grid_loc[0]-local_size, 0), min(agent_grid_loc[1]+local_size, oracle_map_size[i][0][1])], device=self.device)
                                    
                                stubborn_goal_queues[i].append(_goal_dir)
                            else: 
                                if len(stubborn_goal_queues[i]) == 0:
                                    local_w, local_h = oracle_map_size[i][0]
                                    stubborn_goal_queues[i].append(torch.tensor([local_w, local_h], device=self.device))
                                    stubborn_goal_queues[i].append(torch.tensor([local_w, 0], device=self.device))
                                    stubborn_goal_queues[i].append(torch.tensor([0, 0], device=self.device))
                                    stubborn_goal_queues[i].append(torch.tensor([0, local_h], device=self.device))

                                _goal = stubborn_goal_queues[i].pop(0)
                                goal_grid_loc = _goal
                                stubborn_goal_queues[i].append(_goal)
                            
                        else:
                            # Random Exploration Strategy: selecting a random location around the agent
                            explore_radius = self.config.RL.POLICY.EXPLORE_RADIUS
                            
                            # sample a point around agent
                            goal_grid_loc = torch.tensor([
                                        random.randint(max(0, agent_grid_loc[0]-explore_radius), min(oracle_map_size[i][0][0], agent_grid_loc[0]+explore_radius)),
                                        random.randint(max(0, agent_grid_loc[1]-explore_radius), min(oracle_map_size[i][0][1], agent_grid_loc[1]+explore_radius))],
                                        device=self.device)
                        
                        is_goal[i] = 0
                        
                    else:
                        is_goal[i] = 1
                        goal_grid_loc = goal_grid_loc.float().mean(axis=0).type(torch.uint8)  # select one
                    
                    goal_grid[i] = goal_grid_loc
                    
                    # 3) Convert map position to world position in 3D
                    _locs = self.from_grid(goal_grid_loc[0], goal_grid_loc[1])
                    if self.config.RL.SEM_MAP_POLICY.use_world_loc:
                        goal_world_coordinates[i] = torch.stack([_locs[1], torch.zeros_like(_locs[0]), -_locs[0]], axis=-1)
                    else:
                        goal_world_coordinates[i] = torch.stack([_locs[0], _locs[1]], axis=-1)
                
                goal_grid_loc = goal_grid[i]
                    
                # Mark the sampled goal on the map for visualization
                self.object_maps[i, :, :, 3] = 0 # reset goal position
                self.object_maps[i, 
                            int(max(0, goal_grid_loc[0]-self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)): int(min(oracle_map_size[i][0][0], goal_grid_loc[0]+self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)), 
                            int(max(0, goal_grid_loc[1]-self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)): int(min(oracle_map_size[i][0][1], goal_grid_loc[1]+self.config.TASK_CONFIG.TASK.OBJECT_MAP_SENSOR.object_padding)), 
                            3] = 11  # num of objects=8, agent marked as 10
                if is_goal[i] == 0:
                    steps_towards_short_term_goal[i] += 1
                
                # 4) Convert 3D world positions to polar coordinates
                if self.config.RL.SEM_MAP_POLICY.use_world_loc:
                    agent_position = batch[i]["agent_position"]
                    agent_rotation = batch[i]["agent_rotation"]
                    
                    # convert goal location to goal world location
                    goal_position = goal_world_coordinates[i].cpu().numpy()
                    g_position_world = quaternion_rotate_vector(
                        quaternion_from_coeff(current_episodes[i].start_rotation), goal_position
                    ) + current_episodes[i].start_position
                    goal_observations[i] = torch.tensor(multion_maps.compute_pointgoal(
                                        agent_position.cpu().numpy(), 
                                        agent_rotation.cpu().numpy(), 
                                        g_position_world), device=self.device)
                    
                else:
                    agent_position = batch[i]["episodic_gps"]
                    agent_rotation = batch[i]["episodic_rotation"]  # quaternion version of episodic_compass
                    goal_observations[i] = torch.tensor(multion_maps.compute_pointgoal(
                                            agent_position.cpu().numpy(), 
                                            agent_rotation.cpu().numpy(), 
                                            goal_world_coordinates[i].cpu().numpy()), device=self.device)
                
                # 5) Move to the goal
            with torch.no_grad():
                (
                    _,
                    actions,
                    _,
                    test_recurrent_hidden_states,
                ) = self.actor_critic.act(
                    batch,
                    test_recurrent_hidden_states,
                    prev_actions,
                    not_done_masks,
                    goal_observations,
                    deterministic=False,
                )

                prev_actions.copy_(actions)  # type: ignore
            # NB: Move actions to CPU.  If CUDA tensors are
            # sent in to env.step(), that will create CUDA contexts
            # in the subprocesses.
            # For backwards compatibility, we also call .item() to convert to
            # an int
            step_data = [{"action": a.item(), "action_args": {"is_goal": is_goal[i].item()}} for i,a in enumerate(actions.to(device="cpu"))]

            outputs = self.envs.step(step_data)

            observations, rewards_l, dones, infos = [
                list(x) for x in zip(*outputs)
            ]
            batch = batch_obs(  # type: ignore
                observations,
                device=self.device,
                cache=self._obs_batching_cache,
            )
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)  # type: ignore

            not_done_masks = torch.tensor(
                [[not done] for done in dones],
                dtype=torch.bool,
                device="cpu",
            )

            rewards = torch.tensor(
                rewards_l, dtype=torch.float, device="cpu"
            ).unsqueeze(1)
            current_episode_reward += rewards
            next_episodes = self.envs.current_episodes()
            envs_to_pause = []
            n_envs = self.envs.num_envs
            for i in range(n_envs):
                if (
                    next_episodes[i].scene_id,
                    next_episodes[i].episode_id,
                ) in stats_episodes:
                    envs_to_pause.append(i)

                # episode ended
                if not not_done_masks[i].item():
                    pbar.update()
                    episode_stats = {
                        "reward": current_episode_reward[i].item()
                    }
                    infos[i].update(
                        {"pred_iou": np.array(pred_ious[i]).mean()}
                    )
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

                    if len(self.config.VIDEO_OPTION) > 0:
                        frame = observations_to_image(
                                observation=batch[i], info=infos[i], action=actions[i].cpu().numpy(),
                                object_map=self.object_maps[i],
                                predicted_semantic=semantic[i], 
                                #semantic_projections=(self.map > 0), #projection[i], 
                                #global_object_map=global_object_map[i], 
                                #agent_view=agent_view[i],
                                config=self.config
                        )
                        if self.config.VIDEO_RENDER_ALL_INFO:
                            _m = self._extract_scalars_from_info(infos[i])
                            _m["reward"] = current_episode_reward[i].item()
                            _m["next_goal"] = multion_maps.MULTION_CYL_OBJECT_MAP[next_goal_category[i].item()]
                            if self.config.RL.POLICY.EXPLORATION_STRATEGY == "stubborn":
                                _m["collision_count"] = collision_threshold_steps[i].item()
                            frame = overlay_frame(frame, _m)

                        rgb_frames[i].append(frame)
                        
                        generate_video(
                            video_option=self.config.VIDEO_OPTION,
                            video_dir=self.config.VIDEO_DIR,
                            images=rgb_frames[i],
                            episode_id=os.path.basename(current_episodes[i].scene_id) + '_' + current_episodes[i].episode_id,
                            checkpoint_idx=checkpoint_index,
                            metrics=self._extract_scalars_from_info(infos[i]),
                            fps=self.config.VIDEO_FPS,
                            tb_writer=writer,
                            keys_to_include_in_name=self.config.EVAL_KEYS_TO_INCLUDE_IN_NAME,
                        )

                        rgb_frames[i] = []
                    
                    global_object_map[i] = torch.zeros(
                        self.config.RL.SEM_MAP_POLICY.global_map_size,
                        self.config.RL.SEM_MAP_POLICY.global_map_size,
                        self.config.RL.SEM_MAP_POLICY.MAP_CHANNELS,
                        device=self.device,
                        dtype=torch.float,
                    )
                    self.object_maps[i] = torch.zeros(
                        self.map_grid_size,
                        self.map_grid_size,
                        self.config.RL.SEM_MAP_POLICY.MAP_CHANNELS,
                        device=self.device,
                        dtype=torch.float,
                    )
                    stubborn_goal_queues[i] = []
                    if self.config.RL.SEM_MAP_POLICY.use_world_loc:
                        goal_world_coordinates[i] = torch.zeros(
                            3,  # 3D coordinates 
                            device=self.device,
                            dtype=torch.float,
                        )
                    else:
                        goal_world_coordinates[i] = torch.zeros(
                            2,  # 2D episodic coordinates 
                            device=self.device,
                            dtype=torch.float,
                        )
                    goal_observations[i] = torch.zeros(
                        2,  # polar coordinates
                        device=self.device,
                        dtype=torch.float,
                    )
                    goal_grid[i] = torch.zeros(
                        2,
                        device=self.device,
                        dtype=torch.float
                    )
                    is_goal[i] = 0
                    collision_threshold_steps[i] = 0
                    self.grid_map[i,:,:,:] = np.zeros([self.map_grid_size, self.map_grid_size, 2], dtype=uint8)
                    pred_ious[i] = []

                # episode continues
                else:
                    if len(self.config.VIDEO_OPTION) > 0:
                        frame = observations_to_image(
                                observation=observations[i], info=infos[i], action=actions[i].cpu().numpy(),
                                object_map=self.object_maps[i], 
                                predicted_semantic=semantic[i],
                                #semantic_projections=(self.map > 0), #projection[i], 
                                #global_object_map=global_object_map[i], 
                                #agent_view=agent_view[i],
                                config=self.config
                        )
                        if self.config.VIDEO_RENDER_ALL_INFO:
                            _m = self._extract_scalars_from_info(infos[i])
                            _m["reward"] = current_episode_reward[i].item()
                            _m["next_goal"] = multion_maps.MULTION_CYL_OBJECT_MAP[next_goal_category[i].item()]
                            if self.config.RL.POLICY.EXPLORATION_STRATEGY == "stubborn":
                                _m["collision_count"] = collision_threshold_steps[i].item()
                            frame = overlay_frame(frame, _m)

                        rgb_frames[i].append(frame)

                    if self.config.RL.POLICY.EXPLORATION_STRATEGY == "stubborn" and infos[i]["collisions"]["is_collision"]:
                        collision_threshold_steps[i] += 1
                    else:
                        collision_threshold_steps[i] = 0

                    if ((actions[i].item() == 0 and infos[i]["sub_success"] == 1) or
                            is_goal[i] == 0 and (
                            ((self.config.RL.POLICY.EXPLORATION_STRATEGY == "" or self.config.RL.POLICY.EXPLORATION_STRATEGY == "random") 
                                and steps_towards_short_term_goal[i].item() >= self.config.RL.POLICY.MAX_STEPS_BEFORE_GOAL_SELECTION) or
                            (self.config.RL.POLICY.EXPLORATION_STRATEGY == "stubborn" and 
                                (infos[i]["collisions"]["is_collision"] and
                                    collision_threshold_steps[i] > self.config.RL.POLICY.collision_threshold) 
                                or steps_towards_short_term_goal[i].item() >= self.config.RL.POLICY.MAX_STEPS_BEFORE_GOAL_SELECTION))):
                        
                        test_recurrent_hidden_states[i] = torch.zeros(
                            self.actor_critic.num_recurrent_layers,
                            ppo_cfg.hidden_size,
                            device=self.device,
                        )
                        prev_actions[i] = torch.zeros(
                            *action_shape,
                            device=self.device,
                            dtype=torch.long if discrete_actions else torch.float,
                        )

            not_done_masks = not_done_masks.to(device=self.device)
            (
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                goal_observations,
                global_object_map,
                is_goal,
                self.grid_map,
                self.object_maps,
                #pred_ious
            ) = self._pause_envs(
                envs_to_pause,
                self.envs,
                test_recurrent_hidden_states,
                not_done_masks,
                current_episode_reward,
                prev_actions,
                batch,
                rgb_frames,
                goal_observations,
                global_object_map,
                is_goal,
                self.grid_map,
                self.object_maps,
                #pred_ious
            )

        num_episodes = len(stats_episodes)
        aggregated_stats = {}
        for stat_key in next(iter(stats_episodes.values())).keys():
            aggregated_stats[stat_key] = (
                sum(v[stat_key] for v in stats_episodes.values())
                / num_episodes
            )

        logger.info(f"Number of episodes evaluated: {num_episodes}")
        for k, v in aggregated_stats.items():
            logger.info(f"Average episode {k}: {v:.4f}")

        step_id = checkpoint_index
        if "extra_state" in ckpt_dict and "step" in ckpt_dict["extra_state"]:
            step_id = ckpt_dict["extra_state"]["step"]

        writer.add_scalar(
            "eval_reward/average_reward", aggregated_stats["reward"], step_id
        )

        metrics = {k: v for k, v in aggregated_stats.items() if k != "reward"}
        for k, v in metrics.items():
            writer.add_scalar(f"eval_metrics/{k}", v, step_id)

        self.envs.close()

    def build_map(self, observations, object_maps, grid_map):
        depth = (observations['depth'] * 10).squeeze(-1).cpu().numpy()
        #depth = (observations['depth']).squeeze(-1).cpu().numpy()
        depth[depth == 0] = np.NaN
        #depth[depth > 10] = np.NaN

        gt_semantic = observations['semantic'].squeeze(-1).cpu().numpy()
        theta = observations['episodic_compass'].cpu().numpy()
        location = observations["episodic_gps"].cpu().numpy()
        semantic = np.zeros_like(depth)
        results = self._detector.predict(observations['rgb'].squeeze(-1))
        pred_ious = torch.zeros(self.config.NUM_ENVIRONMENTS)
        for i, res in enumerate(results):
            if len(res['boxes'])> 0:
                gt_bbox = []
                for j, b in enumerate(res['boxes']):
                    b = np.round(b).astype(int)
                    semantic[i, b[1]:b[3], b[0]:b[2]] = res["labels"][j] + 1  # object semantic labels start at 1
                    
                    # get GT bbox
                    if (res["labels"][j] + 1) in gt_semantic[i]:
                        gt_locs = (gt_semantic[i] * (gt_semantic[i] == (res["labels"][j] + 1))).nonzero()
                        gt_bbox.append(torch.tensor([np.min(gt_locs[1]), np.min(gt_locs[0]), np.max(gt_locs[1]), np.max(gt_locs[0])]))
                
                if len(gt_bbox) > 0:
                    pred_ious[i] = ops.box_iou(torch.stack(gt_bbox), torch.tensor(res['boxes'])).mean()
                    
        semantic = semantic[..., np.newaxis]
        
        coords = self._unproject_to_world(depth, location, theta)
        grid_map = self._add_to_map(coords, semantic, grid_map)
        _agent_locs = self.to_grid(location)
        object_maps[:, :, :, :2] = torch.tensor(grid_map)
        
        return object_maps, _agent_locs, semantic, pred_ious
    
    def _unproject_to_world(self, depth, location, theta):
        point_cloud = du.get_point_cloud_from_z(depth, self.camera)

        agent_view = du.transform_camera_view(point_cloud,
                                              self.camera_height, self.elevation)

        geocentric_pc = du.transform_pose(agent_view, location, theta)

        return geocentric_pc
    
    def _add_to_map(self, coords, semantic, grid_map):
        XYZS = np.concatenate((coords, semantic),axis=-1)
        depth_counts, sem_map_counts = du.bin_points_w_sem(
            XYZS,
            self.map_grid_size,
            self.z_bins,
            self.map_resolution,
            self.map_center)

        map = grid_map[:, :, :, 0] + depth_counts[:, :, :, 1]
        map[map < 1] = 0.0
        map[map >= 1] = 1.0
        grid_map[:, :, :, 0] = map
        
        grid_map[:, :, :, 1] = np.maximum(grid_map[:, :, :, 1], sem_map_counts[:, :, :, 1])
        #grid_map[:, :, :, 1] = sem_map_counts[:, :, :, 1]

        return grid_map
        
    def to_grid(self, xy):
        return (np.round(xy / self.map_resolution) + self.map_center).astype(int)
        
    def from_grid(self, grid_x, grid_y):
        return [
            (grid_x - self.map_center[0]) * self.map_resolution,
            (grid_y - self.map_center[1]) * self.map_resolution,
            ]