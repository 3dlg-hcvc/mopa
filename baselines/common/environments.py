#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
r"""
This file hosts task-specific or trainer-specific environments for trainers.
All environments here should be a (direct or indirect ) subclass of Env class
in habitat. Customized environments should be registered using
``@baseline_registry.register_env(name="myEnv")` for reusability
"""

from typing import Optional, Type, Union, Dict, Any

import numpy as np
import habitat
from habitat import Config, Dataset
from habitat.datasets import make_dataset
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

def get_env_class(env_name: str) -> Type[habitat.RLEnv]:
    r"""Return environment class based on name.

    Args:
        env_name: name of the environment.

    Returns:
        Type[habitat.RLEnv]: env class.
    """
    return baseline_registry.get_env(env_name)

@baseline_registry.register_env(name="NavRLEnv")
@baseline_registry.register_env(name="MultiObjNavRLEnv")
class MultiObjNavRLEnv(habitat.RLEnv):
    def __init__(self, config: Config, dataset: Optional[Dataset] = None):
        self._rl_config = config.RL
        self._core_env_config = config.TASK_CONFIG
        self._reward_measure_name = self._rl_config.REWARD_MEASURE
        self._success_measure_name = self._rl_config.SUCCESS_MEASURE
        self._subsuccess_measure_name = self._rl_config.SUBSUCCESS_MEASURE

        self._previous_measure = None
        self._previous_action = None
        
        try:
            from multion import actions, measures, sensors
            from multion.config.default import get_extended_config
            from multion.task import MultiObjectNavDatasetV1
        except ImportError as e:
            nav_import_error = e

            @registry.register_dataset(name="MultiObjectNav-v1")
            class MultiObjectNavDatasetV1(PointNavDatasetV1):
                def __init__(self, *args, **kwargs):
                    raise nav_import_error
        
        super().__init__(self._core_env_config, dataset)
        
        self.task = self._env.task

    def reset(self):
        self._previous_action = None
        self.task.current_goal_index = 0
        observations = super().reset()
        self._previous_measure = self._env.get_metrics()[
            self._reward_measure_name
        ]
        return observations

    def step(self, action: Union[int, str, Dict[str, Any]], **kwargs):
        
        # Support simpler interface as well
        if isinstance(action, (str, int, np.integer)):
            self.task.is_found_called = bool(action == 0)
            action = {"action": action}
        else:
            self.task.is_found_called = bool(action["action"] == 0)
        
        observations = super().step(action, **kwargs)
        
        ##Terminates episode if wrong found is called
        if self.task.is_found_called == True and \
            self.task.measurements.measures[
            "sub_success" #"current_goal_success"
        ].get_metric() == 0:
            self.task._is_episode_active = False
        
        ##Terminates episode if all goals are found
        if self.task.is_found_called == True and \
            self.task.current_goal_index == len(self.current_episode.goals):
            self.task._is_episode_active = False
        
        return observations

    def get_reward_range(self):
        return (
            self._rl_config.SLACK_REWARD - 1.0,
            self._rl_config.SUCCESS_REWARD + 1.0,
        )

    def get_reward(self, observations, **kwargs):
        reward = self._rl_config.SLACK_REWARD

        current_measure = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_subsuccess():
            current_measure = self._env.task.foundDistance

        reward += self._previous_measure - current_measure
        self._previous_measure = current_measure

        if self._episode_subsuccess():
            self._previous_measure = self._env.get_metrics()[self._reward_measure_name]

        if self._episode_success():
            reward += self._rl_config.SUCCESS_REWARD
        elif self._episode_subsuccess():
            reward += self._rl_config.SUBSUCCESS_REWARD
        elif self._env.task.is_found_called and self._rl_config.FALSE_FOUND_PENALTY:
            reward -= self._rl_config.FALSE_FOUND_PENALTY_VALUE

        return reward
    
    def _episode_success(self):
        return self._env.get_metrics()[self._success_measure_name]

    def _episode_subsuccess(self):
        return self._env.get_metrics()[self._subsuccess_measure_name]

    def get_done(self, observations):
        done = False
        if self._env.episode_over or self._episode_success():
            done = True
        return done

    def get_info(self, observations):
        return self.habitat_env.get_metrics()
    

