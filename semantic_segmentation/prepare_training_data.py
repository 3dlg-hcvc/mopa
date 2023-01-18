import os
import json
import math
from collections import defaultdict
import h5py

import torch
import numpy as np
from habitat import logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import append_text_to_image
from baselines.common.viz_utils import (
    observations_to_image
)
from baselines.common.utils import (
    draw_projection, 
    rotate_tensor, 
    to_grid
)
from tqdm import tqdm

from habitat_baselines.utils.common import generate_video
from baselines.common.environments import MultiObjNavRLEnv
from baselines.common.utils import extract_scalars_from_info
from baselines.config.default import get_config

OBJECT_MAP = {0: 'cylinder_red', 1: 'cylinder_green', 2: 'cylinder_blue', 3: 'cylinder_yellow', 
              4: 'cylinder_white', 5:'cylinder_pink', 6: 'cylinder_black', 7: 'cylinder_cyan'}
METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "raw_metrics", "traj_metrics"}

def make_data(config: Config) -> None:
    split = config.EVAL.SPLIT
    config.defrost()
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
    # config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    config.freeze()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    os.makedirs(config.output_dir, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    env = MultiObjNavRLEnv(config=config)
    agent = OracleAgent(config.TASK_CONFIG, env)

    num_episodes = (min(config.TEST_EPISODE_COUNT, len(env.episodes)) 
                    if config.TEST_EPISODE_COUNT > 0 
                    else len(env.episodes))
    
    rgb_frames = []
    depth_frames = []
    semantic_frames = []
    
    filename = os.path.join(config.output_dir, '{}_data_small.h5'.format(split))
    
    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        agent.reset()
        done = False
        
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            
            rgb_frames.append(np.expand_dims(obs["rgb"], axis=0))
            depth_frames.append(np.expand_dims(np.squeeze(obs["depth"], axis=-1), axis=0))
            semantic_frames.append(np.expand_dims(np.squeeze(obs["semantic"], axis=-1), axis=0))

        if (len(rgb_frames) % 500) >= 0:
            rgb_frames = np.concatenate(rgb_frames, axis=0)
            depth_frames = np.concatenate(depth_frames, axis=0)
            semantic_frames = np.concatenate(semantic_frames, axis=0)

            if not os.path.exists(filename):
                with h5py.File(filename, 'w') as f:
                    f.create_dataset('rgb', data=rgb_frames, dtype=np.float32, compression="gzip", chunks=True, maxshape=(None,256,256,3))
                    f.create_dataset('depth', data=depth_frames, dtype=np.float32, compression="gzip", chunks=True, maxshape=(None,256,256,))
                    f.create_dataset('semantic', data=semantic_frames, dtype=np.float32, compression="gzip", chunks=True, maxshape=(None,256,256,))
            else:
                with h5py.File(filename, 'a') as f:
                    f["rgb"].resize((f["rgb"].shape[0] + rgb_frames.shape[0]), axis = 0)
                    f["rgb"][-rgb_frames.shape[0]:] = rgb_frames
                    
                    f["depth"].resize((f["depth"].shape[0] + depth_frames.shape[0]), axis = 0)
                    f["depth"][-depth_frames.shape[0]:] = depth_frames
                    
                    f["semantic"].resize((f["semantic"].shape[0] + semantic_frames.shape[0]), axis = 0)
                    f["semantic"][-semantic_frames.shape[0]:] = semantic_frames

            rgb_frames = []
            depth_frames = []
            semantic_frames = []

    rgb_frames = np.concatenate(rgb_frames, axis=0)
    depth_frames = np.concatenate(depth_frames, axis=0)
    semantic_frames = np.concatenate(semantic_frames, axis=0)

    if not os.path.exists(filename):
        with h5py.File(filename, 'w') as f:
            f.create_dataset('rgb', data=rgb_frames, dtype=np.float32, compression="gzip", chunks=True, maxshape=(None,256,256,3))
            f.create_dataset('depth', data=depth_frames, dtype=np.float32, compression="gzip", chunks=True, maxshape=(None,256,256,))
            f.create_dataset('semantic', data=semantic_frames, dtype=np.float32, compression="gzip", chunks=True, maxshape=(None,256,256,))
    else:
        with h5py.File(filename, 'a') as f:
            f["rgb"].resize((f["rgb"].shape[0] + rgb_frames.shape[0]), axis = 0)
            f["rgb"][-rgb_frames.shape[0]:] = rgb_frames
            
            f["depth"].resize((f["depth"].shape[0] + depth_frames.shape[0]), axis = 0)
            f["depth"][-depth_frames.shape[0]:] = depth_frames
            
            f["semantic"].resize((f["semantic"].shape[0] + semantic_frames.shape[0]), axis = 0)
            f["semantic"][-semantic_frames.shape[0]:] = semantic_frames


class OracleAgent(Agent):
    def __init__(self, task_config: Config, env):
        self.actions = [
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]
        self.env = env._env

    def reset(self):
        self.follower = ShortestPathFollower(
            self.env.sim, goal_radius=0.25, return_one_hot=False,
            stop_on_error=True # False for debugging
        )
        self.num_tries = 10

    def act(self, observations):
        current_goal = self.env.task.current_goal_index
        
        best_action = 0
        try:
            best_action = self.follower.get_next_action(self.env.current_episode.goals[current_goal].position)
        except:
            while self.num_tries > 0:
                best_action = np.random.choice(self.actions)
                self.num_tries -= 1
                
        return best_action
    
if __name__ == '__main__':
    
    exp_config = "/localhome/sraychau/Projects/Research/MultiON/codebase/multi-obj-nav/baselines/config/sem_seg/train_rednet_cyl_prepare_data.yaml"
    config = get_config(exp_config)

    make_data(config)