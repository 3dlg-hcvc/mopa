import os
import json
import math
from collections import defaultdict

import numpy as np
from habitat import logger
from habitat.config.default import Config
from habitat.core.agent import Agent
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations.utils import observations_to_image, append_text_to_image
from tqdm import tqdm

from habitat_baselines.utils.common import generate_video
from baselines.common.environments import MultiObjNavRLEnv
from baselines.common.utils import extract_scalars_from_info

OBJECT_MAP = {0: 'cylinder_red', 1: 'cylinder_green', 2: 'cylinder_blue', 3: 'cylinder_yellow', 
              4: 'cylinder_white', 5:'cylinder_pink', 6: 'cylinder_black', 7: 'cylinder_cyan'}
METRICS_BLACKLIST = {"top_down_map", "collisions.is_collision", "raw_metrics", "traj_metrics"}

def evaluate_agent(config: Config) -> None:
    split = config.EVAL.SPLIT
    config.defrost()
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
    config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = -1
    config.TASK_CONFIG.DATASET.SPLIT = split
    if len(config.VIDEO_OPTION) > 0:
        config.defrost()
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP")
        config.TASK_CONFIG.TASK.MEASUREMENTS.append("COLLISIONS")
        os.makedirs(config.VIDEO_DIR, exist_ok=True)
    config.freeze()
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    logger.add_filehandler(config.LOG_FILE)

    env = MultiObjNavRLEnv(config=config)

    assert config.EVAL.NONLEARNING.AGENT in [
        "OracleAgent",
        "RandomAgent",
        "HandcraftedAgent",
    ], "EVAL.NONLEARNING.AGENT must be either OracleAgent or RandomAgent or HandcraftedAgent."

    if config.EVAL.NONLEARNING.AGENT == "OracleAgent":
        agent = OracleAgent(config.TASK_CONFIG, env)
    elif config.EVAL.NONLEARNING.AGENT == "RandomAgent":
        agent = RandomAgent()
    else:
        agent = HandcraftedAgent()

    stats = defaultdict(float)
    num_episodes = (min(config.EVAL.EPISODE_COUNT, len(env.episodes)) 
                    if config.EVAL.EPISODE_COUNT > 0 
                    else len(env.episodes))
    
    for i in tqdm(range(num_episodes)):
        obs = env.reset()
        agent.reset()
        done = False

        rgb_frames = []
        while not done:
            action = agent.act(obs)
            obs, reward, done, info = env.step(action)
            if len(config.VIDEO_OPTION) > 0:
                frame = observations_to_image(obs, info=info)
                txt_to_show = ('Action: '+ str(action) + 
                                '; Dist_to_multi_goal:' + str(round(info['distance_to_multi_goal'],2)) + 
                                '; Dist_to_curr_goal:' + str(round(info['distance_to_currgoal'],2)) + 
                                '; Current Goal:' + str(OBJECT_MAP[obs['multiobjectgoal'][0]]) + 
                                '; Found_called:' + str(env.task.is_found_called) +
                                '; Success:' + str(info['success']) +
                                '; Sub_success:' + str(info['sub_success']) +
                                '; Progress:' + str(round(info['progress'],2)) +
                                '; Reward:' + str(round(reward,2)))
                
                goal_str = ";".join([g.object_category for g in env.current_episode.goals])
                distr_str = ";".join([d.object_category for d in env.current_episode.distractors])
                txt_to_show += "\n:Goals=" + goal_str + "; distractors=" + distr_str

                frame = append_text_to_image(
                        frame, txt_to_show
                    )
                rgb_frames.append(frame)

        if len(config.VIDEO_OPTION) > 0:
            generate_video(
                video_option=config.VIDEO_OPTION,
                video_dir=config.VIDEO_DIR,
                images=rgb_frames,
                episode_id=f"{os.path.basename(env.current_episode.scene_id)}_{env.current_episode.episode_id}",
                checkpoint_idx=0,
                metrics=extract_scalars_from_info(info, METRICS_BLACKLIST),
                tb_writer=None,
            )
            
        if "top_down_map" in info:
            del info["top_down_map"]
        if "collisions" in info:
            del info["collisions"]
            
        for m, v in info.items():
            stats[m] += v

    stats = {k: v / num_episodes for k, v in stats.items()}

    logger.info(f"Averaged benchmark for {config.EVAL.NONLEARNING.AGENT}:")
    for stat_key in stats.keys():
        logger.info("{}: {:.3f}".format(stat_key, stats[stat_key]))

    with open(f"stats_{config.EVAL.NONLEARNING.AGENT}_{split}.json", "w") as f:
        json.dump(stats, f, indent=4)

class RandomAgent(Agent):
    r"""Selects an action at each time step by sampling from the oracle action
    distribution of the training set.
    """

    def __init__(self, probs=None):
        self.num_actions = 100
        self.actions = [
            HabitatSimActions.MOVE_FORWARD,
            HabitatSimActions.TURN_LEFT,
            HabitatSimActions.TURN_RIGHT,
        ]

    def reset(self):
        self.num_actions = 100

    def act(self, observations):
        if self.num_actions > 0:
            self.num_actions -= 1
            return np.random.choice(self.actions)
        
        return 0 # Stop


class HandcraftedAgent(Agent):
    r"""Agent picks a random heading and takes 37 forward actions (average
    oracle path length) before calling stop.
    """

    def __init__(self):
        self.reset()

    def reset(self):
        # 9.27m avg oracle path length in Train.
        # Fwd step size: 0.25m. 9.25m/0.25m = 37
        self.forward_steps = 37
        self.turns = np.random.randint(0, int(360 / 15) + 1)

    def act(self, observations):
        if self.turns > 0:
            self.turns -= 1
            return {"action": HabitatSimActions.TURN_RIGHT}
        if self.forward_steps > 0:
            self.forward_steps -= 1
            return {"action": HabitatSimActions.MOVE_FORWARD}
        return {"action": HabitatSimActions.STOP}

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