import os
from typing import Any, Optional

import numpy as np
from scipy import ndimage
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import HeadingSensor
from habitat.core.dataset import Dataset
from habitat.utils.geometry_utils import (
    quaternion_from_coeff,
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from multion.task import MultiObjectGoalNavEpisode
from habitat.utils.visualizations import maps, fog_of_war
from multion import maps as multion_maps
from PIL import Image   #debugging

SLURM_JOBID = os.environ.get("SLURM_JOB_ID", None)
SLURM_TMPDIR = os.environ.get("SLURM_TMPDIR", None)

@registry.register_sensor
class MultiObjectGoalSensor(Sensor):
    r"""A sensor for Object Goal specification as observations which is used in
    ObjectGoal Navigation. The goal is expected to be specified by object_id or
    semantic category id.
    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.
    Args:
        sim: a reference to the simulator for calculating task observations.
        config: a config for the ObjectGoalSensor sensor. Can contain field
            GOAL_SPEC that specifies which id use for goal specification,
            GOAL_SPEC_MAX_VAL the maximum object_id possible used for
            observation space definition.
        dataset: a Object Goal navigation dataset that contains dictionaries
        of categories id to text mapping.
    """

    def __init__(
        self, sim, config: Config, dataset: Dataset, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._dataset = dataset
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "multiobjectgoal"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (1,)
        max_value = (self.config.GOAL_SPEC_MAX_VAL - 1,)
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            max_value = max(
                self._dataset.category_to_task_category_id.values()
            )

        return spaces.Box(
            low=0, high=max_value, shape=sensor_shape, dtype=np.int64
        )

    def get_observation(
        self,
        observations,
        *args: Any,
        episode: MultiObjectGoalNavEpisode,
        **kwargs: Any,
    ) -> Optional[int]:
        if self.config.GOAL_SPEC == "TASK_CATEGORY_ID":
            if len(episode.goals) == 0:
                logger.error(
                    f"No goal specified for episode {episode.episode_id}."
                )
                return None
            category_name = [i.object_category for i in episode.goals]
            goalArray = np.array(
                [self._dataset.category_to_task_category_id[i] for i in category_name],
                dtype=np.int64,
            )
            return goalArray[kwargs["task"].current_goal_index:kwargs["task"].current_goal_index+1]
        elif self.config.GOAL_SPEC == "OBJECT_ID":
            return np.array([episode.goals[0].object_name_id], dtype=np.int64)
        else:
            raise RuntimeError(
                "Wrong GOAL_SPEC specified for ObjectGoalSensor."
            )

@registry.register_sensor(name="PositionSensor")
class AgentPositionSensor(Sensor):
    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "agent_position"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(3,),
            dtype=np.float32,
        )

    def get_observation(
        self, observations, *args: Any, episode, **kwargs: Any
    ):
        return self._sim.get_agent_state().position


@registry.register_sensor
class HeadingSensor(Sensor):
    r"""Sensor for observing the agent's heading in the global coordinate
    frame.
    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the sensor.
    """

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "heading"

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float)

    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return self._quat_to_xy_heading(rotation_world_agent.inverse())

@registry.register_sensor(name="CompassSensor")
class EpisodicCompassSensor(HeadingSensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "compass"
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )

@registry.register_sensor(name="GPSSensor")
class EpisodicGPSSensor(Sensor):
    r"""The agents current location in the coordinate frame defined by the episode,
    i.e. the axis it faces along and the origin is defined by its state at t=0
    Args:
        sim: reference to the simulator for calculating task observations.
        config: Contains the DIMENSIONALITY field for the number of dimensions to express the agents position
    Attributes:
        _dimensionality: number of dimensions used to specify the agents position
    """
    cls_uuid: str = "gps"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim

        self._dimensionality = getattr(config, "DIMENSIONALITY", 2)
        assert self._dimensionality in [2, 3]
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.POSITION

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        sensor_shape = (self._dimensionality,)
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=sensor_shape,
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()

        origin = np.array(episode.start_position, dtype=np.float32)
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        agent_position = agent_state.position

        agent_position = quaternion_rotate_vector(
            rotation_world_start.inverse(), agent_position - origin
        )
        if self._dimensionality == 2:
            return np.array(
                [-agent_position[2], agent_position[0]], dtype=np.float32
            )
        else:
            return agent_position.astype(np.float32)

@registry.register_sensor(name="SemOccSensor")
class SemOccSensor(Sensor):
    r"""
        Oracle Occupancy Map Sensor with Goals and Distractors marked
    Args:
        sim: reference to the simulator for calculating task observations.
        config: sensor config
    Attributes:
        
    """
    cls_uuid: str = "semMap"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        
        self.mapCache = {}
        self.cache_max_size = config.cache_max_size
        self.map_size = config.map_size
        self.meters_per_pixel = config.meters_per_pixel
        self.num_samples = config.num_samples
        self.nav_threshold = config.nav_threshold
        self.map_channels = config.MAP_CHANNELS
        self.draw_border = config.draw_border   #false
        self.with_sampling = config.with_sampling # true
        self.channel_num_goals = config.channel_num_goals #1
        self.objIndexOffset = config.objIndexOffset #1
        if config.INCLUDE_DISTRACTORS:
            if config.ORACLE_MAP_INCLUDE_DISTRACTORS_W_GOAL:
                self.channel_num_distractors = 1
            else:
                self.channel_num_distractors = 2
        else:
            self.channel_num_distractors = 0
        self.num_distractors_included = config.num_distractors_included
        self.mask_map = config.mask_map # false for "oracle", true for "oracle-ego"
        
        #debugging
        # self.count = 0

        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=2,
            shape=(self.map_size, self.map_size, 3),
            dtype=np.int,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        agent_vertical_pos = str(round(agent_position[1],2))
        
        if (episode.scene_id in self.mapCache and 
                agent_vertical_pos in self.mapCache[episode.scene_id]):
            self.currMap = self.mapCache[episode.scene_id][agent_vertical_pos].copy()
            
        else:
            top_down_map = multion_maps.get_topdown_map_from_sim(
                self._sim,
                draw_border=self.draw_border,
                meters_per_pixel=self.meters_per_pixel,
                with_sampling=self.with_sampling,
                num_samples=self.num_samples,
                nav_threshold=self.nav_threshold
            )
            top_down_map += 1 # update topdown map to have 1 if occupied, 2 if unoccupied/navigable

            tmp_map = np.zeros((top_down_map.shape[0],top_down_map.shape[1],self.map_channels))
            tmp_map[:top_down_map.shape[0], :top_down_map.shape[1], 0] = top_down_map
            self.currMap = tmp_map
            
            # Adding goal information on the map
            for i in range(len(episode.goals)):
                loc0 = episode.goals[i].position[0]
                loc2 = episode.goals[i].position[2]
                grid_loc = maps.to_grid(
                    loc2,
                    loc0,
                    self.currMap.shape[0:2],
                    sim=self._sim,
                )
                self.currMap[grid_loc[0]-1:grid_loc[0]+2, 
                            grid_loc[1]-1:grid_loc[1]+2, 
                            self.channel_num_goals] = (
                                                            kwargs['task'].object_to_datset_mapping[episode.goals[i].object_category] 
                                                            + self.objIndexOffset
                                                        )
                
            if self.channel_num_distractors > 0:
                # Adding distractor information on the map
                self.num_distractors_included = (self.num_distractors_included 
                                                if self.num_distractors_included > 0 
                                                else len(episode.distractors))
                for i in range(self.num_distractors_included):
                    loc0 = episode.distractors[i].position[0]
                    loc2 = episode.distractors[i].position[2]
                    grid_loc = maps.to_grid(
                        loc2,
                        loc0,
                        self.currMap.shape[0:2],
                        sim=self._sim,
                    )
                    self.currMap[grid_loc[0]-1:grid_loc[0]+2, 
                                grid_loc[1]-1:grid_loc[1]+2, 
                                self.channel_num_distractors] = (
                                                kwargs['task'].object_to_datset_mapping[episode.distractors[i].object_category] 
                                                + self.distrIndexOffset
                                            )
                                
            if episode.scene_id not in self.mapCache:
                if len(self.mapCache) > self.cache_max_size:
                    # Reset cache when cache size exceeds max size
                    self.mapCache = {}
                self.mapCache[episode.scene_id] = {}
            self.mapCache[episode.scene_id][agent_vertical_pos] = self.currMap.copy()

        currPix = maps.to_grid(
                agent_position[2],
                agent_position[0],
                self.currMap.shape[0:2],
                sim=self._sim,
            )
        
        if self.mask_map:
            self.expose = np.repeat(
                self.task.measurements.measures["fow_map"].get_metric()[:, :, np.newaxis], 3, axis = 2
            )
            patch_tmp = self.currMap * self.expose
        else:
            patch_tmp = self.currMap
            
        patch = patch_tmp[max(currPix[0]-40,0):currPix[0]+40, max(currPix[1]-40,0):currPix[1]+40,:]
        if patch.shape[0] < 80 or patch.shape[1] < 80:
            if currPix[0] < 40:
                curr_x = currPix[0]
            else:
                curr_x = 40
            if currPix[1] < 40:
                curr_y = currPix[1]
            else:
                curr_y = 40
                
            map_mid = (80//2)
            tmp = np.zeros((80, 80,self.map_channels))
            tmp[map_mid-curr_x:map_mid-curr_x+patch.shape[0],
                    map_mid-curr_y:map_mid-curr_y+patch.shape[1], :] = patch
            patch = tmp
            
        if "heading" in observations:
            agent_heading = observations["heading"]
        else:
            headingSensor = HeadingSensor(self._sim,self.config)
            agent_heading = headingSensor.get_observation(observations, episode, kwargs)
            
        patch = ndimage.interpolation.rotate(patch, -(agent_heading[0] * 180/np.pi) + 90, 
                                             axes=(0,1), order=0, reshape=False)
        
        sem_map = patch[40-25:40+25, 40-25:40+25, :]
        
        # debugging
        """ Image.fromarray(
            maps.colorize_topdown_map(
                (sem_map[:,:,1]-1+multion_maps.MULTION_TOP_DOWN_MAP_START).astype(np.uint8)
                )
            ).save(
            f"test_maps/{episode.episode_id}_sem_map_{self.count}.png")
        Image.fromarray(
            maps.colorize_topdown_map(
                (self.currMap[:,:,1]-1+multion_maps.MULTION_TOP_DOWN_MAP_START).astype(np.uint8)
                )
            ).save(
            f"test_maps/{episode.episode_id}_{self.count}.png")
        self.count+=1 """
        # debugging - end
        
        return sem_map
