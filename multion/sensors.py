import os
from typing import Any, Optional, Union, Dict

import pickle
import numpy as np
from scipy import ndimage
from gym import spaces
from habitat.config import Config
from habitat.core.registry import registry
from habitat.core.simulator import Sensor, SensorTypes, Simulator
from habitat.tasks.nav.nav import HeadingSensor, PointGoalSensor
from habitat.core.simulator import SemanticSensor
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
import quaternion

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
class EpisodicCompassSensor(Sensor):
    r"""The agents heading in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "episodic_compass"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING
    
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid
    
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return self._quat_to_xy_heading(
            rotation_world_agent.inverse() * rotation_world_start
        )
        
@registry.register_sensor(name="EpisodicRotationSensor")
class EpisodicRotationSensor(Sensor):
    r"""The agents rotation quat in the coordinate frame defined by the epiosde,
    theta=0 is defined by the agents state at t=0
    """
    cls_uuid: str = "episodic_rotation"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        super().__init__(config=config)

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING
    
    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid
    
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32)
    
    def _quat_to_xy_heading(self, quat):
        direction_vector = np.array([0, 0, -1])

        heading_vector = quaternion_rotate_vector(quat, direction_vector)

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        return np.array([phi], dtype=np.float32)

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation
        rotation_world_start = quaternion_from_coeff(episode.start_rotation)

        return quaternion.as_float_array(
            rotation_world_agent.inverse() * rotation_world_start
        )
        
@registry.register_sensor(name="RotationSensor")
class RotationSensor(Sensor):
    r"""The agent's episodic rotation as quaternion
        -   similar to CompassSensor
    """
    cls_uuid: str = "agent_rotation"
    def __init__(self, sim, config, **kwargs: Any):
        self._sim = sim
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid
    
    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.HEADING

    # Defines the size and range of the observations of the sensor
    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=np.finfo(np.float32).min,
            high=np.finfo(np.float32).max,
            shape=(34,),
            dtype=np.float32,
        )

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        rotation_world_agent = agent_state.rotation

        return quaternion.as_float_array(rotation_world_agent)

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
    cls_uuid: str = "episodic_gps"
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
        self.pregenerated = config.pregenerated
        if self.pregenerated:
            with open(config.pregenerated_file_path, 'rb') as handle:
                self.mapCache = pickle.load(handle)
        
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
    def conv_grid(
        self,
        realworld_x,
        realworld_y,
        coordinate_min = -120.3241-1e-6,
        coordinate_max = 120.0399+1e-6,
        grid_resolution = (300, 300)
    ):
        r"""Return gridworld index of realworld coordinates assuming top-left corner
        is the origin. The real world coordinates of lower left corner are
        (coordinate_min, coordinate_min) and of top right corner are
        (coordinate_max, coordinate_max)
        """
        grid_size = (
            (coordinate_max - coordinate_min) / grid_resolution[0],
            (coordinate_max - coordinate_min) / grid_resolution[1],
        )
        grid_x = int((coordinate_max - realworld_x) / grid_size[0])
        grid_y = int((realworld_y - coordinate_min) / grid_size[1])
        return grid_x, grid_y

    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        agent_vertical_pos = str(round(agent_position[1],2))
        
        if self.pregenerated:
            self.currMap = np.copy(self.mapCache[episode.scene_id])
            
            # Adding goal information on the map
            for i in range(len(episode.goals)):
                loc0 = episode.goals[i].position[0]
                loc2 = episode.goals[i].position[2]
                grid_loc = self.conv_grid(loc0, loc2)
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
                    grid_loc = self.conv_grid(loc0, loc2)
                    self.currMap[grid_loc[0]-1:grid_loc[0]+2, 
                                grid_loc[1]-1:grid_loc[1]+2, 
                                self.channel_num_distractors] = (
                                                kwargs['task'].object_to_datset_mapping[episode.distractors[i].object_category] 
                                                + self.distrIndexOffset
                                            )
            currPix = self.conv_grid(
                agent_position[2],
                agent_position[0]
            )
        else:
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

@registry.register_sensor(name="ObjectMapSensor")
class ObjectMapSensor(Sensor):
    r"""
        Map with Goals and Distractors marked
    Args:
        sim: reference to the simulator for calculating task observations.
        config: sensor config
    Attributes:
        
    """
    cls_uuid: str = "object_map"
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
        self.mask_map = config.mask_map
        self.visibility_dist = config.VISIBILITY_DIST
        self.fov = config.FOV
        self.object_ind_offset = config.object_ind_offset
        self.channel_num = 1
        self.object_padding = config.object_padding
        
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=10,
            shape=(self.map_size, self.map_size, self.map_channels),
            dtype=np.int,
        )
        
    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation

        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )

        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        z_neg_z_flip = np.pi
        return np.array(phi) + z_neg_z_flip
        
    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        agent_vertical_pos = str(round(agent_position[1], 2))
        
        if (episode.scene_id in self.mapCache and 
                agent_vertical_pos in self.mapCache[episode.scene_id]):
            top_down_map = self.mapCache[episode.scene_id][agent_vertical_pos].copy()
            
        else:
            top_down_map = multion_maps.get_topdown_map_from_sim(
                self._sim,
                draw_border=self.draw_border,
                meters_per_pixel=self.meters_per_pixel,
                with_sampling=self.with_sampling,
                num_samples=self.num_samples,
                nav_threshold=self.nav_threshold
            )
            if episode.scene_id not in self.mapCache:
                if len(self.mapCache) > self.cache_max_size:
                    # Reset cache when cache size exceeds max size
                    self.mapCache = {}
                self.mapCache[episode.scene_id] = {}
            self.mapCache[episode.scene_id][agent_vertical_pos] = top_down_map.copy()
            
        object_map = np.zeros((top_down_map.shape[0], top_down_map.shape[1], self.map_channels))
        object_map[:top_down_map.shape[0], :top_down_map.shape[1], 0] = top_down_map

        # Get agent location on map
        agent_loc = maps.to_grid(
                    agent_position[2],
                    agent_position[0],
                    top_down_map.shape[0:2],
                    sim=self._sim,
                )

        # Mark the agent location
        object_map[max(0, agent_loc[0]-self.object_padding):min(top_down_map.shape[0], agent_loc[0]+self.object_padding),
                    max(0, agent_loc[1]-self.object_padding):min(top_down_map.shape[1], agent_loc[1]+self.object_padding),
                    self.channel_num+1] = 10 #len(kwargs['task'].object_to_datset_mapping) + self.object_ind_offset
        

        # Mask the map
        if self.mask_map:
            _fog_of_war_mask = np.zeros_like(top_down_map)
            _fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                top_down_map,
                _fog_of_war_mask,
                np.array(agent_loc),
                self.get_polar_angle(),
                fov=self.fov,
                max_line_len=self.visibility_dist
                / self.meters_per_pixel,
            )
            object_map[:, :, self.channel_num] += 1
            object_map[:, :, self.channel_num] *= _fog_of_war_mask # Hide unobserved areas

        # Adding goal information on the map
        for i in range(len(episode.goals)):
            loc0 = episode.goals[i].position[0]
            loc2 = episode.goals[i].position[2]
            grid_loc = maps.to_grid(
                loc2,
                loc0,
                top_down_map.shape[0:2],
                sim=self._sim,
            )
            object_map[grid_loc[0]-self.object_padding:grid_loc[0]+self.object_padding, 
                        grid_loc[1]-self.object_padding:grid_loc[1]+self.object_padding,
                        self.channel_num] = (
                                kwargs['task'].object_to_datset_mapping[episode.goals[i].object_category]
                                + self.object_ind_offset
                            )

        for i in range(len(episode.distractors)):
            loc0 = episode.distractors[i].position[0]
            loc2 = episode.distractors[i].position[2]
            grid_loc = maps.to_grid(
                loc2,
                loc0,
                top_down_map.shape[0:2],
                sim=self._sim,
            )
            object_map[grid_loc[0]-self.object_padding:grid_loc[0]+self.object_padding, 
                        grid_loc[1]-self.object_padding:grid_loc[1]+self.object_padding,
                        self.channel_num] = (
                                        kwargs['task'].object_to_datset_mapping[episode.distractors[i].object_category] 
                                        + self.object_ind_offset
                                    )

        # Hide the  out-of-view objects
        if self.mask_map:
            object_map[:, :, self.channel_num] *= _fog_of_war_mask   
            
        final_object_map = np.zeros((self.map_size, self.map_size, self.map_channels))
        final_object_map[:top_down_map.shape[0], :top_down_map.shape[1], :] = object_map

        return final_object_map
    
@registry.register_sensor(name="OracleMapSizeSensor")
class OracleMapSizeSensor(Sensor):
    r"""
        Oracle Map size
    Args:
        sim: reference to the simulator for calculating task observations.
        config: sensor config
    Attributes:
        
    """
    cls_uuid: str = "oracle_map_size"
    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        
        self.mapCache = {}
        self.cache_max_size = config.cache_max_size
        self.meters_per_pixel = config.meters_per_pixel
        self.num_samples = config.num_samples
        self.nav_threshold = config.nav_threshold
        self.draw_border = config.draw_border   #false
        self.with_sampling = config.with_sampling # true
        
        super().__init__(config=config)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def _get_sensor_type(self, *args: Any, **kwargs: Any):
        return SensorTypes.SEMANTIC

    def _get_observation_space(self, *args: Any, **kwargs: Any):
        return spaces.Box(
            low=0,
            high=1000,
            shape=(3, 2),
            dtype=np.int,
        )
        
    def get_observation(
        self, *args: Any, observations, episode, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        agent_vertical_pos = str(round(agent_position[1], 2))
        
        if (episode.scene_id in self.mapCache and 
                agent_vertical_pos in self.mapCache[episode.scene_id]):
            top_down_map = self.mapCache[episode.scene_id][agent_vertical_pos]
            
        else:
            top_down_map = multion_maps.get_topdown_map_from_sim(
                self._sim,
                draw_border=self.draw_border,
                meters_per_pixel=self.meters_per_pixel,
                with_sampling=self.with_sampling,
                num_samples=self.num_samples,
                nav_threshold=self.nav_threshold
            )
            if episode.scene_id not in self.mapCache:
                if len(self.mapCache) > self.cache_max_size:
                    # Reset cache when cache size exceeds max size
                    self.mapCache = {}
                self.mapCache[episode.scene_id] = {}
            self.mapCache[episode.scene_id][agent_vertical_pos] = top_down_map
            
        pathfinder = self._sim.pathfinder
        lower_bound, upper_bound = pathfinder.get_bounds()
            
        return np.array([np.array(top_down_map.shape), [lower_bound[2], lower_bound[0]], [upper_bound[2], upper_bound[0]]])

@registry.register_sensor(name="PointGoalWithGPSCompassSensor")
class IntegratedPointGoalGPSAndCompassSensor(PointGoalSensor):
    r"""Sensor that integrates PointGoals observations (which are used PointGoal Navigation) and GPS+Compass.

    For the agent in simulator the forward direction is along negative-z.
    In polar coordinate format the angle returned is azimuth to the goal.

    Args:
        sim: reference to the simulator for calculating task observations.
        config: config for the PointGoal sensor. Can contain field for
            GOAL_FORMAT which can be used to specify the format in which
            the pointgoal is specified. Current options for goal format are
            cartesian and polar.

            Also contains a DIMENSIONALITY field which specifes the number
            of dimensions ued to specify the goal, must be in [2, 3]

    Attributes:
        _goal_format: format for specifying the goal which can be done
            in cartesian or polar coordinates.
        _dimensionality: number of dimensions used to specify the goal
    """
    cls_uuid: str = "pointgoal_with_gps_compass"

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return self.cls_uuid

    def get_observation(
        self, observations, episode, *args: Any, **kwargs: Any
    ):
        agent_state = self._sim.get_agent_state()
        agent_position = agent_state.position
        rotation_world_agent = agent_state.rotation
        goal_position = np.array(episode.goals[kwargs["task"].current_goal_index].position, dtype=np.float32)

        return self._compute_pointgoal(
            agent_position, rotation_world_agent, goal_position
        )
