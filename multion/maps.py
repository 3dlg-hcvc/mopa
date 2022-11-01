#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import numpy as np
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector, quaternion_from_coeff
import quaternion

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass


# Multion Objects
MULTION_CYL_OBJECT_CATEGORY = {'cylinder_red':1, 'cylinder_green':2, 'cylinder_blue':3, 'cylinder_yellow':4, 
                            'cylinder_white':5, 'cylinder_pink':6, 'cylinder_black':7, 'cylinder_cyan':8}
MULTION_CYL_OBJECT_MAP = dict((v,k) for k,v in MULTION_CYL_OBJECT_CATEGORY.items())
MULTION_REAL_OBJECT_CATEGORY = {'guitar': 1, 'electric_piano': 2, 'basket_ball': 3, 'toy_train': 4, 
                                'teddy_bear': 5, 'rocking_horse': 6, 'backpack': 7, 'trolley_bag': 8}
MULTION_REAL_OBJECT_MAP = dict((v,k) for k,v in MULTION_REAL_OBJECT_CATEGORY.items())

MULTION_TOP_DOWN_MAP_START = 20
OBJECT_MAP_COLORS = np.full((100, 3), 150, dtype=np.uint8)
OBJECT_MAP_COLORS[0] = [50, 50, 50]
OBJECT_MAP_COLORS[1] = [150, 150, 150]
OBJECT_MAP_COLORS[2:10] = np.array(
    [[200, 0, 0], [0, 200, 0], [0, 0, 200], 
    [255, 255, 0], [250, 250, 250], [250, 45, 185], 
    [0, 0, 0], [0,255,255]], 
    dtype=np.uint8
)
OBJECT_MAP_COLORS[10] = [255,165,0]   # Agent location
OBJECT_MAP_COLORS[11] = [143, 0, 255]   # Sampled Goal location

OCC_MAP_COLORS = np.full((5, 3), 150, dtype=np.uint8)
OCC_MAP_COLORS[0] = [150, 150, 150]  # not explored
OCC_MAP_COLORS[1] = [250, 250, 250] # seen + empty
OCC_MAP_COLORS[2] = [0, 0, 0] # seen + occupied

maps.TOP_DOWN_MAP_COLORS[
    MULTION_TOP_DOWN_MAP_START + min(MULTION_CYL_OBJECT_CATEGORY.values()):
    MULTION_TOP_DOWN_MAP_START + max(MULTION_CYL_OBJECT_CATEGORY.values()) + 1] = np.array(
    [[200, 0, 0], [0, 200, 0], [0, 0, 200], 
    [255, 255, 0], [250, 250, 250], [250, 45, 185], 
    [0, 0, 0], [0,255,255]], 
    dtype=np.uint8
)
# maps.TOP_DOWN_MAP_COLORS[MULTION_TOP_DOWN_MAP_START+8] = [255,165,0]   # Agent location
# maps.TOP_DOWN_MAP_COLORS[MULTION_TOP_DOWN_MAP_START+9] = [143, 0, 255]   # Sampled Goal location


def get_topdown_map(
    pathfinder,
    height: float,
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
    with_sampling: Optional[bool] = True,
    num_samples: Optional[float] = 50,
    nav_threshold: Optional[float] = 0.3,
) -> np.ndarray:
    r"""Return a top-down occupancy map for a sim. Note, this only returns valid
    values for whatever floor the agent is currently on.

    :param pathfinder: A habitat-sim pathfinder instances to get the map from
    :param height: The height in the environment to make the topdown map
    :param map_resolution: Length of the longest side of the map.  Used to calculate :p:`meters_per_pixel`
    :param draw_border: Whether or not to draw a border
    :param meters_per_pixel: Overrides map_resolution an

    :return: Image containing 0 if occupied, 1 if unoccupied, and 2 if border (if
        the flag is set).
    """

    if meters_per_pixel is None:
        meters_per_pixel = maps.calculate_meters_per_pixel(
            map_resolution, pathfinder=pathfinder
        )

    if with_sampling:
        top_down_map = pathfinder.get_topdown_view_with_sampling(
            meters_per_pixel=meters_per_pixel, height=height,
            num_samples=num_samples, nav_threshold=nav_threshold
        ).astype(np.uint8)
    else:
        top_down_map = pathfinder.get_topdown_view(
            meters_per_pixel=meters_per_pixel, height=height
        ).astype(np.uint8)

    # Draw border if necessary
    if draw_border:
        maps._outline_border(top_down_map)

    return np.ascontiguousarray(top_down_map)


def get_topdown_map_from_sim(
    sim: "HabitatSim",
    map_resolution: int = 1024,
    draw_border: bool = True,
    meters_per_pixel: Optional[float] = None,
    agent_id: int = 0,
    with_sampling: Optional[bool] = True,
    num_samples: Optional[float] = 50,
    nav_threshold: Optional[float] = 0.3,
) -> np.ndarray:
    r"""Wrapper around :py:`get_topdown_map` that retrieves that pathfinder and heigh from the current simulator

    :param sim: Simulator instance.
    :param agent_id: The agent ID
    """
    return get_topdown_map(
        sim.pathfinder,
        sim.get_agent(agent_id).state.position[1],
        map_resolution,
        draw_border,
        meters_per_pixel,
        with_sampling,
        num_samples,
        nav_threshold
    )

def to_grid(
    realworld_locs,
    meters_per_pixel=None,
    grid_resolution=None,
    lower_bound=None,
    upper_bound=None,
):
    r"""Similar to habitat.utils.visualizations.maps.to_grid, 
        but without _sim
    """
    if meters_per_pixel is None:
        grid_size = torch.tensor(
            [abs(upper_bound[2] - lower_bound[2]) / grid_resolution,
            abs(upper_bound[0] - lower_bound[0]) / grid_resolution],
        device=realworld_locs.device)
    else:
        grid_size = torch.tensor([meters_per_pixel, meters_per_pixel], device=realworld_locs.device)
        
    grid_locs = ((realworld_locs - torch.tensor(lower_bound, device=realworld_locs.device)) / grid_size).type(torch.LongTensor)
    
    return grid_locs


def from_grid(
    grid_locs,
    meters_per_pixel=None,
    grid_resolution=None,
    lower_bound=None,
    upper_bound=None,
    is_3d=True,
):
    r"""
        Similar to habitat.utils.visualizations.maps.from_grid, 
        but without _sim
    """

    if meters_per_pixel is None:
        grid_size_x = abs(upper_bound[0] - lower_bound[0]) / grid_resolution[0]
        grid_size_y = abs(upper_bound[1] - lower_bound[1]) / grid_resolution[1]
    else:
        grid_size_x = grid_size_y = meters_per_pixel
    
    realworld_x = lower_bound[0] + grid_locs[0] * grid_size_x
    realworld_y = lower_bound[1] + grid_locs[1] * grid_size_y
    
    if is_3d:
        return torch.stack([realworld_y, torch.zeros_like(realworld_x), -realworld_x], axis=0)
    
    return torch.stack([realworld_x, realworld_y], axis=0)

def compute_pointgoal(
        source_position, source_rotation, goal_position
    ):
        if source_position.shape[0] == 2:
            source_position = np.array(
                [source_position[1], 0.0, -source_position[0]], dtype=np.float32
            )
        if goal_position.shape[0] == 2:
            goal_position = np.array(
                [goal_position[1], 0.0, -goal_position[0]], dtype=np.float32
            )
        direction_vector = goal_position - source_position
        
        if not isinstance(source_rotation, quaternion.quaternion):
            source_rotation = quaternion.as_quat_array(source_rotation)
        
        direction_vector_agent = quaternion_rotate_vector(
            source_rotation.inverse(), direction_vector
        )

        rho, phi = cartesian_to_polar(
            -direction_vector_agent[2], direction_vector_agent[0]
        )
        return np.array([rho, -phi], dtype=np.float32)
    
def to_global_map_grid(
    episodic_locs,
    agent_episodic_start=None,
    agent_episodic_start_map=None,
):
    r"""Find out location on global map
    """
    
    episodic_locs_map = (agent_episodic_start_map - agent_episodic_start + episodic_locs).type(torch.LongTensor)
    
    return episodic_locs_map

def from_global_map_grid(
    episodic_locs_map,
    agent_episodic_start=None,
    agent_episodic_start_map=None,
):
    r"""Find out episodic location
    """
    
    episodic_locs = (episodic_locs_map - agent_episodic_start_map + agent_episodic_start).type(torch.LongTensor)
    
    episodic_locs_3d = torch.stack([episodic_locs[1], torch.zeros_like(episodic_locs[0]), -episodic_locs[0]], axis=0)
    
    return episodic_locs_3d