#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Optional

import torch
import numpy as np
from habitat.utils.visualizations import maps
from habitat.tasks.utils import cartesian_to_polar
from habitat.utils.geometry_utils import quaternion_rotate_vector
import quaternion

try:
    from habitat.sims.habitat_simulator.habitat_simulator import HabitatSim
except ImportError:
    pass


# Multion Objects
MULTION_CYL_OBJECT_CATEGORY = {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2, 'cylinder_yellow':3, 
                            'cylinder_white':4, 'cylinder_pink':5, 'cylinder_black':6, 'cylinder_cyan':7}
MULTION_CYL_OBJECT_MAP = {0: 'cylinder_red', 1: 'cylinder_green', 2: 'cylinder_blue', 3: 'cylinder_yellow', 
                            4: 'cylinder_white', 5:'cylinder_pink', 6: 'cylinder_black', 7: 'cylinder_cyan'}
MULTION_TOP_DOWN_MAP_START = 20
maps.TOP_DOWN_MAP_COLORS[MULTION_TOP_DOWN_MAP_START-1] = [150, 150, 150]
maps.TOP_DOWN_MAP_COLORS[MULTION_TOP_DOWN_MAP_START-2] = [250, 250, 250]
maps.TOP_DOWN_MAP_COLORS[MULTION_TOP_DOWN_MAP_START:MULTION_TOP_DOWN_MAP_START+8] = np.array(
    [[200, 0, 0], [0, 200, 0], [0, 0, 200], 
    [255, 255, 0], [250, 250, 250], [250, 45, 185], 
    [0, 0, 0], [0,255,255]], 
    dtype=np.uint8
)
maps.TOP_DOWN_MAP_COLORS[MULTION_TOP_DOWN_MAP_START+8] = [255,165,0]   # Agent location


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

def from_grid(
    grid_locs,
    grid_resolution,
    lower_bound,
    upper_bound,
):
    r"""
        Similar to habitat.utils.visualizations.maps.from_grid, 
        but without _sim
    """

    grid_size_x = abs(upper_bound[0] - lower_bound[0]) / grid_resolution[0]
    grid_size_y = abs(upper_bound[1] - lower_bound[1]) / grid_resolution[1]
    
    realworld_x = lower_bound[0] + grid_locs[0] * grid_size_x
    realworld_y = lower_bound[1] + grid_locs[1] * grid_size_y
    realworld_coordinates = torch.stack([realworld_y, torch.zeros_like(realworld_x), realworld_x], axis=0)
    
    return realworld_coordinates

def compute_pointgoal(
        source_position, source_rotation, goal_position
    ):
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
    