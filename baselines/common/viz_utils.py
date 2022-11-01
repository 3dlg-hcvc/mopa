#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import textwrap
from typing import Dict, List, Optional, Tuple

import imageio
import numpy as np
import tqdm
import torch
from einops import rearrange
import math
import torch.nn.functional as F
import torch_scatter

from habitat.core.logging import logger
from habitat.core.utils import try_cv2_import
from habitat.utils.visualizations import maps
from multion import maps as multion_maps
import matplotlib.pyplot as plt
from matplotlib import colors
import matplotlib

cv2 = try_cv2_import()

def subplot(plt, Y_X, sz_y_sz_x = (10, 10)):
    Y,X = Y_X
    sz_y, sz_x = sz_y_sz_x
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    return fig, axes

def paste_overlapping_image(
    background: np.ndarray,
    foreground: np.ndarray,
    location: Tuple[int, int],
    mask: Optional[np.ndarray] = None,
):
    r"""Composites the foreground onto the background dealing with edge
    boundaries.
    Args:
        background: the background image to paste on.
        foreground: the image to paste. Can be RGB or RGBA. If using alpha
            blending, values for foreground and background should both be
            between 0 and 255. Otherwise behavior is undefined.
        location: the image coordinates to paste the foreground.
        mask: If not None, a mask for deciding what part of the foreground to
            use. Must be the same size as the foreground if provided.
    Returns:
        The modified background image. This operation is in place.
    """
    assert mask is None or mask.shape[:2] == foreground.shape[:2]
    foreground_size = foreground.shape[:2]
    min_pad = (
        max(0, foreground_size[0] // 2 - location[0]),
        max(0, foreground_size[1] // 2 - location[1]),
    )

    max_pad = (
        max(
            0,
            (location[0] + (foreground_size[0] - foreground_size[0] // 2))
            - background.shape[0],
        ),
        max(
            0,
            (location[1] + (foreground_size[1] - foreground_size[1] // 2))
            - background.shape[1],
        ),
    )

    background_patch = background[
        (location[0] - foreground_size[0] // 2 + min_pad[0]) : (
            location[0]
            + (foreground_size[0] - foreground_size[0] // 2)
            - max_pad[0]
        ),
        (location[1] - foreground_size[1] // 2 + min_pad[1]) : (
            location[1]
            + (foreground_size[1] - foreground_size[1] // 2)
            - max_pad[1]
        ),
    ]
    foreground = foreground[
        min_pad[0] : foreground.shape[0] - max_pad[0],
        min_pad[1] : foreground.shape[1] - max_pad[1],
    ]
    if foreground.size == 0 or background_patch.size == 0:
        # Nothing to do, no overlap.
        return background

    if mask is not None:
        mask = mask[
            min_pad[0] : foreground.shape[0] - max_pad[0],
            min_pad[1] : foreground.shape[1] - max_pad[1],
        ]

    if foreground.shape[2] == 4:
        # Alpha blending
        foreground = (
            background_patch.astype(np.int32) * (255 - foreground[:, :, [3]])
            + foreground[:, :, :3].astype(np.int32) * foreground[:, :, [3]]
        ) // 255
    if mask is not None:
        background_patch[mask] = foreground[mask]
    else:
        background_patch[:] = foreground
    return background


def images_to_video(
    images: List[np.ndarray],
    output_dir: str,
    video_name: str,
    fps: int = 10,
    quality: Optional[float] = 5,
    **kwargs,
):
    r"""Calls imageio to run FFMPEG on a list of images. For more info on
    parameters, see https://imageio.readthedocs.io/en/stable/format_ffmpeg.html
    Args:
        images: The list of images. Images should be HxWx3 in RGB order.
        output_dir: The folder to put the video in.
        video_name: The name for the video.
        fps: Frames per second for the video. Not all values work with FFMPEG,
            use at your own risk.
        quality: Default is 5. Uses variable bit rate. Highest quality is 10,
            lowest is 0.  Set to None to prevent variable bitrate flags to
            FFMPEG so you can manually specify them using output_params
            instead. Specifying a fixed bitrate using ‘bitrate’ disables
            this parameter.
    """
    assert 0 <= quality <= 10
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    video_name = video_name.replace(" ", "_").replace("\n", "_") + ".mp4"
    writer = imageio.get_writer(
        os.path.join(output_dir, video_name),
        fps=fps,
        quality=quality,
        **kwargs,
    )
    logger.info(f"Video created: {os.path.join(output_dir, video_name)}")
    for im in tqdm.tqdm(images):
        writer.append_data(im)
    writer.close()


def draw_collision(view: np.ndarray, alpha: float = 0.4) -> np.ndarray:
    r"""Draw translucent red strips on the border of input view to indicate
    a collision has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of red collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([255, 0, 0]) + (1.0 - alpha) * view)[mask]
    return view

def draw_subsuccess(view: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    r"""Draw translucent blue strips on the border of input view to indicate
    a subsuccess event has taken place.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of blue collision strip. 1 is completely non-transparent.
    Returns:
        A view with collision effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([0, 0, 255]) + (1.0 - alpha) * view)[mask]
    return view


def draw_found(view: np.ndarray, alpha: float = 1) -> np.ndarray:
    r"""Draw translucent blue strips on the border of input view to indicate
    that a found action has been called.
    Args:
        view: input view of size HxWx3 in RGB order.
        alpha: Opacity of blue collision strip. 1 is completely non-transparent.
    Returns:
        A view with found action effect drawn.
    """
    strip_width = view.shape[0] // 20
    mask = np.ones(view.shape)
    mask[strip_width:-strip_width, strip_width:-strip_width] = 0
    mask = mask == 1
    view[mask] = (alpha * np.array([0, 0, 255]) + (1.0 - alpha) * view)[mask]
    return view

def observations_to_image(observation: Dict, projected_features: np.ndarray=None, 
        egocentric_projection: np.ndarray=None, global_map: np.ndarray=None, 
        info: Dict=None, action: np.ndarray=None, object_map: np.ndarray=None, predicted_semantic=None,
        semantic_projections: np.ndarray=None, global_object_map: np.ndarray=None, 
        agent_view: np.ndarray=None, config: np.ndarray=None) -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        action: action returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    egocentric_view = []
    if "rgb" in observation:
        observation_size = observation["rgb"].shape[0]
        rgb = observation["rgb"]
        if not isinstance(rgb, np.ndarray):
            rgb = rgb.cpu().numpy()

        egocentric_view.append(rgb)

    # draw depth map if observation has depth info
    if "depth" in observation:
        observation_size = observation["depth"].shape[0]
        depth_map = observation["depth"].squeeze() * 255.0
        if not isinstance(depth_map, np.ndarray):
            depth_map = depth_map.cpu().numpy()

        depth_map = depth_map.astype(np.uint8)
        depth_map = np.stack([depth_map for _ in range(3)], axis=2)
        egocentric_view.append(depth_map)
        
    # draw semantic map if observation has depth info
    if "semantic" in observation:
        observation_size = observation["semantic"].shape[0]
        semantic_map = observation["semantic"].squeeze()
        if not isinstance(semantic_map, np.ndarray):
            semantic_map = semantic_map.cpu().numpy()

        semantic_map = (semantic_map + 1).astype(np.uint8)

        semantic_map = multion_maps.OBJECT_MAP_COLORS[semantic_map]
        egocentric_view.append(semantic_map)
    
    if info is not None and "top_down_map" in info and info["top_down_map"] is not None:
        top_down_map = info["top_down_map"]["map"]
        top_down_map = maps.colorize_topdown_map(
            top_down_map, info["top_down_map"]["fog_of_war_mask"]
        )
        map_agent_pos = info["top_down_map"]["agent_map_coord"]
        top_down_map = maps.draw_agent(
            image=top_down_map,
            agent_center_coord=map_agent_pos,
            agent_rotation=info["top_down_map"]["agent_angle"],
            agent_radius_px=top_down_map.shape[0] // 16,
        )

        if top_down_map.shape[0] > top_down_map.shape[1]:
            top_down_map = np.rot90(top_down_map, 1)

        # scale top down map to align with rgb view
        old_h, old_w, _ = top_down_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        top_down_map = cv2.resize(
            top_down_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(top_down_map)
        #frame = np.concatenate((frame, top_down_map), axis=1)
    
    if projected_features is not None and len(projected_features)>0:
        projected_features = cv2.resize(
            projected_features,
            depth_map.shape[:2],
            interpolation=cv2.INTER_CUBIC,
        )
        # projected_features /= np.max(projected_features)
        # projected_features  = cv2.applyColorMap(np.uint8(255 * projected_features), cv2.COLORMAP_JET)
        egocentric_view.append(projected_features)
        
    if semantic_projections is not None and len(semantic_projections)>0:
        if not isinstance(semantic_projections, np.ndarray):
            semantic_projections = semantic_projections.cpu().numpy()
        semantic_projections = semantic_projections + 1
        semantic_projections = (semantic_projections).astype(np.uint8)
        semantic_projections = cv2.resize(
            semantic_projections,
            depth_map.shape[:2],
            interpolation=cv2.INTER_NEAREST,
        )
        semantic_projections = multion_maps.OBJECT_MAP_COLORS[semantic_projections]
        #semantic_projections  = cv2.applyColorMap(np.uint8(255 * semantic_projections), cv2.COLORMAP_JET)
        egocentric_view.append(semantic_projections)

    if egocentric_projection is not None and len(egocentric_projection)>0:
        if not isinstance(egocentric_projection, np.ndarray):
            egocentric_projection = egocentric_projection.cpu().numpy()

        egocentric_projection = (egocentric_projection * 255).astype(np.uint8)
        egocentric_projection = np.stack([egocentric_projection for _ in range(3)], axis=2)
        egocentric_projection = cv2.resize(
            egocentric_projection,
            depth_map.shape[:2],
            interpolation=cv2.INTER_CUBIC,
        )

        egocentric_view.append(egocentric_projection)

    if global_map is not None and len(global_map)>0:
        global_map = cv2.resize(
            global_map,
            depth_map.shape[:2],
            interpolation=cv2.INTER_CUBIC,
        )
        egocentric_view.append(global_map)
    
    assert (
        len(egocentric_view) > 0
    ), "Expected at least one visual sensor enabled."
    egocentric_view = np.concatenate(egocentric_view, axis=1)

    # draw collision
    if info is not None and "collisions" in info and info["collisions"] is not None and info["collisions"]["is_collision"]:
        egocentric_view = draw_collision(egocentric_view)

    if action is not None and action[0] == 0:
        egocentric_view = draw_found(egocentric_view)

    frame = egocentric_view

    if "semMap" in observation:
        # Add occupancy map
        occ_map = observation["semMap"][:,:,0].squeeze()
        occ_map_size = occ_map.shape[0]
        if not isinstance(occ_map, np.ndarray):
            occ_map = occ_map.cpu().numpy()
        occ_map = occ_map.astype(np.uint8)
        occ_map[occ_map_size//2, occ_map_size//2] = 8
        occ_map = maps.colorize_topdown_map(occ_map)
        
        # scale map to align with rgb view
        old_h, old_w, _ = occ_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        occ_map = cv2.resize(
            occ_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, occ_map), axis=1)
        
        # Add goals/distractors map
        goal_map = observation["semMap"][:,:,1].squeeze()
        if not isinstance(goal_map, np.ndarray):
            goal_map = goal_map.cpu().numpy()
        goal_map = goal_map.astype(np.uint8)
        goal_map[occ_map_size//2, occ_map_size//2] = 8
        goal_map = maps.colorize_topdown_map(goal_map-1+multion_maps.MULTION_TOP_DOWN_MAP_START)
        
        # scale map to align with rgb view
        old_h, old_w, _ = goal_map.shape
        top_down_height = observation_size
        top_down_width = int(float(top_down_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        goal_map = cv2.resize(
            goal_map,
            (top_down_width, top_down_height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, goal_map), axis=1)
        
    if agent_view is not None:
        if not isinstance(agent_view, np.ndarray):
            agent_view = agent_view.cpu().numpy()
        agent_view[:,:,1] = agent_view[:,:,1] + 1
        agent_view = (agent_view[:,:,1:].max(axis=-1)).astype(np.uint8)
        agent_view = multion_maps.OBJECT_MAP_COLORS[agent_view]
        
        # scale map to align with rgb view
        old_h, old_w, _ = agent_view.shape
        _height = observation_size
        _width = int(float(_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        agent_view = cv2.resize(
            agent_view,
            (_width, _height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, agent_view), axis=1)
        
    if predicted_semantic is not None:
        observation_size = predicted_semantic.shape[0]
        if not isinstance(predicted_semantic, np.ndarray):
            predicted_semantic = predicted_semantic.cpu().numpy()

        predicted_semantic = (predicted_semantic + 1).squeeze().astype(np.uint8)

        predicted_semantic = multion_maps.OBJECT_MAP_COLORS[predicted_semantic]
        # scale map to align with rgb view
        old_h, old_w, _ = predicted_semantic.shape
        _height = observation_size
        _width = int(float(_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        predicted_semantic = cv2.resize(
            predicted_semantic,
            (_width, _height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, predicted_semantic), axis=1)
        
    if object_map is not None:
        if not isinstance(object_map, np.ndarray):
            object_map = object_map.cpu().numpy()
            
        # objects
        object_map[:,:,1] = object_map[:,:,1] + 1
        obj_map = (object_map[:,:,1:].max(axis=-1)).astype(np.uint8)
        obj_map = multion_maps.OBJECT_MAP_COLORS[obj_map]
        
        # scale map to align with rgb view
        old_h, old_w, _ = obj_map.shape
        _height = observation_size
        _width = int(float(_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        obj_map = cv2.resize(
            obj_map,
            (_width, _height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, obj_map), axis=1)
        
        # occupancy
        occ_map = (np.maximum(object_map[:,:,0],object_map[:,:,2])).astype(np.uint8) #object_map[:,:,0].astype(np.uint8)
        #occ_map = multion_maps.OCC_MAP_COLORS[occ_map]
        occ_map = multion_maps.OBJECT_MAP_COLORS[occ_map]
        
        # cv2 resize (dsize is width first)
        occ_map = cv2.resize(
            occ_map,
            (_width, _height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, occ_map), axis=1)
        
    if global_object_map is not None:
        if not isinstance(global_object_map, np.ndarray):
            global_object_map = global_object_map.cpu().numpy()
        global_object_map = global_object_map + 1
        global_object_map = (global_object_map[:,:,1:].max(axis=-1)).astype(np.uint8)
        global_object_map = multion_maps.OBJECT_MAP_COLORS[global_object_map]
        
        # scale map to align with rgb view
        old_h, old_w, _ = global_object_map.shape
        _height = observation_size
        _width = int(float(_height) / old_h * old_w)
        # cv2 resize (dsize is width first)
        global_object_map = cv2.resize(
            global_object_map,
            (_width, _height),
            interpolation=cv2.INTER_CUBIC,
        )
        frame = np.concatenate((frame, global_object_map), axis=1)
        

    return frame

def save_map_image(grid_map, depth, semantic,
                   all_agent_marks_x=None,
                   all_agent_marks_y=None,
                   agent_location_x=None,
                   agent_location_y=None,
                   file_name="") -> np.ndarray:
    r"""Generate image of single frame from observation and info
    returned from a single environment step().

    Args:
        observation: observation returned from an environment step().
        info: info returned from an environment step().
        action: action returned from an environment step().

    Returns:
        generated image of a single frame.
    """
    
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig.set_size_inches(13.5, 7)
    fig.subplots_adjust(wspace=0, hspace=0)
    
    #cmap = colors.ListedColormap(multion_maps.OBJECT_MAP_COLORS/255.)
    cmap = matplotlib.cm.get_cmap('Paired_r', 20)
    ax2.imshow(grid_map, cmap=cmap) #, vmin=0, vmax=1)
    #ax2.imshow(1-(grid_map), cmap='gray', vmin=0, vmax=1)
    
    ax0.imshow(depth, cmap='gray')
    ax1.imshow(semantic, cmap='gray')
    #ax2.plot(all_agent_marks_x, all_agent_marks_y, linestyle='-', color='green')
    #ax2.scatter(agent_location_x, agent_location_y, marker='*', color='red')
    ax0.axis('off')
    ax1.axis('off')
    ax2.axis('off')
    
    fig.savefig(os.path.join("test_maps", f"{file_name}.png"))
    #plt.close('all')
    
