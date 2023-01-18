#!/usr/bin/env python3

from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn
from einops import rearrange
import torch_scatter
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.contiguous().view(x.size(0), -1)

def extract_scalars_from_info(
    info: Dict[str, Any],
    metrics_blacklist: List[str] = None
) -> Dict[str, float]:
    
    result = {}
    if metrics_blacklist is None:
        metrics_blacklist = []
        
    for k, v in info.items():
        if not isinstance(k, str) or k in metrics_blacklist:
            continue

        if isinstance(v, dict):
            result.update(
                {
                    k + "." + subk: subv
                    for subk, subv in extract_scalars_from_info(
                        v
                    ).items()
                    if isinstance(subk, str)
                    and k + "." + subk not in metrics_blacklist
                }
            )
        # Things that are scalar-like will have an np.size of 1.
        # Strings also have an np.size of 1, so explicitly ban those
        elif np.size(v) == 1 and not isinstance(v, str):
            if isinstance(v, list):
                result[k] = float(v[0])
            else:
                result[k] = float(v)

    return result

class to_grid():
    def __init__(self, global_map_size, coordinate_min, coordinate_max):
        self.global_map_size = global_map_size
        self.coordinate_min = coordinate_min
        self.coordinate_max = coordinate_max
        self.grid_size = (coordinate_max - coordinate_min) / global_map_size

    def get_grid_coords(self, positions):
        grid_x = ((self.coordinate_max - positions[:, 0]) / self.grid_size).round()
        grid_y = ((positions[:, 1] - self.coordinate_min) / self.grid_size).round()
        return grid_x, grid_y

def draw_projection(image, 
                    depth,
                    config=None,
                    meters_per_pixel=None,  
                    s=None, 
                    global_map_size=None, 
                    coordinate_min=None, 
                    coordinate_max=None):
    image = torch.tensor(image).permute(0, 3, 1, 2)
    depth = torch.tensor(depth).permute(0, 3, 1, 2)
    spatial_locs, valid_inputs = _compute_spatial_locs(depth, config, meters_per_pixel, 
                                                       s, global_map_size, coordinate_min, coordinate_max)
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

def _compute_spatial_locs(depth_inputs, 
                          config=None,
                          meters_per_pixel=None, 
                          s=None, 
                          global_map_size=None, 
                          coordinate_min=None, 
                          coordinate_max=None):
    bs, _, imh, imw = depth_inputs.shape
    if meters_per_pixel is None:
        local_scale = float(coordinate_max - coordinate_min)/float(global_map_size)
    else:
        local_scale = float(meters_per_pixel)
        
    cx, cy = config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH/2., config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT/2.
    fx = fy =  (cx / 2.) / np.tan(np.deg2rad(config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV / 2.))

    #2D image coordinates
    x    = rearrange(torch.arange(0, imw), 'w -> () () () w').to(depth_inputs.device)
    y    = rearrange(torch.arange(imh, 0, step=-1), 'h -> () () h ()').to(depth_inputs.device)
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
    A = torch.zeros((x_gp.size(0), 2, 3), device=x_gp.device)
    A[:, 0, 0] = cos_t
    A[:, 0, 1] = sin_t
    A[:, 1, 0] = -sin_t
    A[:, 1, 1] = cos_t

    grid = F.affine_grid(A, x_gp.size(), align_corners=False)
    rotated_x_gp = F.grid_sample(x_gp, grid, mode='nearest', align_corners=False)
    return rotated_x_gp

def get_grid(pose, grid_size, device):
    """
    Input:
        `pose` FloatTensor(bs, 3)
        `grid_size` 4-tuple (bs, _, grid_h, grid_w)
        `device` torch.device (cpu or gpu)
    Output:
        `rot_grid` FloatTensor(bs, grid_h, grid_w, 2)
        `trans_grid` FloatTensor(bs, grid_h, grid_w, 2)
    """
    pose = pose.float()
    x = pose[:, 0]
    y = pose[:, 1]
    t = pose[:, 2]

    bs = x.size(0)
    cos_t = t.cos()
    sin_t = t.sin()

    theta11 = torch.stack([cos_t, -sin_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta12 = torch.stack([sin_t, cos_t,
                           torch.zeros(cos_t.shape).float().to(device)], 1)
    theta1 = torch.stack([theta11, theta12], 1)

    theta21 = torch.stack([torch.ones(x.shape).to(device),
                           -torch.zeros(x.shape).to(device), x], 1)
    theta22 = torch.stack([torch.zeros(x.shape).to(device),
                           torch.ones(x.shape).to(device), y], 1)
    theta2 = torch.stack([theta21, theta22], 1)

    rot_grid = F.affine_grid(theta1, torch.Size(grid_size), align_corners=False)
    trans_grid = F.affine_grid(theta2, torch.Size(grid_size), align_corners=False)

    return rot_grid, trans_grid
