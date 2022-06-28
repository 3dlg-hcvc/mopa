#!/usr/bin/env python3

from typing import Dict, List, Any
import numpy as np
import torch
import torch.nn as nn

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
