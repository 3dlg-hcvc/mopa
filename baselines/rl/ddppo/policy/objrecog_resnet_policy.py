#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from gym import spaces
from torch import nn as nn
from torch.nn import functional as F

from habitat.config import Config
from habitat.tasks.nav.nav import (
    EpisodicCompassSensor,
    EpisodicGPSSensor,
    HeadingSensor,
    ImageGoalSensor,
    IntegratedPointGoalGPSAndCompassSensor,
    PointGoalSensor,
    ProximitySensor,
)
from habitat.tasks.nav.object_nav_task import ObjectGoalSensor
from multion.sensors import (
    MultiObjectGoalSensor, 
    EpisodicGPSSensor as GPSSensor, 
    EpisodicCompassSensor as CompassSensor)
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.rl.ddppo.policy import resnet
from habitat_baselines.rl.ddppo.policy.running_mean_and_var import (
    RunningMeanAndVar,
)
from habitat_baselines.rl.models.rnn_state_encoder import (
    build_rnn_state_encoder,
)
from habitat_baselines.rl.ppo import Net, NetPolicy
from baselines.rl.ppo.policy import MapNetPolicy
from habitat_baselines.utils.common import get_num_actions

from baselines.rl.models.simple_cnn import MapCNN
from baselines.common.object_detector_cyl import ObjectDetector
import baselines.common.depth_utils as du
import baselines.common.rotation_utils as ru

@baseline_registry.register_policy
class ObjRecogResNetPolicy(MapNetPolicy):
    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int = 512,
        num_recurrent_layers: int = 1,
        rnn_type: str = "GRU",
        resnet_baseplanes: int = 32,
        backbone: str = "resnet18",
        normalize_visual_inputs: bool = False,
        force_blind_policy: bool = False,
        policy_config: Config = None,
        config: Config = None,
        fuse_keys: Optional[List[str]] = None,
        **kwargs
    ):
        if policy_config is not None:
            discrete_actions = (
                policy_config.action_distribution_type == "categorical"
            )
            self.action_distribution_type = (
                policy_config.action_distribution_type
            )
        else:
            discrete_actions = True
            self.action_distribution_type = "categorical"

        # if "has_rgb" in policy_config and policy_config.has_rgb:
        #     del observation_space.spaces['rgb']

        super().__init__(
            ObjRecogResNetNet(
                observation_space=observation_space,
                action_space=action_space,  # for previous action
                hidden_size=hidden_size,
                num_recurrent_layers=num_recurrent_layers,
                rnn_type=rnn_type,
                backbone=backbone,
                resnet_baseplanes=resnet_baseplanes,
                normalize_visual_inputs=normalize_visual_inputs,
                fuse_keys=fuse_keys,
                force_blind_policy=force_blind_policy,
                discrete_actions=discrete_actions,
                policy_config=policy_config,
                config=config
            ),
            dim_actions=get_num_actions(action_space),
            policy_config=policy_config,
        )

    @classmethod
    def from_config(
        cls,
        config: Config,
        observation_space: spaces.Dict,
        action_space,
    ):
        return cls(
            observation_space=observation_space,
            action_space=action_space,
            hidden_size=config.RL.PPO.hidden_size,
            rnn_type=config.RL.DDPPO.rnn_type,
            num_recurrent_layers=config.RL.DDPPO.num_recurrent_layers,
            backbone=config.RL.DDPPO.backbone,
            normalize_visual_inputs=("rgb" in observation_space.spaces and "has_rgb" in config.RL.POLICY and config.RL.POLICY.has_rgb),
            force_blind_policy=config.FORCE_BLIND_POLICY,
            policy_config=config.RL.POLICY,
            config=config,
            fuse_keys=config.TASK_CONFIG.GYM.OBS_KEYS,
        )


class ResNetEncoder(nn.Module):
    def __init__(
        self,
        observation_space: spaces.Dict,
        baseplanes: int = 32,
        ngroups: int = 32,
        spatial_size: int = 128,
        make_backbone=None,
        normalize_visual_inputs: bool = False,
        has_rgb: bool = True
    ):
        super().__init__()

        self.has_rgb = has_rgb
        # Determine which visual observations are present
        self.rgb_keys = [k for k in observation_space.spaces if "rgb" in k and has_rgb]
        self.depth_keys = [k for k in observation_space.spaces if "depth" in k]

        # Count total # of channels for rgb and for depth
        self._n_input_rgb, self._n_input_depth = [
            # sum() returns 0 for an empty list
            sum([observation_space.spaces[k].shape[2] for k in keys])
            for keys in [self.rgb_keys, self.depth_keys]
        ]

        if normalize_visual_inputs:
            self.running_mean_and_var: nn.Module = RunningMeanAndVar(
                self._n_input_depth + self._n_input_rgb
            )
        else:
            self.running_mean_and_var = nn.Sequential()

        if not self.is_blind:
            all_keys = self.rgb_keys + self.depth_keys
            spatial_size_h = (
                observation_space.spaces[all_keys[0]].shape[0] // 2
            )
            spatial_size_w = (
                observation_space.spaces[all_keys[0]].shape[1] // 2
            )
            input_channels = self._n_input_depth + self._n_input_rgb
            self.backbone = make_backbone(input_channels, baseplanes, ngroups)

            final_spatial_h = int(
                np.ceil(spatial_size_h * self.backbone.final_spatial_compress)
            )
            final_spatial_w = int(
                np.ceil(spatial_size_w * self.backbone.final_spatial_compress)
            )
            after_compression_flat_size = 2048
            num_compression_channels = int(
                round(
                    after_compression_flat_size
                    / (final_spatial_h * final_spatial_w)
                )
            )
            self.compression = nn.Sequential(
                nn.Conv2d(
                    self.backbone.final_channels,
                    num_compression_channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.GroupNorm(1, num_compression_channels),
                nn.ReLU(True),
            )

            self.output_shape = (
                num_compression_channels,
                final_spatial_h,
                final_spatial_w,
            )

    @property
    def is_blind(self):
        return self._n_input_rgb + self._n_input_depth == 0

    def layer_init(self):
        for layer in self.modules():
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                nn.init.kaiming_normal_(
                    layer.weight, nn.init.calculate_gain("relu")
                )
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, val=0)

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:  # type: ignore
        if self.is_blind:
            return None

        cnn_input = []
        for k in self.rgb_keys:
            rgb_observations = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            rgb_observations = rgb_observations.permute(0, 3, 1, 2)
            rgb_observations = (
                rgb_observations.float() / 255.0
            )  # normalize RGB
            cnn_input.append(rgb_observations)

        for k in self.depth_keys:
            depth_observations = observations[k]
            # permute tensor to dimension [BATCH x CHANNEL x HEIGHT X WIDTH]
            depth_observations = depth_observations.permute(0, 3, 1, 2)
            cnn_input.append(depth_observations)

        x = torch.cat(cnn_input, dim=1)
        x = F.avg_pool2d(x, 2)

        x = self.running_mean_and_var(x)
        x = self.backbone(x)
        x = self.compression(x)
        return x


class ObjRecogResNetNet(Net):
    """Network which passes the input image through CNN and concatenates
    goal vector with CNN's output and passes that through RNN.
    """

    prev_action_embedding: nn.Module

    def __init__(
        self,
        observation_space: spaces.Dict,
        action_space,
        hidden_size: int,
        num_recurrent_layers: int,
        rnn_type: str,
        backbone,
        resnet_baseplanes,
        normalize_visual_inputs: bool,
        fuse_keys: Optional[List[str]],
        force_blind_policy: bool = False,
        discrete_actions: bool = True,
        policy_config: Config = None,
        config: Config = None,
    ):
        super().__init__()
        self.prev_action_embedding: nn.Module
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.prev_action_embedding = nn.Embedding(action_space.n + 1, 32)
        else:
            num_actions = get_num_actions(action_space)
            self.prev_action_embedding = nn.Linear(num_actions, 32)

        self._n_prev_action = 32
        rnn_input_size = self._n_prev_action  # test
        
        self.policy_config = policy_config
        
        # Camera config
        self.config = config
        self.camera = du.get_camera_matrix(
                        self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HEIGHT, 
                        self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.WIDTH, 
                        self.config.TASK_CONFIG.SIMULATOR.DEPTH_SENSOR.HFOV)
        self.elevation = 0. #np.rad2deg(env_config.SIMULATOR.DEPTH_SENSOR.ORIENTATION[0])
        self.camera_height = self.config.TASK_CONFIG.SIMULATOR.AGENT_0.HEIGHT
        self.map_resolution = self.config.RL.MAP.map_resolution
        self.meters_covered = self.config.RL.MAP.meters_covered
        self.map_grid_size = np.round(self.meters_covered / self.map_resolution).astype(int)
        self.map_center = np.array([np.round(self.map_grid_size/2.).astype(int), np.round(self.map_grid_size/2.).astype(int)])
        self.z_bins = [0.5, 1.5]
        # End - Camera config

        # Only fuse the 1D state inputs. Other inputs are processed by the
        # visual encoder
        self._fuse_keys: List[str] = (
            [
                k
                for k in fuse_keys
                if len(observation_space.spaces[k].shape) == 1
            ]
            if fuse_keys is not None
            else []
        )
        if len(self._fuse_keys) != 0:
            rnn_input_size += sum(
                [observation_space.spaces[k].shape[0] for k in self._fuse_keys]
            )

        if (
            IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            in observation_space.spaces
        ):
            n_input_goal = (
                observation_space.spaces[
                    IntegratedPointGoalGPSAndCompassSensor.cls_uuid
                ].shape[0]
                + 1
            )
            self.tgt_embeding = nn.Linear(n_input_goal, 32)
            rnn_input_size += 32

        if ObjectGoalSensor.cls_uuid in observation_space.spaces:
            self._n_object_categories = (
                int(
                    observation_space.spaces[ObjectGoalSensor.cls_uuid].high[0]
                )
                + 1
            )
            self.obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            
        if MultiObjectGoalSensor.cls_uuid in observation_space.spaces:
            # Goal embedding
            self._n_object_categories = (
                int(
                    observation_space.spaces[MultiObjectGoalSensor.cls_uuid].high[0]
                ) + 1
            )
            self.multi_obj_categories_embedding = nn.Embedding(
                self._n_object_categories, 32
            )
            rnn_input_size += 32
            
            # Object embedding on the map
            self.total_num_embedding = (self._n_object_categories # objects
                                        + 2 # occupancy
                                        + 1  # agent
                                        # + 1 # current goal
                                        + 1
                                        )
            self.object_embedding = nn.Embedding(self.total_num_embedding, 32)
            
            # Map encoder
            self.enc_output_size = self.config.RL.MAP.enc_output_size
            if "pred_labels" in policy_config and policy_config.pred_labels:
                # use predicted
                self.map_size = self.config.RL.MAP.map_size
                self._detector = ObjectDetector()
            else:
                # use oracle
                self.map_size = observation_space.spaces["object_map"].shape[0]
                self._detector = None
                
            _n_input_map = 32 * self.config.RL.MAP.map_depth
            self.map_encoder = MapCNN(self.map_size, 
                                        self.enc_output_size, 
                                        agent_type="obj-recog",
                                        _n_input_map=_n_input_map)
            rnn_input_size += self.enc_output_size

        if EpisodicGPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                EpisodicGPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32
        
        if GPSSensor.cls_uuid in observation_space.spaces:
            input_gps_dim = observation_space.spaces[
                GPSSensor.cls_uuid
            ].shape[0]
            self.gps_embedding = nn.Linear(input_gps_dim, 32)
            rnn_input_size += 32

        if PointGoalSensor.cls_uuid in observation_space.spaces:
            input_pointgoal_dim = observation_space.spaces[
                PointGoalSensor.cls_uuid
            ].shape[0]
            self.pointgoal_embedding = nn.Linear(input_pointgoal_dim, 32)
            rnn_input_size += 32

        if HeadingSensor.cls_uuid in observation_space.spaces:
            input_heading_dim = (
                observation_space.spaces[HeadingSensor.cls_uuid].shape[0] + 1
            )
            assert input_heading_dim == 2, "Expected heading with 2D rotation."
            self.heading_embedding = nn.Linear(input_heading_dim, 32)
            rnn_input_size += 32

        if ProximitySensor.cls_uuid in observation_space.spaces:
            input_proximity_dim = observation_space.spaces[
                ProximitySensor.cls_uuid
            ].shape[0]
            self.proximity_embedding = nn.Linear(input_proximity_dim, 32)
            rnn_input_size += 32

        if EpisodicCompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[EpisodicCompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32
            
        if CompassSensor.cls_uuid in observation_space.spaces:
            assert (
                observation_space.spaces[CompassSensor.cls_uuid].shape[
                    0
                ]
                == 1
            ), "Expected compass with 2D rotation."
            input_compass_dim = 2  # cos and sin of the angle
            self.compass_embedding = nn.Linear(input_compass_dim, 32)
            rnn_input_size += 32

        if ImageGoalSensor.cls_uuid in observation_space.spaces:
            goal_observation_space = spaces.Dict(
                {"rgb": observation_space.spaces[ImageGoalSensor.cls_uuid]}
            )
            self.goal_visual_encoder = ResNetEncoder(
                goal_observation_space,
                baseplanes=resnet_baseplanes,
                ngroups=resnet_baseplanes // 2,
                make_backbone=getattr(resnet, backbone),
                normalize_visual_inputs=normalize_visual_inputs,
            )

            self.goal_visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.goal_visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

            rnn_input_size += hidden_size

        self._hidden_size = hidden_size

        if force_blind_policy:
            use_obs_space = spaces.Dict({})
        else:
            use_obs_space = (
                spaces.Dict(
                    {
                        k: observation_space.spaces[k]
                        for k in fuse_keys
                        if len(observation_space.spaces[k].shape) == 3
                    }
                )
                if fuse_keys is not None
                else observation_space
            )

        self.visual_encoder = ResNetEncoder(
            use_obs_space,
            baseplanes=resnet_baseplanes,
            ngroups=resnet_baseplanes // 2,
            make_backbone=getattr(resnet, backbone),
            normalize_visual_inputs=normalize_visual_inputs,
            has_rgb=policy_config.has_rgb if "has_rgb" in policy_config else False
        )

        if not self.visual_encoder.is_blind:
            self.visual_fc = nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    np.prod(self.visual_encoder.output_shape), hidden_size
                ),
                nn.ReLU(True),
            )

        self.state_encoder = build_rnn_state_encoder(
            (0 if self.is_blind else self._hidden_size) + rnn_input_size,
            self._hidden_size,
            rnn_type=rnn_type,
            num_layers=num_recurrent_layers,
        )

        self.train()

    @property
    def output_size(self):
        return self._hidden_size

    @property
    def is_blind(self):
        return self.visual_encoder.is_blind

    @property
    def num_recurrent_layers(self):
        return self.state_encoder.num_recurrent_layers

    def forward(
        self,
        observations: Dict[str, torch.Tensor],
        rnn_hidden_states,
        prev_actions,
        masks,
        object_map,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        x = []
        if not self.is_blind:
            visual_feats = observations.get(
                "visual_features", self.visual_encoder(observations)
            )
            visual_feats = self.visual_fc(visual_feats)
            x.append(visual_feats)

        if len(self._fuse_keys) != 0:
            fuse_states = torch.cat(
                [observations[k] for k in self._fuse_keys], dim=-1
            )
            x.append(fuse_states)

        if IntegratedPointGoalGPSAndCompassSensor.cls_uuid in observations:
            goal_observations = observations[
                IntegratedPointGoalGPSAndCompassSensor.cls_uuid
            ]
            if goal_observations.shape[1] == 2:
                # Polar Dimensionality 2
                # 2D polar transform
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1]),
                        torch.sin(-goal_observations[:, 1]),
                    ],
                    -1,
                )
            else:
                assert (
                    goal_observations.shape[1] == 3
                ), "Unsupported dimensionality"
                vertical_angle_sin = torch.sin(goal_observations[:, 2])
                # Polar Dimensionality 3
                # 3D Polar transformation
                goal_observations = torch.stack(
                    [
                        goal_observations[:, 0],
                        torch.cos(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.sin(-goal_observations[:, 1])
                        * vertical_angle_sin,
                        torch.cos(goal_observations[:, 2]),
                    ],
                    -1,
                )

            x.append(self.tgt_embeding(goal_observations))

        if PointGoalSensor.cls_uuid in observations:
            goal_observations = observations[PointGoalSensor.cls_uuid]
            x.append(self.pointgoal_embedding(goal_observations))

        if ProximitySensor.cls_uuid in observations:
            sensor_observations = observations[ProximitySensor.cls_uuid]
            x.append(self.proximity_embedding(sensor_observations))

        if HeadingSensor.cls_uuid in observations:
            sensor_observations = observations[HeadingSensor.cls_uuid]
            sensor_observations = torch.stack(
                [
                    torch.cos(sensor_observations[0]),
                    torch.sin(sensor_observations[0]),
                ],
                -1,
            )
            x.append(self.heading_embedding(sensor_observations))

        if ObjectGoalSensor.cls_uuid in observations:
            object_goal = observations[ObjectGoalSensor.cls_uuid].long()
            x.append(self.obj_categories_embedding(object_goal).squeeze(dim=1))
            
        if MultiObjectGoalSensor.cls_uuid in observations:
            multi_object_goal = observations[MultiObjectGoalSensor.cls_uuid].long()
            x.append(self.multi_obj_categories_embedding(multi_object_goal).squeeze(dim=1))
            
            bs = multi_object_goal.shape[0]
            global_map_embedding = []
            global_occ_map_embedding = []
            agent_position_embedding = []
            
            if "pred_labels" in self.policy_config and self.policy_config.pred_labels:
                object_map = self.build_map(observations, object_map).type(torch.long)
                # global_object_map = object_map[:, :, :, 1].type(torch.long)
                # global_occ_map = object_map[:, :, :, 0].type(torch.long)
                # agent_pos_map = object_map[:, :, :, 2].type(torch.long)
            else:
                object_map = observations["object_map"].type(torch.long)
                # global_object_map = _map[:, :, :, 1]
                # global_occ_map = _map[:, :, :, 0]
                # agent_pos_map = _map[:, :, :, 2]
                
            global_map_embedding.append(self.object_embedding(object_map.view(-1))
                                            .view(bs, self.map_size, self.map_size, -1))
            global_map_embedding = torch.cat(global_map_embedding, dim=-1)
            map_embed = self.map_encoder(global_map_embedding)
            x.append(map_embed.squeeze(dim=1))

        if EpisodicCompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[EpisodicCompassSensor.cls_uuid]),
                    torch.sin(observations[EpisodicCompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )
            
        if CompassSensor.cls_uuid in observations:
            compass_observations = torch.stack(
                [
                    torch.cos(observations[CompassSensor.cls_uuid]),
                    torch.sin(observations[CompassSensor.cls_uuid]),
                ],
                -1,
            )
            x.append(
                self.compass_embedding(compass_observations.squeeze(dim=1))
            )

        if EpisodicGPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[EpisodicGPSSensor.cls_uuid])
            )
            
        if GPSSensor.cls_uuid in observations:
            x.append(
                self.gps_embedding(observations[GPSSensor.cls_uuid])
            )

        if ImageGoalSensor.cls_uuid in observations:
            goal_image = observations[ImageGoalSensor.cls_uuid]
            goal_output = self.goal_visual_encoder({"rgb": goal_image})
            x.append(self.goal_visual_fc(goal_output))

        if self.discrete_actions:
            prev_actions = prev_actions.squeeze(-1)
            start_token = torch.zeros_like(prev_actions)
            prev_actions = self.prev_action_embedding(
                torch.where(masks.view(-1), prev_actions + 1, start_token)
            )
        else:
            prev_actions = self.prev_action_embedding(
                masks * prev_actions.float()
            )

        x.append(prev_actions)

        out = torch.cat(x, dim=1)
        out, rnn_hidden_states = self.state_encoder(
            out, rnn_hidden_states, masks
        )

        return out, rnn_hidden_states, object_map
    
    def build_map(self, observations, object_map):
        depth = (observations['depth'] * 10).squeeze(-1).cpu().numpy()
        #depth = (observations['depth']).squeeze(-1).cpu().numpy()
        depth[depth == 0] = np.NaN
        #depth[depth > 10] = np.NaN

        #gt_semantic = observations['semantic'].squeeze(-1).cpu().numpy()
        theta = observations['episodic_compass'].cpu().numpy()
        location = observations["episodic_gps"].cpu().numpy()
        semantic = np.zeros_like(depth)
        results = self._detector.predict(observations['rgb'].squeeze(-1))
        for i, res in enumerate(results):
            if len(res['boxes'])> 0:
                gt_bbox = []
                for j, b in enumerate(res['boxes']):
                    b = np.round(b).astype(int)
                    semantic[i, b[1]:b[3], b[0]:b[2]] = res["labels"][j] + 1  # object semantic labels start at 1
                    
        semantic = semantic[..., np.newaxis]
        semantic += self.config.RL.MAP.object_ind_offset   # object labels are 3-10
        
        coords = self._unproject_to_world(depth, location, theta)
        grid_map = self._add_to_map(coords, semantic, object_map.cpu().numpy())
        agent_locs = self.to_grid(location)
        object_map[:, :, :, :2] = torch.tensor(grid_map)[:, :, :, :2]
        
        # mark agent
        object_map[:, :, :, 2] = 0
        object_map[:,agent_locs[:,0],agent_locs[:,1],2] = 11
        
        return object_map  #, _agent_locs, semantic
    
    def _unproject_to_world(self, depth, location, theta):
        point_cloud = du.get_point_cloud_from_z(depth, self.camera)

        agent_view = du.transform_camera_view(point_cloud,
                                              self.camera_height, self.elevation)

        geocentric_pc = du.transform_pose(agent_view, location, theta)

        return geocentric_pc
    
    def _add_to_map(self, coords, semantic, grid_map):
        XYZS = np.concatenate((coords, semantic),axis=-1)
        depth_counts, sem_map_counts = du.bin_points_w_sem(
            XYZS,
            self.map_grid_size,
            self.z_bins,
            self.map_resolution,
            self.map_center)

        map = grid_map[:, :, :, 0] + depth_counts[:, :, :, 1]
        map[map < 1] = 0.0
        map[map >= 1] = 1.0
        grid_map[:, :, :, 0] = map
        
        grid_map[:, :, :, 1] = np.maximum(grid_map[:, :, :, 1], sem_map_counts[:, :, :, 1])
        #grid_map[:, :, :, 1] = sem_map_counts[:, :, :, 1]

        return grid_map
        
    def to_grid(self, xy):
        return (np.round(xy / self.map_resolution) + self.map_center).astype(int)
        
    def from_grid(self, grid_x, grid_y):
        return [
            (grid_x - self.map_center[0]) * self.map_resolution,
            (grid_y - self.map_center[1]) * self.map_resolution,
            ]
    