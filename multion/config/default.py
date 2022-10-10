#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import List, Optional, Union

import yacs.config


# Default Habitat config node
class Config(yacs.config.CfgNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, new_allowed=True)


CN = Config

DEFAULT_CONFIG_DIR = "configs/"
CONFIG_FILE_SEPARATOR = ","

# -----------------------------------------------------------------------------
# Config definition
# -----------------------------------------------------------------------------
_C = CN()
_C.SEED = 100
# -----------------------------------------------------------------------------
# ENVIRONMENT
# -----------------------------------------------------------------------------
_C.ENVIRONMENT = CN()
_C.ENVIRONMENT.MAX_EPISODE_STEPS = 1000
_C.ENVIRONMENT.MAX_EPISODE_SECONDS = 10000000
_C.ENVIRONMENT.ITERATOR_OPTIONS = CN()
_C.ENVIRONMENT.ITERATOR_OPTIONS.CYCLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.GROUP_BY_SCENE = True
_C.ENVIRONMENT.ITERATOR_OPTIONS.NUM_EPISODE_SAMPLE = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_EPISODES = -1
_C.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = int(1e4)
_C.ENVIRONMENT.ITERATOR_OPTIONS.STEP_REPETITION_RANGE = 0.2
# -----------------------------------------------------------------------------
# TASK
# -----------------------------------------------------------------------------
_C.TRAINER_NAME = ""
# -----------------------------------------------------------------------------
# # NAVIGATION TASK
# -----------------------------------------------------------------------------
_C.TASK = CN()
_C.TASK.TYPE = "MultiObjectNav-v1"
_C.TASK.SUCCESS_DISTANCE = 0.2
_C.TASK.SENSORS = []
_C.TASK.MEASUREMENTS = ['DISTANCE_TO_GOAL','DISTANCE_TO_CURRENT_OBJECT_GOAL', 'CURRENT_GOAL_SUCCESS', 'PROGRESS', 'MULTION_SUCCESS', 'MULTION_PPL', 'MULTION_SPL']
_C.TASK.GOAL_SENSOR_UUID = "multiobjectgoal"
_C.TASK.POSSIBLE_ACTIONS = ["STOP", "MOVE_FORWARD", "TURN_LEFT", "TURN_RIGHT"]
# -----------------------------------------------------------------------------
# # ACTIONS
# -----------------------------------------------------------------------------
ACTIONS = CN()
ACTIONS.STOP = CN()
ACTIONS.STOP.TYPE = "StopAction"
# -----------------------------------------------------------------------------
# # NAVIGATION ACTIONS
# -----------------------------------------------------------------------------
ACTIONS.MOVE_FORWARD = CN()
ACTIONS.MOVE_FORWARD.TYPE = "MoveForwardAction"
ACTIONS.TURN_LEFT = CN()
ACTIONS.TURN_LEFT.TYPE = "TurnLeftAction"
ACTIONS.TURN_RIGHT = CN()
ACTIONS.TURN_RIGHT.TYPE = "TurnRightAction"
ACTIONS.LOOK_UP = CN()
ACTIONS.LOOK_UP.TYPE = "LookUpAction"
ACTIONS.LOOK_DOWN = CN()
ACTIONS.LOOK_DOWN.TYPE = "LookDownAction"
ACTIONS.TELEPORT = CN()
ACTIONS.TELEPORT.TYPE = "TeleportAction"
ACTIONS.VELOCITY_CONTROL = CN()
ACTIONS.VELOCITY_CONTROL.TYPE = "VelocityAction"
ACTIONS.VELOCITY_CONTROL.LIN_VEL_RANGE = [0.0, 0.25]  # meters per sec
ACTIONS.VELOCITY_CONTROL.ANG_VEL_RANGE = [-10.0, 10.0]  # deg per sec
ACTIONS.VELOCITY_CONTROL.MIN_ABS_LIN_SPEED = 0.025  # meters per sec
ACTIONS.VELOCITY_CONTROL.MIN_ABS_ANG_SPEED = 1.0  # deg per sec
ACTIONS.VELOCITY_CONTROL.TIME_STEP = 1.0  # seconds

_C.TASK.ACTIONS = ACTIONS
# -----------------------------------------------------------------------------
# # TASK SENSORS
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# POINTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_SENSOR = CN()
_C.TASK.POINTGOAL_SENSOR.TYPE = "PointGoalSensor"
_C.TASK.POINTGOAL_SENSOR.GOAL_FORMAT = "POLAR"
_C.TASK.POINTGOAL_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# POINTGOAL WITH GPS+COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR = _C.TASK.POINTGOAL_SENSOR.clone()
_C.TASK.POINTGOAL_WITH_GPS_COMPASS_SENSOR.TYPE = (
    "PointGoalWithGPSCompassSensor"
)
# -----------------------------------------------------------------------------
# OBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECTGOAL_SENSOR = CN()
_C.TASK.OBJECTGOAL_SENSOR.TYPE = "ObjectGoalSensor"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC = "TASK_CATEGORY_ID"
_C.TASK.OBJECTGOAL_SENSOR.GOAL_SPEC_MAX_VAL = 50
# -----------------------------------------------------------------------------
# MULTIOBJECTGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.MULTI_OBJECT_GOAL_SENSOR = CN()
_C.TASK.MULTI_OBJECT_GOAL_SENSOR.TYPE = "MultiObjectGoalSensor"
_C.TASK.MULTI_OBJECT_GOAL_SENSOR.GOAL_SPEC = "TASK_CATEGORY_ID"
_C.TASK.MULTI_OBJECT_GOAL_SENSOR.GOAL_SPEC_MAX_VAL = 50
# -----------------------------------------------------------------------------
# IMAGEGOAL SENSOR
# -----------------------------------------------------------------------------
_C.TASK.IMAGEGOAL_SENSOR = CN()
_C.TASK.IMAGEGOAL_SENSOR.TYPE = "ImageGoalSensor"
# -----------------------------------------------------------------------------
# POSITION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.POSITION_SENSOR = CN()
_C.TASK.POSITION_SENSOR.TYPE = "PositionSensor"
# -----------------------------------------------------------------------------
# HEADING SENSOR
# -----------------------------------------------------------------------------
_C.TASK.HEADING_SENSOR = CN()
_C.TASK.HEADING_SENSOR.TYPE = "HeadingSensor"
# -----------------------------------------------------------------------------
# COMPASS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.COMPASS_SENSOR = CN()
_C.TASK.COMPASS_SENSOR.TYPE = "CompassSensor"
# -----------------------------------------------------------------------------
# GPS SENSOR
# -----------------------------------------------------------------------------
_C.TASK.GPS_SENSOR = CN()
_C.TASK.GPS_SENSOR.TYPE = "GPSSensor"
_C.TASK.GPS_SENSOR.DIMENSIONALITY = 2
# -----------------------------------------------------------------------------
# AGENT ROTATION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ROTATION_SENSOR = CN()
_C.TASK.ROTATION_SENSOR.TYPE = "RotationSensor"
# -----------------------------------------------------------------------------
# PROXIMITY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.PROXIMITY_SENSOR = CN()
_C.TASK.PROXIMITY_SENSOR.TYPE = "ProximitySensor"
_C.TASK.PROXIMITY_SENSOR.MAX_DETECTION_RADIUS = 2.0
# -----------------------------------------------------------------------------
# SEMANTIC MAP SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SEMANTIC_MAP_SENSOR = CN()
_C.TASK.SEMANTIC_MAP_SENSOR.TYPE = "SemanticMapSensor"
_C.TASK.SEMANTIC_MAP_SENSOR.METERS_PER_PIXEL = 0.5
_C.TASK.SEMANTIC_MAP_SENSOR.EGOCENTRIC_MAP_SIZE = 51
_C.TASK.SEMANTIC_MAP_SENSOR.CROPPED_MAP_SIZE = 80
_C.TASK.SEMANTIC_MAP_SENSOR.MAP_CHANNELS = 3
# -----------------------------------------------------------------------------
# SEMANTIC OBJECTS VISIBLE SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SEMANTIC_OBJECTS_VISIBLE_SENSOR = CN()
_C.TASK.SEMANTIC_OBJECTS_VISIBLE_SENSOR.TYPE = "SemanticObjectVisibleSensor"
_C.TASK.SEMANTIC_OBJECTS_VISIBLE_SENSOR.TOTAL_NUM_OBJECTS = 9
# -----------------------------------------------------------------------------
# SEMANTIC OCCUPANCY SENSOR
# -----------------------------------------------------------------------------
_C.TASK.SEM_OCC_SENSOR = CN()
_C.TASK.SEM_OCC_SENSOR.TYPE = "SemOccSensor"
_C.TASK.SEM_OCC_SENSOR.pregenerated = False
_C.TASK.SEM_OCC_SENSOR.pregenerated_file_path = "data/oracle_maps/map300.pickle"
_C.TASK.SEM_OCC_SENSOR.meters_per_pixel = 0.3
_C.TASK.SEM_OCC_SENSOR.num_samples = 50
_C.TASK.SEM_OCC_SENSOR.nav_threshold = 0.3
_C.TASK.SEM_OCC_SENSOR.MAP_CHANNELS = 3
_C.TASK.SEM_OCC_SENSOR.draw_border = False
_C.TASK.SEM_OCC_SENSOR.with_sampling = True
_C.TASK.SEM_OCC_SENSOR.mask_map = False
_C.TASK.SEM_OCC_SENSOR.cache_max_size = 2
_C.TASK.SEM_OCC_SENSOR.map_size = 50

# Goals
_C.TASK.SEM_OCC_SENSOR.channel_num_goals = 1
_C.TASK.SEM_OCC_SENSOR.objIndexOffset = 1

# Distractors
_C.TASK.SEM_OCC_SENSOR.INCLUDE_DISTRACTORS = False
_C.TASK.SEM_OCC_SENSOR.ORACLE_MAP_INCLUDE_DISTRACTORS_W_GOAL = False
_C.TASK.SEM_OCC_SENSOR.num_distractors_included = -1
# -----------------------------------------------------------------------------
# Object Map SENSOR
# -----------------------------------------------------------------------------
_C.TASK.OBJECT_MAP_SENSOR = CN()
_C.TASK.OBJECT_MAP_SENSOR.TYPE = "ObjectMapSensor"
_C.TASK.OBJECT_MAP_SENSOR.meters_per_pixel = 0.3
_C.TASK.OBJECT_MAP_SENSOR.num_samples = 50
_C.TASK.OBJECT_MAP_SENSOR.nav_threshold = 0.3
_C.TASK.OBJECT_MAP_SENSOR.MAP_CHANNELS = 3
_C.TASK.OBJECT_MAP_SENSOR.draw_border = False
_C.TASK.OBJECT_MAP_SENSOR.with_sampling = True
_C.TASK.OBJECT_MAP_SENSOR.mask_map = False
_C.TASK.OBJECT_MAP_SENSOR.cache_max_size = 2
_C.TASK.OBJECT_MAP_SENSOR.map_size = 50
_C.TASK.OBJECT_MAP_SENSOR.mask_map = False
_C.TASK.OBJECT_MAP_SENSOR.VISIBILITY_DIST = 5.0
_C.TASK.OBJECT_MAP_SENSOR.FOV = 80
_C.TASK.OBJECT_MAP_SENSOR.object_padding = 2
_C.TASK.OBJECT_MAP_SENSOR.object_ind_offset = 2
# -----------------------------------------------------------------------------
# Oracle Map Size SENSOR
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_MAP_SIZE_SENSOR = CN()
_C.TASK.ORACLE_MAP_SIZE_SENSOR.TYPE = "OracleMapSizeSensor"
_C.TASK.ORACLE_MAP_SIZE_SENSOR.meters_per_pixel = _C.TASK.OBJECT_MAP_SENSOR.meters_per_pixel
_C.TASK.ORACLE_MAP_SIZE_SENSOR.num_samples = _C.TASK.OBJECT_MAP_SENSOR.num_samples
_C.TASK.ORACLE_MAP_SIZE_SENSOR.nav_threshold = _C.TASK.OBJECT_MAP_SENSOR.nav_threshold
_C.TASK.ORACLE_MAP_SIZE_SENSOR.draw_border = _C.TASK.OBJECT_MAP_SENSOR.draw_border
_C.TASK.ORACLE_MAP_SIZE_SENSOR.with_sampling = _C.TASK.OBJECT_MAP_SENSOR.with_sampling
_C.TASK.ORACLE_MAP_SIZE_SENSOR.cache_max_size = 2
# -----------------------------------------------------------------------------
# SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SUCCESS = CN()
_C.TASK.SUCCESS.TYPE = "Success"
_C.TASK.SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SPL = CN()
_C.TASK.SPL.TYPE = "SPL"
# -----------------------------------------------------------------------------
# SOFT-SPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SOFT_SPL = CN()
_C.TASK.SOFT_SPL.TYPE = "SoftSPL"
# -----------------------------------------------------------------------------
### FOW ####
# -----------------------------------------------------------------------------
_C.TASK.FOW_MAP = CN()
_C.TASK.FOW_MAP.TYPE = "FowMap"
_C.TASK.FOW_MAP.VISIBILITY_DIST = 6.0
_C.TASK.FOW_MAP.FOV = 80
# -----------------------------------------------------------------------------
# TopDownMap MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.TOP_DOWN_MAP = CN()
_C.TASK.TOP_DOWN_MAP.TYPE = "TopDownMap"
_C.TASK.TOP_DOWN_MAP.MAX_EPISODE_STEPS = _C.ENVIRONMENT.MAX_EPISODE_STEPS
_C.TASK.TOP_DOWN_MAP.MAP_PADDING = 3
_C.TASK.TOP_DOWN_MAP.NUM_TOPDOWN_MAP_SAMPLE_POINTS = 20000
_C.TASK.TOP_DOWN_MAP.MAP_RESOLUTION = 1250
_C.TASK.TOP_DOWN_MAP.DRAW_SOURCE = True
_C.TASK.TOP_DOWN_MAP.DRAW_BORDER = True
_C.TASK.TOP_DOWN_MAP.DRAW_SHORTEST_PATH = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR = CN()
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.DRAW = True
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.VISIBILITY_DIST = 5.0
_C.TASK.TOP_DOWN_MAP.FOG_OF_WAR.FOV = 90
_C.TASK.TOP_DOWN_MAP.DRAW_VIEW_POINTS = True
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_POSITIONS = True
# Axes aligned bounding boxes
_C.TASK.TOP_DOWN_MAP.DRAW_GOAL_AABBS = True
_C.TASK.TOP_DOWN_MAP.DRAW_DISTRACTORS = True
# -----------------------------------------------------------------------------
# COLLISIONS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.COLLISIONS = CN()
_C.TASK.COLLISIONS.TYPE = "Collisions"
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# # EQA TASK
# -----------------------------------------------------------------------------
_C.TASK.ACTIONS.ANSWER = CN()
_C.TASK.ACTIONS.ANSWER.TYPE = "AnswerAction"
# # EQA TASK QUESTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.QUESTION_SENSOR = CN()
_C.TASK.QUESTION_SENSOR.TYPE = "QuestionSensor"
# -----------------------------------------------------------------------------
# # EQA TASK CORRECT_ANSWER measure for training
# -----------------------------------------------------------------------------
_C.TASK.CORRECT_ANSWER = CN()
_C.TASK.CORRECT_ANSWER.TYPE = "CorrectAnswer"
# -----------------------------------------------------------------------------
# # EQA TASK ANSWER SENSOR
# -----------------------------------------------------------------------------
_C.TASK.EPISODE_INFO = CN()
_C.TASK.EPISODE_INFO.TYPE = "EpisodeInfo"
# -----------------------------------------------------------------------------
# # VLN TASK INSTRUCTION SENSOR
# -----------------------------------------------------------------------------
_C.TASK.INSTRUCTION_SENSOR = CN()
_C.TASK.INSTRUCTION_SENSOR.TYPE = "InstructionSensor"
_C.TASK.INSTRUCTION_SENSOR_UUID = "instruction"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_GOAL = CN()
_C.TASK.DISTANCE_TO_GOAL.TYPE = "DistanceToGoal"
_C.TASK.DISTANCE_TO_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # ANSWER_ACCURACY MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ANSWER_ACCURACY = CN()
_C.TASK.ANSWER_ACCURACY.TYPE = "AnswerAccuracy"
# -----------------------------------------------------------------------------
# # MULTION TASK
# -----------------------------------------------------------------------------
_C.TASK.CYL_OBJECTS_PATH = "data/multion_cyl_objects"
_C.TASK.REAL_OBJECTS_PATH = "data/multion_real_objects"
_C.TASK.OBJECTS_TYPE = "CYL" #"REAL" or "CYL"
_C.TASK.ORACLE_MAP_PATH = "data/hm3d_map300.pickle"
_C.TASK.NUM_GOALS = -1
_C.TASK.INCLUDE_DISTRACTORS = True
_C.TASK.ORACLE_MAP_INCLUDE_DISTRACTORS_W_GOAL = True
_C.TASK.NUM_DISTRACTORS = -1
_C.TASK.ACTIONS.FOUND = CN()
_C.TASK.ACTIONS.FOUND.TYPE = "FoundObjectAction"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_MULTI_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_MULTI_GOAL = CN()
_C.TASK.DISTANCE_TO_MULTI_GOAL.TYPE = "DistanceToMultiGoal"
_C.TASK.DISTANCE_TO_MULTI_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # EPISODE_LENGTH MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.EPISODE_LENGTH = CN()
_C.TASK.EPISODE_LENGTH.TYPE = "EpisodeLength"
# -----------------------------------------------------------------------------
# # RATIO MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.RATIO = CN()
_C.TASK.RATIO.TYPE = "Ratio"
_C.TASK.RATIO.DISTANCE_TO = "POINT"

_C.TASK.RAW_METRICS = CN()
_C.TASK.RAW_METRICS.TYPE = "RawMetrics"
# -----------------------------------------------------------------------------
# # DISTANCE_TO_CURRENT_GOAL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.DISTANCE_TO_CURR_GOAL = CN()
_C.TASK.DISTANCE_TO_CURR_GOAL.TYPE = "DistanceToCurrGoal"
_C.TASK.DISTANCE_TO_CURR_GOAL.DISTANCE_TO = "POINT"
# -----------------------------------------------------------------------------
# # SUB_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SUB_SUCCESS = CN()
_C.TASK.SUB_SUCCESS.TYPE = "SubSuccess"
_C.TASK.SUB_SUCCESS.SUCCESS_DISTANCE = 1.0
# -----------------------------------------------------------------------------
# # ORACLE_SUB_SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.ORACLE_SUB_SUCCESS = CN()
_C.TASK.ORACLE_SUB_SUCCESS.TYPE = "OracleSubSuccess"
_C.TASK.ORACLE_SUB_SUCCESS.SUCCESS_DISTANCE = 1.0
# -----------------------------------------------------------------------------
# # PROGRESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PROGRESS = CN()
_C.TASK.PROGRESS.TYPE = "Progress"
_C.TASK.PROGRESS.SUCCESS_DISTANCE = 1.0
# -----------------------------------------------------------------------------
# SUCCESS MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.SUCCESS = CN()
_C.TASK.SUCCESS.TYPE = "Success"
_C.TASK.SUCCESS.SUCCESS_DISTANCE = 0.2
# -----------------------------------------------------------------------------
# MSPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.MSPL = CN()
_C.TASK.MSPL.TYPE = "MSPL"
# -----------------------------------------------------------------------------
# PSPL MEASUREMENT
# -----------------------------------------------------------------------------
_C.TASK.PSPL = CN()
_C.TASK.PSPL.TYPE = "PSPL"

# -----------------------------------------------------------------------------
# SIMULATOR
# -----------------------------------------------------------------------------
_C.SIMULATOR = CN()
_C.SIMULATOR.TYPE = "Sim-v0"
_C.SIMULATOR.ACTION_SPACE_CONFIG = "v0"
_C.SIMULATOR.FORWARD_STEP_SIZE = 0.25  # in metres
_C.SIMULATOR.CREATE_RENDERER = False
_C.SIMULATOR.REQUIRES_TEXTURES = True
_C.SIMULATOR.LAG_OBSERVATIONS = 0
_C.SIMULATOR.AUTO_SLEEP = False
_C.SIMULATOR.STEP_PHYSICS = True
_C.SIMULATOR.UPDATE_ROBOT = True
_C.SIMULATOR.CONCUR_RENDER = False
_C.SIMULATOR.NEEDS_MARKERS = (
    True  # If markers should be updated at every step.
)
_C.SIMULATOR.UPDATE_ROBOT = (
    True  # If the robot camera positions should be updated at every step.
)
_C.SIMULATOR.SCENE = (
    "data/scene_datasets/habitat-test-scenes/van-gogh-room.glb"
)
_C.SIMULATOR.SCENE_DATASET = "default"  # the scene dataset to load in the MetaDataMediator. Should contain SIMULATOR.SCENE
_C.SIMULATOR.ADDITIONAL_OBJECT_PATHS = (
    []
)  # a list of directory or config paths to search in addition to the dataset for object configs. Should match the generated episodes for the task.
_C.SIMULATOR.SEED = _C.SEED
_C.SIMULATOR.TURN_ANGLE = 10  # angle to rotate left or right in degrees
_C.SIMULATOR.TILT_ANGLE = 15  # angle to tilt the camera up or down in degrees
_C.SIMULATOR.DEFAULT_AGENT_ID = 0
_C.SIMULATOR.DEBUG_RENDER = False
# If in render mode a visualization of the rearrangement goal position should
# also be displayed.
_C.SIMULATOR.DEBUG_RENDER_GOAL = True
_C.SIMULATOR.ROBOT_JOINT_START_NOISE = 0.0
# Rearrange Agent Setup
_C.SIMULATOR.ARM_REST = [0.6, 0.0, 0.9]
_C.SIMULATOR.CTRL_FREQ = 120.0
_C.SIMULATOR.AC_FREQ_RATIO = 4
_C.SIMULATOR.ROBOT_URDF = "data/robots/hab_fetch/robots/hab_fetch.urdf"
_C.SIMULATOR.ROBOT_TYPE = "FetchRobot"
_C.SIMULATOR.EE_LINK_NAME = None
_C.SIMULATOR.LOAD_OBJS = False
# Rearrange Agent Grasping
_C.SIMULATOR.HOLD_THRESH = 0.09
_C.SIMULATOR.GRASP_IMPULSE = 1000.0
# ROBOT
_C.SIMULATOR.IK_ARM_URDF = "data/robots/hab_fetch/robots/fetch_onlyarm.urdf"
# -----------------------------------------------------------------------------
# SIMULATOR SENSORS
# -----------------------------------------------------------------------------
SIMULATOR_SENSOR = CN()
SIMULATOR_SENSOR.HEIGHT = 480
SIMULATOR_SENSOR.WIDTH = 640
SIMULATOR_SENSOR.HFOV = 90  # horizontal field of view in degrees
SIMULATOR_SENSOR.POSITION = [0, 1.25, 0]
SIMULATOR_SENSOR.ORIENTATION = [0.0, 0.0, 0.0]  # Euler's angles

# -----------------------------------------------------------------------------
# CAMERA SENSOR
# -----------------------------------------------------------------------------
CAMERA_SIM_SENSOR = SIMULATOR_SENSOR.clone()
CAMERA_SIM_SENSOR.HFOV = 90  # horizontal field of view in degrees
CAMERA_SIM_SENSOR.SENSOR_SUBTYPE = "PINHOLE"

SIMULATOR_DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
SIMULATOR_DEPTH_SENSOR.MIN_DEPTH = 0.0
SIMULATOR_DEPTH_SENSOR.MAX_DEPTH = 10.0
SIMULATOR_DEPTH_SENSOR.NORMALIZE_DEPTH = True

# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.RGB_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.RGB_SENSOR.TYPE = "HabitatSimRGBSensor"
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.DEPTH_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.DEPTH_SENSOR.TYPE = "HabitatSimDepthSensor"
# -----------------------------------------------------------------------------
# SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.SEMANTIC_SENSOR = CAMERA_SIM_SENSOR.clone()
_C.SIMULATOR.SEMANTIC_SENSOR.TYPE = "HabitatSimSemanticSensor"
# -----------------------------------------------------------------------------
# EQUIRECT RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_RGB_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_RGB_SENSOR.TYPE = "HabitatSimEquirectangularRGBSensor"
# -----------------------------------------------------------------------------
# EQUIRECT DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.EQUIRECT_DEPTH_SENSOR.TYPE = (
    "HabitatSimEquirectangularDepthSensor"
)
# -----------------------------------------------------------------------------
# EQUIRECT SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.EQUIRECT_SEMANTIC_SENSOR = SIMULATOR_SENSOR.clone()
_C.SIMULATOR.EQUIRECT_SEMANTIC_SENSOR.TYPE = (
    "HabitatSimEquirectangularSemanticSensor"
)
# -----------------------------------------------------------------------------
# FISHEYE SENSOR
# -----------------------------------------------------------------------------
FISHEYE_SIM_SENSOR = SIMULATOR_SENSOR.clone()
FISHEYE_SIM_SENSOR.HEIGHT = FISHEYE_SIM_SENSOR.WIDTH
# -----------------------------------------------------------------------------
# ROBOT HEAD RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.HEAD_RGB_SENSOR.UUID = "robot_head_rgb"
# -----------------------------------------------------------------------------
# ROBOT HEAD DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.HEAD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.HEAD_DEPTH_SENSOR.UUID = "robot_head_depth"
# -----------------------------------------------------------------------------
# ARM RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.ARM_RGB_SENSOR.UUID = "robot_arm_rgb"
# -----------------------------------------------------------------------------
# ARM DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.ARM_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.ARM_DEPTH_SENSOR.UUID = "robot_arm_depth"
# -----------------------------------------------------------------------------
# 3rd RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_RGB_SENSOR = _C.SIMULATOR.RGB_SENSOR.clone()
_C.SIMULATOR.THIRD_RGB_SENSOR.UUID = "robot_third_rgb"
# -----------------------------------------------------------------------------
# 3rd DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.THIRD_DEPTH_SENSOR = _C.SIMULATOR.DEPTH_SENSOR.clone()
_C.SIMULATOR.THIRD_DEPTH_SENSOR.UUID = "robot_third_rgb"

# The default value (alpha, xi) is set to match the lens "GoPro" found in Table 3 of this paper:
# Vladyslav Usenko, Nikolaus Demmel and Daniel Cremers: The Double Sphere
# Camera Model, The International Conference on 3D Vision (3DV), 2018
# You can find the intrinsic parameters for the other lenses in the same table as well.
FISHEYE_SIM_SENSOR.XI = -0.27
FISHEYE_SIM_SENSOR.ALPHA = 0.57
FISHEYE_SIM_SENSOR.FOCAL_LENGTH = [364.84, 364.86]
# Place camera at center of screen
# Can be specified, otherwise is calculated automatically.
FISHEYE_SIM_SENSOR.PRINCIPAL_POINT_OFFSET = None  # (defaults to (h/2,w/2))
FISHEYE_SIM_SENSOR.SENSOR_MODEL_TYPE = "DOUBLE_SPHERE"
# -----------------------------------------------------------------------------
# FISHEYE RGB SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_RGB_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_RGB_SENSOR.TYPE = "HabitatSimFisheyeRGBSensor"
# -----------------------------------------------------------------------------
# FISHEYE DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR.merge_from_other_cfg(SIMULATOR_DEPTH_SENSOR)
_C.SIMULATOR.FISHEYE_DEPTH_SENSOR.TYPE = "HabitatSimFisheyeDepthSensor"
# -----------------------------------------------------------------------------
# FISHEYE SEMANTIC SENSOR
# -----------------------------------------------------------------------------
_C.SIMULATOR.FISHEYE_SEMANTIC_SENSOR = FISHEYE_SIM_SENSOR.clone()
_C.SIMULATOR.FISHEYE_SEMANTIC_SENSOR.TYPE = "HabitatSimFisheyeSemanticSensor"
# -----------------------------------------------------------------------------
# AGENT
# -----------------------------------------------------------------------------
_C.SIMULATOR.AGENT_0 = CN()
_C.SIMULATOR.AGENT_0.HEIGHT = 1.5
_C.SIMULATOR.AGENT_0.RADIUS = 0.1
_C.SIMULATOR.AGENT_0.SENSORS = ["RGB_SENSOR"]
_C.SIMULATOR.AGENT_0.IS_SET_START_STATE = False
_C.SIMULATOR.AGENT_0.START_POSITION = [0, 0, 0]
_C.SIMULATOR.AGENT_0.START_ROTATION = [0, 0, 0, 1]
_C.SIMULATOR.AGENTS = ["AGENT_0"]
# -----------------------------------------------------------------------------
# SIMULATOR HABITAT_SIM_V0
# -----------------------------------------------------------------------------
_C.SIMULATOR.HABITAT_SIM_V0 = CN()
_C.SIMULATOR.HABITAT_SIM_V0.GPU_DEVICE_ID = 0
# Use Habitat-Sim's GPU->GPU copy mode to return rendering results
# in PyTorch tensors.  Requires Habitat-Sim to be built
# with --with-cuda
# This will generally imply sharing CUDA tensors between processes.
# Read here: https://pytorch.org/docs/stable/multiprocessing.html#sharing-cuda-tensors
# for the caveats that results in
_C.SIMULATOR.HABITAT_SIM_V0.GPU_GPU = False
# Whether or not the agent slides on collisions
_C.SIMULATOR.HABITAT_SIM_V0.ALLOW_SLIDING = True
_C.SIMULATOR.HABITAT_SIM_V0.FRUSTUM_CULLING = True
_C.SIMULATOR.HABITAT_SIM_V0.ENABLE_PHYSICS = False
_C.SIMULATOR.HABITAT_SIM_V0.PHYSICS_CONFIG_FILE = (
    "./data/default.physics_config.json"
)
# Possibly unstable optimization for extra performance with concurrent rendering
_C.SIMULATOR.HABITAT_SIM_V0.LEAVE_CONTEXT_WITH_BACKGROUND_RENDERER = False
# -----------------------------------------------------------------------------
# PYROBOT
# -----------------------------------------------------------------------------
_C.PYROBOT = CN()
_C.PYROBOT.ROBOTS = ["locobot"]  # types of robots supported
_C.PYROBOT.ROBOT = "locobot"
_C.PYROBOT.SENSORS = ["RGB_SENSOR", "DEPTH_SENSOR", "BUMP_SENSOR"]
_C.PYROBOT.BASE_CONTROLLER = "proportional"
_C.PYROBOT.BASE_PLANNER = "none"
# -----------------------------------------------------------------------------
# SENSORS
# -----------------------------------------------------------------------------
PYROBOT_VISUAL_SENSOR = CN()
PYROBOT_VISUAL_SENSOR.HEIGHT = 480
PYROBOT_VISUAL_SENSOR.WIDTH = 640
# -----------------------------------------------------------------------------
# RGB SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.RGB_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.RGB_SENSOR.TYPE = "PyRobotRGBSensor"
_C.PYROBOT.RGB_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# DEPTH SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.DEPTH_SENSOR = PYROBOT_VISUAL_SENSOR.clone()
_C.PYROBOT.DEPTH_SENSOR.TYPE = "PyRobotDepthSensor"
_C.PYROBOT.DEPTH_SENSOR.MIN_DEPTH = 0.0
_C.PYROBOT.DEPTH_SENSOR.MAX_DEPTH = 5.0
_C.PYROBOT.DEPTH_SENSOR.NORMALIZE_DEPTH = True
_C.PYROBOT.DEPTH_SENSOR.CENTER_CROP = False
# -----------------------------------------------------------------------------
# BUMP SENSOR
# -----------------------------------------------------------------------------
_C.PYROBOT.BUMP_SENSOR = CN()
_C.PYROBOT.BUMP_SENSOR.TYPE = "PyRobotBumpSensor"
# -----------------------------------------------------------------------------
# ACTIONS LOCOBOT
# -----------------------------------------------------------------------------
_C.PYROBOT.LOCOBOT = CN()
_C.PYROBOT.LOCOBOT.ACTIONS = ["BASE_ACTIONS", "CAMERA_ACTIONS"]
_C.PYROBOT.LOCOBOT.BASE_ACTIONS = ["go_to_relative", "go_to_absolute"]
_C.PYROBOT.LOCOBOT.CAMERA_ACTIONS = ["set_pan", "set_tilt", "set_pan_tilt"]
# TODO(akadian): add support for Arm actions
# -----------------------------------------------------------------------------
# DATASET
# -----------------------------------------------------------------------------
_C.DATASET = CN()
_C.DATASET.TYPE = "MultiObjectNav-v1"
_C.DATASET.SPLIT = "train"
_C.DATASET.SCENES_DIR = "data/scene_datasets"
_C.DATASET.CONTENT_SCENES = ["*"]
_C.DATASET.DATA_PATH = (
    "data/5_ON_CYL/minival/minival.json.gz" #"data/5_ON_REAL/minival/minival.json.gz"
)
_C.DATASET.NUM_GOALS = 3

# -----------------------------------------------------------------------------
# GYM
# -----------------------------------------------------------------------------
_C.GYM = CN()
_C.GYM.AUTO_NAME = ""
_C.GYM.CLASS_NAME = "RearrangeRLEnv"
_C.GYM.OBS_KEYS = None
_C.GYM.ACTION_KEYS = None
_C.GYM.ACHIEVED_GOAL_KEYS = []
_C.GYM.DESIRED_GOAL_KEYS = []
_C.GYM.FIX_INFO_DICT = True

# -----------------------------------------------------------------------------


def get_extended_config(
    config_paths: Optional[Union[List[str], str]] = None,
    opts: Optional[list] = None,
) -> CN:
    r"""Create a unified config with default values overwritten by values from
    :p:`config_paths` and overwritten by options from :p:`opts`.

    :param config_paths: List of config paths or string that contains comma
        separated list of config paths.
    :param opts: Config options (keys, values) in a list (e.g., passed from
        command line into the config. For example,
        :py:`opts = ['FOO.BAR', 0.5]`. Argument can be used for parameter
        sweeping or quick tests.
    """
    config = _C.clone()
    if config_paths:
        if isinstance(config_paths, str):
            if CONFIG_FILE_SEPARATOR in config_paths:
                config_paths = config_paths.split(CONFIG_FILE_SEPARATOR)
            else:
                config_paths = [config_paths]

        for config_path in config_paths:
            config.merge_from_file(config_path)

    if opts:
        config.merge_from_list(opts)

    config.freeze()
    return config
