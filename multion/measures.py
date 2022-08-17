
from typing import Any

import numpy as np
from habitat.config import Config
from habitat.core.embodied_task import EmbodiedTask, Measure
from habitat.core.logging import logger
from habitat.core.registry import registry
from habitat.core.simulator import Simulator, AgentState
from habitat.tasks.nav.nav import NavigationEpisode
from multion import maps as multion_maps
from habitat.utils.visualizations import maps, fog_of_war
from habitat.utils.geometry_utils import (
    quaternion_rotate_vector,
)
from habitat.tasks.utils import cartesian_to_polar
from habitat.core.utils import try_cv2_import

cv2 = try_cv2_import()

@registry.register_measure
class Success(Measure):
    r"""Whether or not the agent succeeded at its task.
    """

    cls_uuid: str = "success"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        task.measurements.check_measure_dependencies(
            self.uuid, [SubSuccess.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        subsuccess = task.measurements.measures[
            SubSuccess.cls_uuid
        ].get_metric()

        if subsuccess ==1 and task.current_goal_index >= len(episode.goals):
            self._metric = 1
        else:
            self._metric = 0

@registry.register_measure
class SubSuccess(Measure):
    r"""Whether or not the agent succeeded in finding it's
    current goal. This measure depends on DistanceToCurrGoal measure.
    """

    cls_uuid: str = "sub_success"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any): ##Called only when episode begins
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToCurrGoal.cls_uuid]
        )
        task.current_goal_index=0  
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_subgoal = task.measurements.measures[
            DistanceToCurrGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_found_called")
            and task.is_found_called
            and distance_to_subgoal <= self._config.SUCCESS_DISTANCE
        ):
            self._metric = 1
            task.current_goal_index+=1
            task.foundDistance = distance_to_subgoal
        else:
            self._metric = 0

@registry.register_measure
class Progress(Measure):
    r"""Variant of SubSuccess. It tells how much of the episode 
        is successful
    """

    cls_uuid: str = "progress"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any): ##Called only when episode begins
        task.measurements.check_measure_dependencies(
            self.uuid, [DistanceToCurrGoal.cls_uuid]
        )
        self._metric=0
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(
        self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any
    ):
        distance_to_subgoal = task.measurements.measures[
            DistanceToCurrGoal.cls_uuid
        ].get_metric()

        if (
            hasattr(task, "is_found_called")
            and task.is_found_called
            and distance_to_subgoal < self._config.SUCCESS_DISTANCE
        ):
            self._metric += 1/len(episode.goals)


@registry.register_measure
class MSPL(Measure):
    """SPL, but in multigoal case
    """
    cls_uuid: str = "mspl"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                )
        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [Success.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        self._metric = ep_success * (
            self._start_end_episode_distance
            / max(
                self._start_end_episode_distance, self._agent_episode_distance
            )
        )

@registry.register_measure
class PSPL(Measure):
    """SPL, but in multigoal case
    """
    cls_uuid: str = "pspl"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0
        self._start_subgoal_episode_distance = []
        self._start_subgoal_agent_distance = []
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)
        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [SubSuccess.cls_uuid, Progress.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        ep_percentage_success = task.measurements.measures[Progress.cls_uuid].get_metric()
        ep_sub_success = task.measurements.measures[SubSuccess.cls_uuid].get_metric()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position

        if ep_sub_success:
            self._start_subgoal_agent_distance.append(self._agent_episode_distance)

        if ep_percentage_success > 0:
            self._metric = ep_percentage_success * (
                self._start_subgoal_episode_distance[task.current_goal_index - 1]
                / max(
                    self._start_subgoal_agent_distance[-1], self._start_subgoal_episode_distance[task.current_goal_index - 1]
                )
            )
        else:
            self._metric = 0


@registry.register_measure
class DistanceToCurrGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_currgoal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()
        self._start_end_subgoal_distance = self._sim.geodesic_distance(
            self._previous_position, episode.goals[task.current_goal_index].position
        )
        self._agent_subgoal_distance = 0.0
        self._metric = None
        if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._subgoal_view_points = [
                view_point.agent_state.position
                for goal in episode.goals[task.current_goal_index]
                for view_point in goal.view_points
            ]
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        if self._config.DISTANCE_TO == "POINT":
            distance_to_subgoal= self._sim.geodesic_distance(
                current_position, episode.goals[task.current_goal_index].position
            )
        elif self._config.DISTANCE_TO == "VIEW_POINTS":
            distance_to_subgoal = self._sim.geodesic_distance(
                current_position, self._subgoal_view_points
            )

        else:
            logger.error(
                f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
            )

        self._agent_subgoal_distance += self._euclidean_distance(
            current_position, self._previous_position
        )

        self._previous_position = current_position
        
        if (distance_to_subgoal==float("inf")):
            # The simulator found no path from the current position to the goal.
            # Assign a large number to distance_to_subgoal.
            distance_to_subgoal = float(10000)
            logger.info('Inf value from sim.geodesic_distance. Current Position, Goal=')
            logger.info(str(current_position))
            logger.info(str(episode.goals[task.current_goal_index].position))

        self._metric = distance_to_subgoal



@registry.register_measure
class DistanceToMultiGoal(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "distance_to_multi_goal"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        """if self._config.DISTANCE_TO == "VIEW_POINTS":
            self._episode_view_points = [
                view_point.agent_state.position
                # for goal in episode.goals     # Considering only one goal for now
                for view_point in episode.goals[episode.object_index][0].view_points
            ]"""
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, episode, task, *args: Any, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()

        if self._config.DISTANCE_TO == "POINT":
            distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[task.current_goal_index].position
            )
            for goal_number in range(task.current_goal_index, len(episode.goals)-1):
                distance_to_target += self._sim.geodesic_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )
        else:
            logger.error(
                f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
            )

        self._metric = distance_to_target


@registry.register_measure
class EpisodeLength(Measure):
    r"""Calculates the episode length
    """
    cls_uuid: str = "episode_length"

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._episode_length = None
        self._sim = sim
        self._config = config

        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._episode_length = 0
        self._metric = self._episode_length

    def update_metric(
        self,
        *args: Any,
        episode,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        self._episode_length += 1
        self._metric = self._episode_length


@registry.register_measure
class Ratio(Measure):
    """The measure calculates a distance towards the goal.
    """

    cls_uuid: str = "ratio"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config

        super().__init__(**kwargs)

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, episode, task, *args: Any, **kwargs: Any):
        self._metric = None
        current_position = self._sim.get_agent_state().position.tolist()
        if self._config.DISTANCE_TO == "POINT":
            initial_geodesic_distance_to_target = self._sim.geodesic_distance(
                current_position, episode.goals[0].position
            )
            for goal_number in range(0, len(episode.goals)-1):
                initial_geodesic_distance_to_target += self._sim.geodesic_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )

            initial_euclidean_distance_to_target = self._euclidean_distance(
                current_position, episode.goals[0].position
            )
            for goal_number in range(0, len(episode.goals)-1):
                initial_euclidean_distance_to_target += self._euclidean_distance(
                    episode.goals[goal_number].position, episode.goals[goal_number+1].position
                )
        # else:
        #     logger.error(
        #         f"Non valid DISTANCE_TO parameter was provided: {self._config.DISTANCE_TO}"
        #     )
        self._metric = initial_geodesic_distance_to_target / initial_euclidean_distance_to_target

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(
        self,
        *args: Any,
        episode,
        task: EmbodiedTask,
        **kwargs: Any,
    ):
        pass

@registry.register_measure
class RawMetrics(Measure):
    """All the raw metrics we might need
    """
    cls_uuid: str = "raw_metrics"

    def __init__(
        self, sim: Simulator, config: Config, *args: Any, **kwargs: Any
    ):
        self._previous_position = None
        self._start_end_episode_distance = None
        self._agent_episode_distance = None
        self._sim = sim
        self._config = config
        self._episode_view_points = None
        super().__init__(**kwargs)


    def _get_uuid(self, *args: Any, **kwargs: Any):
        return self.cls_uuid

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._previous_position = self._sim.get_agent_state().position.tolist()

        self._start_end_episode_distance = 0
        self._start_subgoal_episode_distance = []
        self._start_subgoal_agent_distance = []
        for goal_number in range(len(episode.goals) ):  # Find distances between successive goals and keep adding them
            if goal_number == 0:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.start_position, episode.goals[0].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)
            else:
                self._start_end_episode_distance += self._sim.geodesic_distance(
                    episode.goals[goal_number - 1].position, episode.goals[goal_number].position
                )
                self._start_subgoal_episode_distance.append(self._start_end_episode_distance)

        self._agent_episode_distance = 0.0
        self._metric = None
        task.measurements.check_measure_dependencies(
            self.uuid, [EpisodeLength.cls_uuid, MSPL.cls_uuid, PSPL.cls_uuid, DistanceToMultiGoal.cls_uuid, DistanceToCurrGoal.cls_uuid, SubSuccess.cls_uuid, Success.cls_uuid, Progress.cls_uuid]
        )
        self.update_metric(*args, episode=episode, task=task, **kwargs)
        ##

    def _euclidean_distance(self, position_a, position_b):
        return np.linalg.norm(
            np.array(position_b) - np.array(position_a), ord=2
        )

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        current_position = self._sim.get_agent_state().position.tolist()
        ep_success = task.measurements.measures[Success.cls_uuid].get_metric()
        p_success = task.measurements.measures[Progress.cls_uuid].get_metric()
        distance_to_curr_subgoal = task.measurements.measures[DistanceToCurrGoal.cls_uuid].get_metric()
        ep_sub_success = task.measurements.measures[SubSuccess.cls_uuid].get_metric()

        self._agent_episode_distance += self._euclidean_distance(
            current_position, self._previous_position
        )
        self._previous_position = current_position
        if ep_sub_success:
            self._start_subgoal_agent_distance.append(self._agent_episode_distance)

        self._metric = {
            'success': ep_success,
            'percentage_success': p_success,
            'geodesic_distances': self._start_subgoal_episode_distance,
            'agent_path_length': self._agent_episode_distance,
            'subgoals_found': task.current_goal_index,
            'distance_to_curr_subgoal': distance_to_curr_subgoal,
            'agent_distances': self._start_subgoal_agent_distance,
            'distance_to_multi_goal': task.measurements.measures[DistanceToMultiGoal.cls_uuid].get_metric(),
            'MSPL': task.measurements.measures[MSPL.cls_uuid].get_metric(),
            'PSPL': task.measurements.measures[PSPL.cls_uuid].get_metric(),
            'episode_lenth': task.measurements.measures[EpisodeLength.cls_uuid].get_metric()
        }

@registry.register_measure
class TopDownMap(Measure):
    r"""Top Down Map measure"""

    def __init__(
        self, sim: "HabitatSim", config: Config, *args: Any, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._grid_delta = config.MAP_PADDING
        self._step_count: Optional[int] = None
        self._map_resolution = config.MAP_RESOLUTION
        self._ind_x_min: Optional[int] = None
        self._ind_x_max: Optional[int] = None
        self._ind_y_min: Optional[int] = None
        self._ind_y_max: Optional[int] = None
        self._previous_xy_location: Optional[Tuple[int, int]] = None
        self._top_down_map: Optional[np.ndarray] = None
        self._shortest_path_points: Optional[List[Tuple[int, int]]] = None
        """ self.line_thickness = int(
            np.round(self._map_resolution * 2 / MAP_THICKNESS_SCALAR)
        ) """
        self.line_thickness = 8
        """ self.point_padding = 2 * int(
            np.ceil(self._map_resolution / MAP_THICKNESS_SCALAR)
        ) """
        self.point_padding = 25
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any) -> str:
        return "top_down_map"

    def get_original_map(self):
        top_down_map = multion_maps.get_topdown_map_from_sim(
            self._sim,
            map_resolution=self._map_resolution,
            draw_border=self._config.DRAW_BORDER,
            with_sampling=False
        )

        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = np.zeros_like(top_down_map)
        else:
            self._fog_of_war_mask = None

        return top_down_map

    def _draw_point(self, position, point_type):
        t_x, t_y = maps.to_grid(
            position[2],
            position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._top_down_map[
            t_x - self.point_padding : t_x + self.point_padding + 1,
            t_y - self.point_padding : t_y + self.point_padding + 1,
        ] = point_type

    def _draw_goals_view_points(self, episode):
        if self._config.DRAW_VIEW_POINTS:
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        if goal.view_points is not None:
                            for view_point in goal.view_points:
                                self._draw_point(
                                    view_point.agent_state.position,
                                    maps.MAP_VIEW_POINT_INDICATOR,
                                )
                    except AttributeError:
                        pass

    def _draw_goals_positions(self, episode):
        if self._config.DRAW_GOAL_POSITIONS:

            ind = 10
            for goal in episode.goals:
                if self._is_on_same_floor(goal.position[1]):
                    try:
                        self._draw_point(
                            goal.position, ind
                        )
                        ind+=1
                    except AttributeError:
                        pass

    def _draw_goals_and_distractors(self, episode):
        for goal in episode.goals:
            if self._is_on_same_floor(goal.position[1]):
                try:
                    color_ind = multion_maps.MULTION_CYL_OBJECT_CATEGORY[goal.object_category]
                    self._draw_point(
                        goal.position, (multion_maps.MULTION_TOP_DOWN_MAP_START + color_ind)
                    )
                except AttributeError:
                    pass
        for distractor in episode.distractors:
            if self._is_on_same_floor(distractor.position[1]):
                try:
                    color_ind = multion_maps.MULTION_CYL_OBJECT_CATEGORY[distractor.object_category]
                    self._draw_point(
                        distractor.position, (multion_maps.MULTION_TOP_DOWN_MAP_START + color_ind)
                    )
                except AttributeError:
                    pass

    def _draw_goals_aabb(self, episode):
        if self._config.DRAW_GOAL_AABBS:
            for goal in episode.goals:
                try:
                    sem_scene = self._sim.semantic_annotations()
                    object_id = goal.object_id
                    assert int(
                        sem_scene.objects[object_id].id.split("_")[-1]
                    ) == int(
                        goal.object_id
                    ), f"Object_id doesn't correspond to id in semantic scene objects dictionary for episode: {episode}"

                    center = sem_scene.objects[object_id].aabb.center
                    x_len, _, z_len = (
                        sem_scene.objects[object_id].aabb.sizes / 2.0
                    )
                    # Nodes to draw rectangle
                    corners = [
                        center + np.array([x, 0, z])
                        for x, z in [
                            (-x_len, -z_len),
                            (-x_len, z_len),
                            (x_len, z_len),
                            (x_len, -z_len),
                            (-x_len, -z_len),
                        ]
                        if self._is_on_same_floor(center[1])
                    ]

                    map_corners = [
                        maps.to_grid(
                            p[2],
                            p[0],
                            self._top_down_map.shape[0:2],
                            sim=self._sim,
                        )
                        for p in corners
                    ]

                    maps.draw_path(
                        self._top_down_map,
                        map_corners,
                        maps.MAP_TARGET_BOUNDING_BOX,
                        self.line_thickness,
                    )
                except AttributeError:
                    pass

    def _draw_shortest_path(
        self, episode: NavigationEpisode, agent_position: AgentState
    ):
        if self._config.DRAW_SHORTEST_PATH:
            _shortest_path_points = (
                self._sim.get_straight_shortest_path_points(
                    agent_position, episode.goals[0].position
                )
            )
            self._shortest_path_points = [
                maps.to_grid(
                    p[2], p[0], self._top_down_map.shape[0:2], sim=self._sim
                )
                for p in _shortest_path_points
            ]
            maps.draw_path(
                self._top_down_map,
                self._shortest_path_points,
                maps.MAP_SHORTEST_PATH_COLOR,
                self.line_thickness,
            )

    def _is_on_same_floor(
        self, height, ref_floor_height=None, ceiling_height=2.0
    ):
        if ref_floor_height is None:
            ref_floor_height = self._sim.get_agent(0).state.position[1]
        return ref_floor_height <= height < ref_floor_height + ceiling_height

    def reset_metric(self, episode, *args: Any, **kwargs: Any):
        self._step_count = 0
        self._metric = None
        self._top_down_map = self.get_original_map()
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        self._previous_xy_location = (a_y, a_x)

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        # draw source and target parts last to avoid overlap
        #self._draw_goals_view_points(episode)
        #self._draw_goals_aabb(episode)
        #self._draw_goals_positions(episode)
        self._draw_goals_and_distractors(episode)

        #self._draw_shortest_path(episode, agent_position)

        if self._config.DRAW_SOURCE:
            self._draw_point(
                episode.start_position, maps.MAP_SOURCE_POINT_INDICATOR
            )

    def update_metric(self, episode, action, *args: Any, **kwargs: Any):
        self._step_count += 1
        house_map, map_agent_x, map_agent_y = self.update_map(
            self._sim.get_agent_state().position
        )

        self._metric = {
            "map": house_map,
            "fog_of_war_mask": self._fog_of_war_mask,
            "agent_map_coord": (map_agent_x, map_agent_y),
            "agent_angle": self.get_polar_angle(),
        }

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

    def update_map(self, agent_position):
        a_x, a_y = maps.to_grid(
            agent_position[2],
            agent_position[0],
            self._top_down_map.shape[0:2],
            sim=self._sim,
        )
        # Don't draw over the source point
        if self._top_down_map[a_x, a_y] != maps.MAP_SOURCE_POINT_INDICATOR:
            color = 10 + min(
                self._step_count * 245 // self._config.MAX_EPISODE_STEPS, 245
            )

            thickness = self.line_thickness
            cv2.line(
                self._top_down_map,
                self._previous_xy_location,
                (a_y, a_x),
                color,
                thickness=thickness,
            )

        self.update_fog_of_war_mask(np.array([a_x, a_y]))

        self._previous_xy_location = (a_y, a_x)
        return self._top_down_map, a_x, a_y

    def update_fog_of_war_mask(self, agent_position):
        if self._config.FOG_OF_WAR.DRAW:
            self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
                self._top_down_map,
                self._fog_of_war_mask,
                agent_position,
                self.get_polar_angle(),
                fov=self._config.FOG_OF_WAR.FOV,
                max_line_len=self._config.FOG_OF_WAR.VISIBILITY_DIST
                / maps.calculate_meters_per_pixel(
                    self._map_resolution, sim=self._sim
                ),
            )


@registry.register_measure
class FowMap(Measure):
    r"""FOW map measure
    """

    def __init__(
        self, *args: Any, sim: Simulator, config: Config, **kwargs: Any
    ):
        self._sim = sim
        self._config = config
        self._map_resolution = (300, 300)
        self._coordinate_min = -62.3241-1e-6
        self._coordinate_max = 90.0399+1e-6
        super().__init__()

    def _get_uuid(self, *args: Any, **kwargs: Any):
        return "fow_map"

    def conv_grid(
        self,
        realworld_x,
        realworld_y
    ):

        grid_size = (
            (self._coordinate_max - self._coordinate_min) / self._map_resolution[0],
            (self._coordinate_max - self._coordinate_min) / self._map_resolution[1],
        )
        grid_x = int((self._coordinate_max - realworld_x) / grid_size[0])
        grid_y = int((realworld_y - self._coordinate_min) / grid_size[1])
        return grid_x, grid_y

    def reset_metric(self, *args: Any, episode, task, **kwargs: Any):
        self._metric = None
        self._top_down_map = task.sceneMap
        self._fog_of_war_mask = np.zeros_like(self._top_down_map)
        self.update_metric(*args, episode=episode, task=task, **kwargs)

    def update_metric(self, *args: Any, episode, task: EmbodiedTask, **kwargs: Any):
        agent_position = self._sim.get_agent_state().position
        a_x, a_y = self.conv_grid(
            agent_position[2],
            agent_position[0]
        )
        agent_position = np.array([a_x, a_y])

        self._fog_of_war_mask = fog_of_war.reveal_fog_of_war(
            self._top_down_map,
            self._fog_of_war_mask,
            agent_position,
            self.get_polar_angle(),
            fov=self._config.FOV,
            max_line_len=self._config.VISIBILITY_DIST
            * max(self._map_resolution)
            / (self._coordinate_max - self._coordinate_min),
        )

        self._metric = self._fog_of_war_mask


    def get_polar_angle(self):
        agent_state = self._sim.get_agent_state()
        # quaternion is in x, y, z, w format
        ref_rotation = agent_state.rotation
        heading_vector = quaternion_rotate_vector(
            ref_rotation.inverse(), np.array([0, 0, -1])
        )
        phi = cartesian_to_polar(-heading_vector[2], heading_vector[0])[1]
        x_y_flip = -np.pi / 2
        return np.array(phi) + x_y_flip

