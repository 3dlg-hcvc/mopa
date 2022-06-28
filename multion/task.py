import gzip
import json
import os
from typing import Dict, List, Optional, Any
import numpy as np
import attr

from habitat.config import Config
from habitat.core.dataset import Dataset
from habitat.datasets.pointnav.pointnav_dataset import (
    CONTENT_SCENES_PATH_FIELD,
    DEFAULT_SCENE_PATH_PREFIX,
    PointNavDatasetV1,
)
from habitat.core.registry import registry
from habitat.core.simulator import Simulator
from habitat.tasks.nav.nav import (
    NavigationEpisode,
    NavigationGoal,
    NavigationTask,
)

@attr.s(auto_attribs=True, kw_only=True)
class MultiObjectGoal(NavigationGoal):
    r"""Object goal provides information about an object that is target for
    navigation. That can be specify object_id, position and object
    category. An important part for metrics calculation are view points that
     describe success area for the navigation.
    Args:
        object_id: id that can be used to retrieve object from the semantic
        scene annotation
        object_name: name of the object
        object_category: object category name usually similar to scene semantic
        categories
        room_id: id of a room where object is located, can be used to retrieve
        room from the semantic scene annotation
        room_name: name of the room, where object is located
        view_points: navigable positions around the object with specified
        proximity of the object surface used for navigation metrics calculation.
        The object is visible from these positions.
    """

    object_category: Optional[str] = None
    room_id: Optional[str] = None
    room_name: Optional[str] = None
    position: Optional[List[List[float]]]

@attr.s(auto_attribs=True, kw_only=True)
class MultiObjectGoalNavEpisode(NavigationEpisode):
    r"""Multi ObjectGoal Navigation Episode

    :param object_category: Category of the obect
    """
    object_category: Optional[List[str]] = None
    object_index: Optional[int]
    current_goal_index: Optional[int] = 0
    distractors: List[Any] = []  

    @property
    def goals_key(self) -> str:
        r"""The key to retrieve the goals
        """
        return [f"{os.path.basename(self.scene_id)}_{i}" for i in self.object_category]


@registry.register_dataset(name="MultiObjectNav-v1")
class MultiObjectNavDatasetV1(PointNavDatasetV1):
    r"""Class inherited from PointNavDataset that loads MultiON dataset."""
    category_to_task_category_id: Dict[str, int]
    episodes: List[Any] = []
    content_scenes_path: str = "{data_path}/content/{scene}.json.gz"

    def __init__(self, config: Optional[Config] = None) -> None:
        super().__init__(config)

    def from_json(
        self, json_str: str, scenes_dir: Optional[str] = None
    ) -> None:
        deserialized = json.loads(json_str)
        if CONTENT_SCENES_PATH_FIELD in deserialized:
            self.content_scenes_path = deserialized[CONTENT_SCENES_PATH_FIELD]

        if "category_to_task_category_id" in deserialized:
            self.category_to_task_category_id = deserialized[
                "category_to_task_category_id"
            ]

        if len(deserialized["episodes"]) == 0:
            return

        for i, episode in enumerate(deserialized["episodes"]):
            episode['object_index'] = 0 ##Shivansh why does this exist
            episode["current_goal_index"] = 0
            episode = MultiObjectGoalNavEpisode(**episode)
            episode.episode_id = str(i)

            if scenes_dir is not None:
                if episode.scene_id.startswith(DEFAULT_SCENE_PATH_PREFIX):
                    episode.scene_id = episode.scene_id[
                        len(DEFAULT_SCENE_PATH_PREFIX) :
                    ]

                episode.scene_id = os.path.join(scenes_dir, episode.scene_id)

            episode.goals = [MultiObjectGoal(**i) for i in episode.goals]
            episode.distractors = [MultiObjectGoal(**i) for i in episode.distractors]

            self.episodes.append(episode)

@registry.register_task(name="MultiObjectNav-v1")
class MultiObjectNavigationTask(NavigationTask):
    r"""An Object Navigation Task class for a task specific methods.
        Used to explicitly state a type of the task in config.
    """
    def __init__(
        self, config: Config, sim: Simulator, dataset: Optional[Dataset] = None
    ) -> None:
        super().__init__(config=config, sim=sim, dataset=dataset)
        self.current_goal_index=0

    def reset(self, episode: MultiObjectGoalNavEpisode):
        rigid_obj_mgr = self._sim.get_rigid_object_manager()
        
        # Remove existing objects from last episode
        rigid_obj_mgr.remove_all_objects()

        # Insert current episode objects
        obj_type = self._config.OBJECTS_TYPE
        if obj_type == "CYL":
            obj_path = self._config.CYL_OBJECTS_PATH
            self.object_to_datset_mapping = {'cylinder_red':0, 'cylinder_green':1, 'cylinder_blue':2, 
                                             'cylinder_yellow':3, 'cylinder_white':4, 'cylinder_pink':5, 
                                             'cylinder_black':6, 'cylinder_cyan':7}
        else:
            obj_path = self._config.REAL_OBJECTS_PATH
            self.object_to_datset_mapping = {'guitar':0, 'electric_piano':1, 'basket_ball':2,'toy_train':3, 
                                             'teddy_bear':4, 'rocking_horse':5, 'backpack': 6, 'trolley_bag':7}
            
        obj_templates_mgr = self._sim.get_object_template_manager()
        obj_templates_mgr.load_configs(obj_path, True)
            
        for i in range(len(episode.goals)):
            current_goal = episode.goals[i].object_category
            dataset_index = self.object_to_datset_mapping[current_goal]
            
            obj_handle_list = obj_templates_mgr.get_template_handles(current_goal)[0]
            object_box = rigid_obj_mgr.add_object_by_template_handle(obj_handle_list)
            
            object_box.semantic_id = dataset_index
            object_box.translation = np.array(episode.goals[i].position)
            
        if self._config.INCLUDE_DISTRACTORS:
            for i in range(len(episode.distractors)):
                current_distractor = episode.distractors[i].object_category
                dataset_index = self.object_to_datset_mapping[current_distractor]
            
                obj_handle_list = obj_templates_mgr.get_template_handles(current_distractor)[0]
                object_box = rigid_obj_mgr.add_object_by_template_handle(obj_handle_list)
                
                object_box.semantic_id = dataset_index
                object_box.translation = np.array(episode.distractors[i].position)

        # Reinitialize current goal index
        self.current_goal_index = 0

        # Initialize self.is_found_called
        self.is_found_called = False

        observations = self._sim.reset()
        observations.update(
            self.sensor_suite.get_observations(
                observations=observations, episode=episode, task=self
            )
        )

        for action_instance in self.actions.values():
            action_instance.reset(episode=episode, task=self)

        return observations

        