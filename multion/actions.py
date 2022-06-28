from typing import Any

from habitat.core.registry import registry
from habitat.core.embodied_task import (
    EmbodiedTask,
    SimulatorTaskAction,
)

@registry.register_task_action
class FoundObjectAction(SimulatorTaskAction):
    name: str = "FOUND"
    def reset(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        task.is_stop_called = False
        task.is_found_called = False ##C

    def step(self, *args: Any, task: EmbodiedTask, **kwargs: Any):
        r"""Update ``_metric``, this method is called from ``Env`` on each
        ``step``.
        """
        task.is_found_called = True
        return self._sim.get_observations_at()