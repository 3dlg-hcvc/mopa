from habitat.core.registry import registry
from habitat.datasets.pointnav.pointnav_dataset import PointNavDatasetV1

from multion import actions, measures, sensors
from multion.config.default import get_extended_config

try:
    from multion.task import MultiObjectNavDatasetV1
except ImportError as e:
    @registry.register_dataset(name="MultiObjectNav-v1")
    class MultiObjectNavDatasetV1(PointNavDatasetV1):
        def __init__(self, *args, **kwargs):
            raise e

