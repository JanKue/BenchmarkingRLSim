from abc import ABC
import os

from alr_sim.core.sim_object.sim_object import SimObject
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable


class SoccerGoal(SimObject, MujocoXmlLoadable, ABC):
    """
    Creates a soccer goal object for the soccer benchmark task.

    The goal dimensions are: width=0.2, height=0.15, depth=0.1.
    The position is static and is defined in the corresponding xml file.

    Uses assets from Meta-World project.
    """

    def __init__(self, name: str):
        super(SoccerGoal, self).__init__(name)

    @staticmethod
    def _file_path():
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def xml_file_path(self):
        return self._file_path() + "/assets/soccer_objects.xml"

    def get_poi(self) -> list:
        return [self.name]
