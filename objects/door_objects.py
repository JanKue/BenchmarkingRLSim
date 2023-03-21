import os
from abc import ABC

from alr_sim.core.sim_object.sim_object import SimObject
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable


class DoorBox(SimObject, MujocoXmlLoadable, ABC):
    """
    Creates a door-box object for the door opening benchmark task.

    The object consists of a large box, attached to which is a door on a hinge.
    The door has a handle large enough for the robot to use.
    The position is defined in the corresponding xml file.

    Uses assets from Meta-World project.
    """

    def __init__(self, name: str):
        super(DoorBox, self).__init__(name)

    @staticmethod
    def _file_path():
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def xml_file_path(self):
        return self._file_path() + "/assets/door_objects.xml"

    def get_poi(self) -> list:
        return [self.name]
