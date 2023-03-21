import os
from abc import ABC

from alr_sim.core.sim_object.sim_object import SimObject
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable


class HammerObjects(SimObject, MujocoXmlLoadable, ABC):

    def __init__(self, name: str):
        super(HammerObjects, self).__init__(name)

    @staticmethod
    def _file_path():
        return os.path.dirname(os.path.abspath(__file__))

    @property
    def xml_file_path(self):
        return self._file_path() + "/assets/hammer_objects.xml"

    def get_poi(self) -> list:
        return [self.name]
