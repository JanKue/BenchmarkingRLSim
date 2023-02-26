from abc import ABC

from alr_sim.core.sim_object.sim_object import SimObject
from alr_sim.sims.mujoco.MujocoLoadable import MujocoXmlLoadable


class HammerObjects(SimObject, MujocoXmlLoadable, ABC):

    def __init__(self,
                 name: str,
                 init_pos=[0.0, 0.0, 0.0],
                 init_quat=[0.0, 0.0, 0.0, 0.0]):
        super(HammerObjects, self).__init__(name, init_pos, init_quat)

        path_to_alr_sim_framework = "/home/jan/SimulationFramework"
        self.xml_file_path = path_to_alr_sim_framework + "/models/mujoco/objects/hammer_objects.xml"

    def xml_file_path(self):
        return self.xml_file_path

    def get_poi(self) -> list:
        return [self.name]
