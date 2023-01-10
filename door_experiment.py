from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from envs_application import training_open_door
import time


class DoorExperiment(experiment.AbstractExperiment):

    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:


        params = cw_config['params']
        env_version = params['env_version']
        env_name = "DoorOpenEnv-v" + str(env_version)

        time.sleep((env_version - 20) * 60)

        training_open_door.main(env_name=env_name, path=cw_config['path'], total_steps=params['total_steps'])

        return

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == '__main__':

    cw = cluster_work.ClusterWork(DoorExperiment)

    cw.run()
