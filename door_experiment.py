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

        lr = params['learning_rate']
        lrd = 0 if lr == 0.0003 else 5 if lr == 0.001 else 10 if lr == 0.005 else 15
        delay = (lrd + rep) * 60 + (env_version - 26) * 30
        time.sleep(delay)

        training_open_door.main(env_name=env_name, path=cw_config['path'], total_steps=params['total_steps'],
                                learning_rate=params['learning_rate'], seed=rep)

        return

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == '__main__':

    cw = cluster_work.ClusterWork(DoorExperiment)

    cw.run()
