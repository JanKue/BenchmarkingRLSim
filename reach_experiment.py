import time

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from envs_application import training_reach


class ReachExperiment(experiment.AbstractExperiment):

    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:

        params = cw_config['params']
        rep_path = cw_config['path'] + "/rep" + str(rep)

        time.sleep(rep * 60)

        training_reach.main(env_name=params['env_name'], path=rep_path, algorithm=params['algorithm'],
                            total_steps=params['total_steps'], seed=rep)

        return

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == '__main__':

    cw = cluster_work.ClusterWork(ReachExperiment)

    cw.run()
