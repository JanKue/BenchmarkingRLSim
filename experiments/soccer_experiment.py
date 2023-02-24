import time

from cw2 import experiment, cw_error, cluster_work
from cw2.cw_data import cw_logging

from training import training_soccer_env


class SoccerExperiment(experiment.AbstractExperiment):

    def initialize(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:
        pass

    def run(self, cw_config: dict, rep: int, logger: cw_logging.LoggerArray) -> None:

        params = cw_config['params']
        rep_path = cw_config['path'] + "/rep" + str(rep)

        add_delays = {'DDPG': 0, 'TD3': 10, 'SAC': 20, 'PPO': 30}
        add_delay = add_delays[params['algorithm']]
        time.sleep((rep + add_delay) * 30)

        training_soccer_env.main(env_name=params['env_name'], path=rep_path, total_steps=params['total_steps'],
                                 seed=rep, algorithm=params['algorithm'])

        return

    def finalize(self, surrender: cw_error.ExperimentSurrender = None, crash: bool = False):
        pass


if __name__ == '__main__':
    cw = cluster_work.ClusterWork(SoccerExperiment)

    cw.run()
