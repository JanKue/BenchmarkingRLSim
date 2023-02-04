from tensorboard.backend.event_processing import event_accumulator
import glob
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tikzplotlib as tpl
from scipy.ndimage import gaussian_filter1d


def parse_tensorboard(filepath):

    ea = event_accumulator.EventAccumulator(
        filepath,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    scalars = ea.Tags()['scalars']
    scalars = ['eval/success_rate', 'eval/mean_reward']

    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def get_dataframes(glob_path):

    paths = glob.glob(glob_path, recursive=True)
    dfs = [parse_tensorboard(path) for path in paths]
    success_dfs = [result['eval/success_rate'].drop(labels='wall_time', axis=1) for result in dfs]
    reward_dfs = [result['eval/mean_reward'].drop(labels='wall_time', axis=1) for result in dfs]

    success_concat = pd.concat(success_dfs)
    # success_concat = success_concat[success_concat['step'] % 20_000 == 0]  # only keep every 2nd eval step

    reward_concat = pd.concat(reward_dfs)
    # reward_concat = reward_concat[reward_concat['step'] % 20_000 == 0]  # only keep every 2nd eval step

    return success_concat, reward_concat


def main():

    plt.close('all')

    sac_success_concat, sac_reward_concat = get_dataframes("./cluster/reach_task_random/reach_random_init_sac/**/events.*")
    ppo_success_concat, ppo_reward_concat = get_dataframes("./cluster/reach_task_random/reach_random_init_ppo_old/**/events.*")
    ddpg_success_concat, ddpg_reward_concat = get_dataframes("./cluster/reach_task_random/reach_random_init_ddpg/**/events.*")
    td3_success_concat, td3_reward_concat = get_dataframes("./cluster/reach_task_random/reach_random_init_td3/**/events.*")

    plt.figure()
    plt.xlim(0, 5e6)
    plt.ylim(0, 1)
    sns.lineplot(sac_success_concat, x='step', y='value', estimator='mean', errorbar=('sd', 2))
    sns.lineplot(ppo_success_concat, x='step', y='value', estimator='mean', errorbar=('sd', 2))
    sns.lineplot(ddpg_success_concat, x='step', y='value', estimator='mean', errorbar=('sd', 2))
    sns.lineplot(td3_success_concat, x='step', y='value', estimator='mean', errorbar=('sd', 2))

    # grouped_df = concat_df.groupby(concat_df.index)

    # for df in all_dfs:
    #     df['value'] = gaussian_filter1d(df['value'], sigma=2)
    #     sns.lineplot(data=df, x='step', y='value').set_title('TD3: success rate')

    # plt.savefig(fname="./plots/reach_td3_success_rates.svg")
    # tpl.clean_figure()
    # tpl.save(filepath="./plots/reach_td3_success_rates.tex")

    plt.show()


if __name__ == '__main__':
    main()

