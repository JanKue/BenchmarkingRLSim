from tensorboard.backend.event_processing import event_accumulator
import glob
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tikzplotlib as tpl
from scipy.ndimage import gaussian_filter1d


selected_scalar = 'eval/success_rate'


def parse_tensorboard(filepath):

    ea = event_accumulator.EventAccumulator(
        filepath,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # scalars = ea.Tags()['scalars']
    scalars = ['eval/success_rate', 'eval/mean_reward']

    return pd.DataFrame(ea.Scalars(selected_scalar))


def get_dataframes(glob_path):

    paths = glob.glob(glob_path, recursive=True)
    dfs = [parse_tensorboard(path) for path in paths]
    extracted_dfs = [df.drop(labels='wall_time', axis=1) for df in dfs]
    concat_dfs = pd.concat(extracted_dfs)
    # concat_dfs = concat_dfs[concat_dfs['step'] % 20_000 == 0]  # only keep every 2nd eval step

    return concat_dfs


def main():

    plt.close('all')

    sac_data_concat = get_dataframes("./cluster/door_open_task/door_open_sac_experiment/**/events.*")
    ppo_data_concat = get_dataframes("./cluster/door_open_task/door_open_ppo_experiment/**/events.*")
    ddpg_data_concat = get_dataframes("./cluster/door_open_task/door_open_ddpg_experiment/**/events.*")
    td3_data_concat = get_dataframes("./cluster/door_open_task/door_open_td3_experiment/**/events.*")

    plt.figure()
    plt.xlim(0, 5e6)
    # plt.ylim(0, 1)
    sns.lineplot(sac_data_concat, x='step', y='value', estimator='mean', errorbar=('sd', 1), label='SAC')
    sns.lineplot(ppo_data_concat, x='step', y='value', estimator='mean', errorbar=('sd', 1), label='PPO')
    sns.lineplot(ddpg_data_concat, x='step', y='value', estimator='mean', errorbar=('sd', 1), label='DDPG')
    sns.lineplot(td3_data_concat, x='step', y='value', estimator='mean', errorbar=('sd', 1), label='TD3')

    # grouped_df = concat_df.groupby(concat_df.index)

    # for df in all_dfs:
    #     df['value'] = gaussian_filter1d(df['value'], sigma=2)
    #     sns.lineplot(data=df, x='step', y='value').set_title('TD3: success rate')

    plt.savefig(fname="./plots/door_success_rates.svg")
    tpl.clean_figure()
    tpl.save(filepath="./plots/door_success_rates.tex")

    plt.show()


if __name__ == '__main__':
    main()
