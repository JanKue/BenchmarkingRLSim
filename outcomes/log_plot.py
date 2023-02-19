from tensorboard.backend.event_processing import event_accumulator
import glob
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
import tikzplotlib as tpl
from scipy.ndimage import gaussian_filter1d


selected_scalar = 'eval/mean_reward'


def parse_tensorboard(filepath):
    ea = event_accumulator.EventAccumulator(
        filepath,
        size_guidance={event_accumulator.SCALARS: 0},
    )
    _absorb_print = ea.Reload()
    # scalars = ea.Tags()['scalars']
    scalars = ['eval/success_rate', 'eval/mean_reward']

    return pd.DataFrame(ea.Scalars(selected_scalar))


def get_dataframes(glob_path, smooth=False):

    paths = glob.glob(glob_path, recursive=True)
    dfs = [parse_tensorboard(path) for path in paths]
    extracted_dfs = [df.drop(labels='wall_time', axis=1) for df in dfs]

    if smooth:
        def smoothing(df):
            df['value'] = gaussian_filter1d(df['value'], sigma=2)
            return df

        smoothed_dfs = [smoothing(df) for df in extracted_dfs]
        concat_dfs = pd.concat(smoothed_dfs)

    else:
        concat_dfs = pd.concat(extracted_dfs)

    # concat_dfs = concat_dfs[concat_dfs['step'] % 20_000 == 0]  # only keep every 2nd eval step

    return concat_dfs


def main():
    plt.close('all')

    ddpg_data_concat = get_dataframes("./cluster/soccer_task_experiment/soccer_random_ddpg/**/events.*")
    td3_data_concat = get_dataframes("./cluster/soccer_task_experiment/soccer_random_td3/**/events.*")
    sac_data_concat = get_dataframes("./cluster/soccer_task_experiment/soccer_random_sac/**/events.*")
    ppo_data_concat = get_dataframes("./cluster/soccer_task_experiment/soccer_random_ppo/**/events.*")

    plt.figure()
    plt.grid(visible=True)
    plt.xlim(0, 5e6)
    if selected_scalar == 'eval/success_rate':
        plt.ylim(0, 1)  # only for success rates

    error = ('se', 2)
    sns.lineplot(ddpg_data_concat, x='step', y='value', estimator='mean', errorbar=error, label='DDPG')
    sns.lineplot(td3_data_concat, x='step', y='value', estimator='mean', errorbar=error, label='TD3')
    sns.lineplot(sac_data_concat, x='step', y='value', estimator='mean', errorbar=error, label='SAC')
    sns.lineplot(ppo_data_concat, x='step', y='value', estimator='mean', errorbar=error, label='PPO')

    # grouped_df = concat_df.groupby(concat_df.index)

    # for df in all_dfs:
    #     df['value'] = gaussian_filter1d(df['value'], sigma=2)
    #     sns.lineplot(data=df, x='step', y='value').set_title('TD3: success rate')

    plt.savefig(fname="./plots/soccer_mean_rewards.svg")
    tpl.clean_figure()
    tpl.save(filepath="./plots/soccer_mean_rewards.tex")

    plt.show()


if __name__ == '__main__':
    main()
