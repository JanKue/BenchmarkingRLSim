from tensorboard.backend.event_processing import event_accumulator
import tensorflow as tf
import glob
import pandas as pd
import os
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

    return {k: pd.DataFrame(ea.Scalars(k)) for k in scalars}


def main():

    plt.close('all')

    all_paths = glob.glob("./cluster/reach_task_random/reach_random_init_td3/**/events.*", recursive=True)
    print(all_paths)
    all_results = [parse_tensorboard(path) for path in all_paths]
    all_dfs = [result['eval/success_rate'] for result in all_results]

    plt.figure()

    for df in all_dfs:
        df['value'] = gaussian_filter1d(df['value'], sigma=2)
        sns.lineplot(data=df, x='step', y='value').set_title('TD3: success rate')

    plt.savefig(fname="./plots/reach_td3_success_rates.svg")
    # tpl.clean_figure()
    # tpl.save(filepath="./plots/reach_td3_success_rates.tex")
    plt.show()


if __name__ == '__main__':
    main()

