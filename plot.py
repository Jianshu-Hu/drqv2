import matplotlib.pyplot as plt
import numpy as np
import math
import os

# eval_env_type = ['normal', 'color_hard', 'video_easy', 'video_hard']
eval_env_type = ["normal"]


def average_over_several_runs(folder):
    mean_all = []
    std_all = []
    for env_type in range(len(eval_env_type)):
        data_all = []
        min_length = np.inf
        runs = os.listdir(folder)
        for i in range(len(runs)):
            data = np.loadtxt(
                folder + "/" + runs[i] + "/eval.csv", delimiter=",", skiprows=1
            )
            evaluation_freq = data[2, -3] - data[1, -3]
            data_all.append(data[:, 2 + env_type])
            if data.shape[0] < min_length:
                min_length = data.shape[0]
        average = np.zeros([len(runs), min_length])
        for i in range(len(runs)):
            average[i, :] = data_all[i][:min_length]
        mean = np.mean(average, axis=0)
        mean_all.append(mean)
        std = np.std(average, axis=0)
        std_all.append(std)

    return mean_all, std_all, evaluation_freq / 1000


def plot_several_folders(
    prefix, folders, action_repeat, label_list=[], plot_or_save="save", title=""
):
    # plt.rcParams["figure.figsize"] = (8, 8)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
        folder_name = "saved_exps/" + prefix + folders[i]
        num_runs = len(os.listdir(folder_name))
        mean_all, std_all, eval_freq = average_over_several_runs(folder_name)
        for j in range(len(eval_env_type)):
            if len(eval_env_type) == 1:
                axs_plot = axs
            else:
                axs_plot = axs[int(j / 2)][j - 2 * (int(j / 2))]
            # plot variance
            if label_list[i] == "ours":
                axs_plot.fill_between(
                    eval_freq * range(len(mean_all[j])),
                    mean_all[j] - std_all[j] / math.sqrt(num_runs),
                    mean_all[j] + std_all[j] / math.sqrt(num_runs),
                    alpha=0.4,
                    color="C3",
                )
            else:
                axs_plot.fill_between(
                    eval_freq * range(len(mean_all[j])),
                    mean_all[j] - std_all[j] / math.sqrt(num_runs),
                    mean_all[j] + std_all[j] / math.sqrt(num_runs),
                    alpha=0.4,
                )
            if len(label_list) == len(folders):
                # specify label
                if label_list[i] == "ours":
                    axs_plot.plot(
                        eval_freq * range(len(mean_all[j])),
                        mean_all[j],
                        label=label_list[i],
                        color="C3",
                    )
                else:
                    axs_plot.plot(
                        eval_freq * range(len(mean_all[j])),
                        mean_all[j],
                        label=label_list[i],
                    )
            else:
                axs_plot.plot(
                    eval_freq * range(len(mean_all[j])), mean_all[j], label=folders[i]
                )

            axs_plot.set_xlabel("evaluation steps(x1000)")
            axs_plot.set_ylabel("episode reward")
            axs_plot.legend(fontsize=10)
            # axs_plot.set_title(eval_env_type[j])
            axs_plot.set_title(title)
    if plot_or_save == "plot":
        plt.show()
    else:
        plt.savefig("saved_figs/" + title)


tasks = ["walker_run"]
feat_aug_ind = [3, 5]
aug_folders = [f"aug1-feat_aug{i}" for i in feat_aug_ind]
labels = [
    # "lix",
    "rand_shear",
    # "rand_w_mean",
    "rand_white",
]
identifier = "e"
for task in tasks:
    prefix = f"{identifier}-{task}/"
    title = f"{identifier}_{task}"
    action_repeat = 2
    plot_several_folders(
        prefix, aug_folders, action_repeat, title=title, label_list=labels
    )
