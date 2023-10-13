import matplotlib.pyplot as plt
import numpy as np
import math
import os

# eval_env_type = ['normal', 'color_hard', 'video_easy', 'video_hard']
eval_env_type = ['normal']


def average_over_several_runs(folder):
    mean_all = []
    std_all = []
    for env_type in range(len(eval_env_type)):
        data_all = []
        min_length = np.inf
        runs = os.listdir(folder)
        for i in range(len(runs)):
            data = np.loadtxt(folder+'/'+runs[i]+'/eval.csv', delimiter=',', skiprows=1)
            evaluation_freq = data[2, -3]-data[1, -3]
            data_all.append(data[:, 2+env_type])
            if data.shape[0] < min_length:
                min_length = data.shape[0]
        average = np.zeros([len(runs), min_length])
        for i in range(len(runs)):
            average[i, :] = data_all[i][:min_length]
        mean = np.mean(average, axis=0)
        mean_all.append(mean)
        std = np.std(average, axis=0)
        std_all.append(std)

    return mean_all, std_all, evaluation_freq/1000


def plot_several_folders(prefix, folders, action_repeat, label_list=[], plot_or_save='save', title=""):
    plt.rcParams["figure.figsize"] = (5, 4)
    fig, axs = plt.subplots(1, 1)
    for i in range(len(folders)):
        folder_name = 'saved_exps/'+prefix+folders[i]
        num_runs = len(os.listdir(folder_name))
        mean_all, std_all, eval_freq = average_over_several_runs(folder_name)
        for j in range(len(eval_env_type)):
            if len(eval_env_type) == 1:
                axs_plot = axs
            else:
                axs_plot = axs[int(j/2)][j-2*(int(j/2))]
            # plot variance
            if label_list[i] == 'ours':
                axs_plot.fill_between(eval_freq*range(len(mean_all[j])),
                        mean_all[j] - std_all[j]/math.sqrt(num_runs),
                        mean_all[j] + std_all[j]/math.sqrt(num_runs), alpha=0.4, color='C3')
            else:
                axs_plot.fill_between(eval_freq*range(len(mean_all[j])),
                        mean_all[j] - std_all[j]/math.sqrt(num_runs),
                        mean_all[j] + std_all[j]/math.sqrt(num_runs), alpha=0.4)
            if len(label_list) == len(folders):
                # specify label
                if label_list[i] == 'ours':
                    axs_plot.plot(eval_freq*range(len(mean_all[j])), mean_all[j], label=label_list[i], color='C3')
                else:
                    axs_plot.plot(eval_freq * range(len(mean_all[j])), mean_all[j], label=label_list[i])
            else:
                axs_plot.plot(eval_freq*range(len(mean_all[j])), mean_all[j], label=folders[i])

            axs_plot.set_xlabel('evaluation steps(x1000)')
            axs_plot.set_ylabel('episode reward')
            axs_plot.legend(fontsize=10)
            # axs_plot.set_title(eval_env_type[j])
            axs_plot.set_title(title)
    if plot_or_save == 'plot':
        plt.show()
    else:
        plt.savefig('saved_figs/'+title)


prefix = 'quadruped_walk/'
action_repeat = 2
# folders_1 = ['drqv2', 'drqv2_aug_2', 'drqv2_aug_2_add_KL', 'drqv2_aug_2_add_KL_add_tangent_prop']
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent_prop']
label_list = ['drqv2', 'ours']
plot_several_folders(prefix, folders_1, action_repeat, title='quadruped_walk', label_list=label_list)

prefix = 'quadruped_run/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='quadruped_run', label_list=label_list)

prefix = 'reach_duplo/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='reach_duplo', label_list=label_list)

prefix = 'hopper_hop/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='hopper_hop', label_list=label_list)


prefix = 'acrobot_swingup/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='acrobot_swingup', label_list=label_list)

prefix = 'reacher_hard/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='reacher_hard', label_list=label_list)

prefix = 'finger_turn_hard/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='finger_turn_hard', label_list=label_list)

prefix = 'walker_run/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='walker_run', label_list=label_list)

prefix = 'finger_spin/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='finger_spin', label_list=label_list)

prefix = 'cheetah_run/'
action_repeat = 2
folders_1 = ['drqv2', 'drqv2_aug_2_add_KL_add_tangent']
plot_several_folders(prefix, folders_1, action_repeat, title='cheetah_run', label_list=label_list)




