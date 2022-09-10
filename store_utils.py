import json
import os
import sys

import matplotlib.pyplot as plt

from datasets.data_util import get_timestamp

home_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(home_dir))


def get_experimental_result_dir(dir):
    output_full_dir = home_dir + dir + "/"
    if not os.path.exists(output_full_dir):
        os.makedirs(output_full_dir)
    return output_full_dir


def get_experimental_result_full_name(dir, filename):
    output_full_dir = get_experimental_result_dir(dir)

    timestamp = get_timestamp()
    file_name = filename + "_" + timestamp

    file_full_name = os.path.join(output_full_dir, file_name)
    return file_full_name


def save_exp_result_to_json(result_json, file_full_name):
    with open(file_full_name, 'w') as outfile:
        json.dump(result_json, outfile)


def save_experimental_results(experiment_result, output_directory_name, task_name, show_fig):
    loss_list = experiment_result["loss_list"]
    acc_list = experiment_result["metrics"]["acc_list"]
    auc_list = experiment_result["metrics"]["auc_list"]
    output_directory_full_name = output_directory_name
    file_full_name = get_experimental_result_full_name(output_directory_full_name, task_name)
    plt.subplot(131)
    plt.plot(loss_list)
    plt.xlabel("loss")
    plt.subplot(132)
    plt.plot(acc_list)
    plt.xlabel("accuracy")
    plt.subplot(133)
    plt.plot(auc_list)
    plt.xlabel("auc")
    if show_fig:
        plt.show()
    else:
        fig_full_name = file_full_name + '.png'
        plt.savefig(fig_full_name)
    json_full_name = file_full_name + '.json'
    print("[INFO] save experimental results to {0}".format(json_full_name))
    save_exp_result_to_json(experiment_result, json_full_name)
