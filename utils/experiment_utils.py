import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_numpy_medical_data(file_full_path):
    return np.load(file_full_path)


def load_pandas_medical_data(file_full_path, header="infer"):
    return pd.read_csv(file_full_path, header=header)


def moving_average(values, window):
    weights = np.repeat(1.0, window) / window
    smas = np.convolve(values, weights, 'valid')
    return smas


def records_mean(records, period):
    records_array = np.array(records)
    internal = int(len(records) / period)
    # print("internal {0}".format(internal))
    new_records = []
    for i in range(internal):
        start_idx = i * period
        end_idx = i * period + period
        result = np.mean(records_array[start_idx:end_idx], axis=0)
        new_records.append(result)
    return new_records


def convert_learning_rate_to_string(learning_rate):
    lr_str = str(learning_rate)
    return lr_str.replace(".", "")


def calculate_ave_for_visulization(file_list, include_time=False, window=50, period=6):
    pf_list = []
    for file in file_list:
        pf = load_pandas_medical_data(file, header=None)
        pf_list.append(pf)

    num_columns = 100
    loss_records = []
    acc_records = []
    auc_records = []
    time_records = []
    for pf in pf_list:
        result_matrix = pf.values
        # print("result_matrix shape:", result_matrix.shape)
        loss = moving_average(result_matrix[0, :num_columns], window=window)
        loss_records.append(loss)
        # loss_records.append(result_matrix[0, :])

        acc = moving_average(result_matrix[1, :num_columns], window=window)
        acc_records.append(acc)
        # acc_records.append(result_matrix[1, :])

        auc = moving_average(result_matrix[2, :num_columns], window=window)
        auc_records.append(auc)
        # auc_records.append(result_matrix[2, :])
        if include_time:
            time = moving_average(result_matrix[3, :num_columns], window=window)

            # print("time1, ", time, len(time))
            for i in range(len(time)):
                if i != 0:
                    time[i] = time[i] + time[i - 1]
            # print("time2, ", time, len(time))
            time_records.append(time)

    loss_records = records_mean(loss_records, period=period)
    acc_records = records_mean(acc_records, period=period)
    auc_records = records_mean(auc_records, period=period)
    if include_time:
        time_records = records_mean(time_records, period=period)
        return loss_records, acc_records, auc_records, time_records
    else:
        return loss_records, acc_records, auc_records


def average_records_and_visualize(file_list, legend_list, include_time=False, window=50, period=6,
                                  bar_score=None, title=None):
    output = calculate_ave_for_visulization(file_list, include_time=include_time, window=window, period=period)
    if_metric_log = [False, False, False]
    if include_time:
        loss_records, acc_records, auc_records, time_records = output
        plot_result(legend_list, loss_records, acc_records, auc_records, metric_name="acc",
                    if_metric_log=if_metric_log, time_records=time_records, x_axis_label="running time", 
                    bar_score=bar_score, title=title)
    else:
        loss_records, acc_records, auc_records = output
        print("comm. rounds to 0.82:", compute_reached_threshold_rounds(auc_records, 0.82))
        print("comm. rounds to 0.83:", compute_reached_threshold_rounds(auc_records, 0.83))
        print("comm. rounds to 0.835:", compute_reached_threshold_rounds(auc_records, 0.835))
        print("comm. rounds to 0.837:", compute_reached_threshold_rounds(auc_records, 0.837))
        print("comm. rounds to 0.839:", compute_reached_threshold_rounds(auc_records, 0.839))

        print("comm. rounds to 0.990:", compute_reached_threshold_rounds(auc_records, 0.99))
        print("comm. rounds to 0.995:", compute_reached_threshold_rounds(auc_records, 0.995))
        print("comm. rounds to 0.997:", compute_reached_threshold_rounds(auc_records, 0.997))
        print("comm. rounds to 0.999:", compute_reached_threshold_rounds(auc_records, 0.999))
        plot_result(legend_list, loss_records, acc_records, auc_records, "acc",
                    if_metric_log=if_metric_log, bar_score=bar_score, title=title)


def visualize(loss_records, acc_records, auc_records, time_records, legend_list):
    if_metric_log = [False, False, False]
    if time_records is not None:
        plot_result(legend_list, loss_records, acc_records, auc_records, metric_name="acc",
                    if_metric_log=if_metric_log, time_records=time_records, x_axis_label="running time")
    else:
        plot_result(legend_list, loss_records, acc_records, auc_records, "acc",
                    if_metric_log=if_metric_log)


def plot_result(lengend_list, loss_records, metric_test_records, metric_train_records,
                metric_name, if_metric_log, time_records=None, x_axis_label="communication rounds", bar_score=None, title=None):

    # params = {'legend.fontsize': 12,
    #           'pdf.fonttype': 42,
    #           'legend.handlelength': 1,
    #           'font.size': 14,
    #           'xtick.labelsize': 10,
    #           'ytick.labelsize': 10}
    # params = {'pdf.fonttype': 42}
    plt.rcParams['pdf.fonttype'] = 42

    # style_list = ["r", "b", "g", "k", "m", "y", "c"]
    # style_list = ["r", "g", "g--", "k", "k--", "y", "y--"]
    # style_list = ["r", "b", "g", "k", "r--", "b--", "g--", "k--"]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]
    # style_list = ["r", "b", "g", "r--", "b--", "g--", "r-.", "b-.", "g-."]

    # style_list = ["r", "b", "g", "k", "m", "y", "c"]
    style_list = ["orchid", "red", "green", "blue", "purple", "peru", "olive", "coral"]

    if len(lengend_list) == 5:
        style_list = ["orchid", "r", "b", "r--", "b--"]

    if len(lengend_list) == 6:
        # style_list = ["orchid", "r", "g", "b", "purple", "peru", "olive", "coral"]
        style_list = ["r", "g", "b", "r--", "g--", "b--"]
        # style_list = ["r", "r--", "b", "b--", "g", "g--"]
        # style_list = ["m", "r", "g", "b", "c", "y", "k"]

    if len(lengend_list) == 7:
        # style_list = ["m", "r", "g", "b", "r--", "g--", "b--"]
        style_list = ["orchid", "r", "g", "b", "r--", "g--", "b--"]

    if len(lengend_list) == 8:
        style_list = ["r", "b", "g", "k", "r--", "b--", "g--", "k--"]

    if len(lengend_list) == 9:
        style_list = ["r", "r--", "r:", "b", "b--", "b:", "g", "g--", "g:"]

    legend_size = 14
    markevery = 50
    markesize = 3

    plt.subplot(111)
    if time_records is None:
        for i, loss_list in enumerate(loss_records):
            if if_metric_log[0]:
                plt.semilogy(loss_list, style_list[i], markersize=markesize, markevery=markevery)
            else:
                # loss_list = loss_list / 1000
                plt.plot(loss_list, style_list[i], markersize=markesize, markevery=markevery)
                # plt.ylim(0.6, 1.0)
    else:
        for i, rest in enumerate(zip(time_records, loss_records)):
            time_list = rest[0]
            loss_list = rest[1]
            if if_metric_log[0]:
                plt.semilogy(time_list, loss_list, style_list[i], markersize=markesize, markevery=markevery)
            else:
                plt.plot(time_list, loss_list, style_list[i], markersize=markesize, markevery=markevery)
                plt.ylim(0.6, 1.0)
    # plt.xlabel(x_axis_label)
    # plt.ylabel("loss")
    # plt.legend(lengend_list, loc='best')
    # plt.show()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(x_axis_label, fontsize=16)
    plt.ylabel("Loss", fontsize=16)
    plt.title(title[0], fontsize=16)
    plt.legend(lengend_list, fontsize=legend_size, loc='best')
    plt.show()

    # plt.subplot(132)
    # if time_records is None:
    #     for i, metric_test_list in enumerate(metric_test_records):
    #         if if_metric_log[1]:
    #             plt.semilogy(metric_test_list, style_list[i], markersize=markesize, markevery=markevery)
    #         else:
    #             plt.plot(metric_test_list, style_list[i], markersize=markesize, markevery=markevery)
    # else:
    #     for i, rest in enumerate(zip(time_records, metric_test_records)):
    #         time_list = rest[0]
    #         metric_test_list = rest[1]
    #         print("len time_list", len(time_list))
    #         print("len metric_test_list", len(metric_test_list))
    #         if if_metric_log[1]:
    #             plt.semilogy(time_list, metric_test_list, style_list[i], markersize=markesize, markevery=markevery)
    #         else:
    #             plt.plot(time_list, metric_test_list, style_list[i], markersize=markesize, markevery=markevery)
    # plt.xlabel(x_axis_label)
    # plt.ylabel("test " + metric_name)
    # plt.legend(lengend_list, loc='best')

    plt.subplot(111)
    if bar_score is not None:
        plt.hlines(bar_score, xmin=-100, xmax=1100, colors="grey")
    if time_records is None:
        for i, metric_train_list in enumerate(metric_train_records):
            if if_metric_log[2]:
                plt.semilogy(metric_train_list, style_list[i], markersize=markesize, markevery=markevery)
            else:
                plt.plot(metric_train_list, style_list[i], markersize=markesize, markevery=markevery)
    else:
        for i, rest in enumerate(zip(time_records, metric_train_records)):
            time_list = rest[0]
            metric_train_list = rest[1]
            if if_metric_log[2]:
                plt.semilogy(time_list, metric_train_list, style_list[i], markersize=markesize, markevery=markevery)
            else:
                plt.plot(time_list, metric_train_list, style_list[i], markersize=markesize, markevery=markevery)
    # plt.xlabel(x_axis_label)
    # plt.ylabel("test auc")
    # plt.legend(lengend_list, loc='best')
    # plt.show()
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.xlabel(x_axis_label, fontsize=16)
    plt.ylabel("AUC", fontsize=16)
    plt.title(title[1], fontsize=16)
    plt.legend(lengend_list, fontsize=legend_size, loc='best')
    plt.show()


def compute_reached_threshold_rounds(metric_records, metric_threshold, count_threshold=3):
    round_list = []
    for metric_record in metric_records:
        count = 0
        round_found = False
        for comm_round, metric in enumerate(metric_record):
            if metric > metric_threshold:
                count += 1
            if count == count_threshold:
                round_list.append(comm_round)
                round_found = True
                break

        if not round_found:
            round_list.append(-1)
    return round_list