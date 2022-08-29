import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import roc_auc_score

from datasets.data_util import convert_to_pos_neg_labels
from datasets.data_util import get_timestamp, save_result, compute_experimental_result_file_name
from vfl import VerticalMultiplePartyLogisticRegressionFederatedLearning

home_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(home_dir))


class FederatedLearningExperimentResult(object):
    def __init__(self, result_dict):
        self.result_dict = result_dict

    def get_epochs(self):
        return self.result_dict["epochs"]

    def is_parallel(self):
        return self.result_dict["is_parallel"]

    def get_global_step(self):
        return self.result_dict["global_step"]

    def get_recording_step(self):
        return self.result_dict["recording_step"]

    def get_loss_list(self):
        return self.result_dict["loss_list"]

    def get_running_time_list(self):
        return self.result_dict["running_time_list"]

    def get_acc_list(self):
        return self.result_dict["metrics"]["acc_list"]

    def get_auc_list(self):
        return self.result_dict["metrics"]["auc_list"]

    def get_batch_size(self):
        return self.result_dict["batch_size"]

    def get_local_iteration_list(self):
        return self.result_dict["local_iteration_list"]


def compute_correct_prediction(*, y_targets, y_prob_preds, threshold=0.5):
    y_hat_lbls = []
    pred_pos_count = 0
    pred_neg_count = 0
    correct_count = 0
    for y_prob, y_t in zip(y_prob_preds, y_targets):
        if y_prob <= threshold:
            pred_neg_count += 1
            y_hat_lbl = 0
        else:
            pred_pos_count += 1
            y_hat_lbl = 1
        y_hat_lbls.append(y_hat_lbl)
        if y_hat_lbl == y_t:
            correct_count += 1

    return np.array(y_hat_lbls), [pred_pos_count, pred_neg_count, correct_count]


def save_vfl_experiment_result(output_directory, task_name, experiment_result: FederatedLearningExperimentResult):
    local_iter_list = experiment_result.get_local_iteration_list()
    batch_size = experiment_result.get_batch_size()
    recording_step = experiment_result.get_recording_step()
    loss_list = experiment_result.get_loss_list()
    running_time_list = experiment_result.get_running_time_list()
    acc_list = experiment_result.get_acc_list()
    auc_list = experiment_result.get_auc_list()
    is_parallel = experiment_result.is_parallel()

    local_iterations = str(local_iter_list[0])
    for party_idx, n_iter in enumerate(local_iter_list):
        if party_idx != 0:
            local_iterations += "_" + str(local_iter_list[party_idx])

    loss_records = list()
    spend_time_records = list()
    acc_records = list()
    auc_records = list()
    loss_records.append(loss_list)
    spend_time_records.append(running_time_list)
    acc_records.append(acc_list)
    auc_records.append(auc_list)

    timestamp = get_timestamp()
    file_name = compute_experimental_result_file_name(n_local=local_iterations,
                                                      batch_size=batch_size,
                                                      comm_rounds=recording_step)

    output_full_dir = home_dir + output_directory + "/"
    output_file_full_name = output_full_dir + task_name + "_parallel_" + str(is_parallel) + "_" + file_name + "_" + timestamp
    save_result(file_full_name=output_file_full_name,
                loss_records=loss_records,
                metric_one_records=acc_records,
                metric_two_records=auc_records,
                spend_time_records=spend_time_records)


class FederatedLearningFixture(object):

    def __init__(self, federated_learning: VerticalMultiplePartyLogisticRegressionFederatedLearning):
        self.federated_learning = federated_learning

    def fit(self, train_data, test_data, is_parallel, epochs=50, batch_size=-1, show_fig=True):

        # TODO: add early stopping
        main_party_id = self.federated_learning.get_main_party_id()
        Xa_train = train_data[main_party_id]["X"]
        y_train = train_data[main_party_id]["Y"]
        Xa_test = test_data[main_party_id]["X"]
        y_test = test_data[main_party_id]["Y"]
        # Xa_train, Xb_train, y_train = train_data
        # Xa_test, Xb_test, y_test = test_data

        # only labels for training should be converted to {-1, 1}

        y_train_cvted = convert_to_pos_neg_labels(y_train.flatten())
        print("y_train_cvted1: {0}".format(y_train_cvted.shape))
        y_train_cvted = np.expand_dims(y_train_cvted, axis=1)
        print("y_train_cvted2: {0}".format(y_train_cvted.shape))

        N = Xa_train.shape[0]
        residual = N % batch_size
        if residual == 0:
            n_batches = N // batch_size
        else:
            n_batches = N // batch_size + 1

        print("number of samples:", N)
        print("batch size:", batch_size)
        print("number of batches:", n_batches)

        global_step = -1
        # the period in terms of global step to record information such as loss, accuracy and AUC
        recording_period = 1
        recording_step = -1
        threshold = 0.5
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.federated_learning.set_session(sess)

            sess.run(init)
            loss_list = []
            running_time_list = []
            acc_list = []
            auc_list = []
            for ep in range(epochs):
                for batch_idx in range(n_batches):
                    global_step += 1

                    Xa_batch = Xa_train[batch_idx * batch_size: batch_idx * batch_size + batch_size]
                    Y_batch = y_train_cvted[batch_idx * batch_size: batch_idx * batch_size + batch_size]

                    party_X_train_batch_dict = dict()
                    for party_id, party_X in train_data["party_list"].items():
                        party_X_train_batch_dict[party_id] = party_X[
                                                             batch_idx * batch_size: batch_idx * batch_size + batch_size]

                    if is_parallel:
                        loss, running_time = self.federated_learning.fit_parallel(Xa_batch, Y_batch,
                                                                                  party_X_train_batch_dict,
                                                                                  global_step)
                    else:
                        loss, running_time = self.federated_learning.fit(Xa_batch, Y_batch,
                                                                         party_X_train_batch_dict,
                                                                         global_step)

                    if global_step % recording_period == 0:
                        recording_step += 1
                        loss_list.append(loss)
                        running_time_list.append(running_time)
                        party_X_test_dict = dict()
                        for party_id, party_X in test_data["party_list"].items():
                            party_X_test_dict[party_id] = party_X
                        y_prob_preds = self.federated_learning.predict(Xa_test, party_X_test_dict)
                        y_hat_lbls, statistics = compute_correct_prediction(y_targets=y_test,
                                                                            y_prob_preds=y_prob_preds,
                                                                            threshold=threshold)
                        pred_pos_count, pred_neg_count, correct_count = statistics
                        acc = correct_count / len(y_test)
                        auc = roc_auc_score(y_test, y_prob_preds, average="weighted")
                        acc_list.append(acc)
                        auc_list.append(auc)
                        print("--- Validation: ---")
                        print("--- neg：", pred_neg_count, "pos:", pred_pos_count)
                        print("--- num of correct:", correct_count)
                        print("--- epoch: {0}, batch: {1}, loss: {2}, acc: {3}, auc: {4}"
                              .format(ep, batch_idx, loss, acc, auc))
                        print("---", precision_recall_fscore_support(y_test, y_hat_lbls, average="weighted"))

            if show_fig:
                plt.subplot(131)
                plt.plot(loss_list)
                plt.xlabel("loss")
                plt.subplot(132)
                plt.plot(acc_list)
                plt.xlabel("accuracy")
                plt.subplot(133)
                plt.plot(auc_list)
                plt.xlabel("auc")
                plt.show()

            print("loss : {0}".format(loss_list))
            print("acc : {0}".format(acc_list))
            print("auc : {0}".format(auc_list))

            result_dict = dict()
            result_dict["epochs"] = epochs
            result_dict["is_parallel"] = is_parallel
            result_dict["batch_size"] = batch_size
            result_dict["global_step"] = global_step
            result_dict["recording_step"] = recording_step
            result_dict["local_iteration_list"] = self.federated_learning.get_local_iteration_of_parties()
            result_dict["loss_list"] = loss_list
            result_dict["running_time_list"] = running_time_list
            result_dict["metrics"] = dict()
            result_dict["metrics"]["acc_list"] = acc_list
            result_dict["metrics"]["auc_list"] = auc_list

            return FederatedLearningExperimentResult(result_dict)
