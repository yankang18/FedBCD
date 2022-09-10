import numpy as np
import tensorflow as tf
from sklearn.metrics import roc_auc_score

from datasets.data_util import convert_to_pos_neg_labels
from vnn_demo.vfl import VerticalMultiplePartyFederatedLearning


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


class VerticalFederatedLearningLearner(object):

    def __init__(self, federated_learning: VerticalMultiplePartyFederatedLearning):
        self.federated_learning = federated_learning

    def fit(self, train_data, test_data, epochs=50, batch_size=-1, is_parallel=True, verbose=False, is_debug=False):

        # TODO: add early stopping
        main_party_id = self.federated_learning.get_main_party_id()
        Xa_train = train_data[main_party_id]["X"]
        y_train = train_data[main_party_id]["Y"]
        Xa_test = test_data[main_party_id]["X"]
        y_test = test_data[main_party_id]["Y"]

        # only training labels should be converted to {-1, 1}
        y_train_cvted = convert_to_pos_neg_labels(y_train.flatten())
        if is_debug: print("[DEBUG] y_train_cvted1: {0}".format(y_train_cvted.shape))
        y_train_cvted = np.expand_dims(y_train_cvted, axis=1)
        if is_debug: print("[DEBUG] y_train_cvted2: {0}".format(y_train_cvted.shape))

        N = Xa_train.shape[0]
        residual = N % batch_size
        if residual == 0:
            n_batches = N // batch_size
        else:
            n_batches = N // batch_size + 1

        print("[INFO] number of samples:{}; batch size:{}; number of batches:{}.", N, batch_size, n_batches)

        global_step = -1
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
                # global_step += 1
                for batch_idx in range(n_batches):
                    global_step += 1

                    Xa_batch = Xa_train[batch_idx * batch_size: batch_idx * batch_size + batch_size]
                    Y_batch = y_train_cvted[batch_idx * batch_size: batch_idx * batch_size + batch_size]

                    party_X_train_batch_dict = dict()
                    for party_id, party_X in train_data["party_list"].items():
                        party_X_train_batch_dict[party_id] = party_X[
                                                             batch_idx * batch_size: batch_idx * batch_size + batch_size]

                    if is_parallel:
                        if verbose: print("[INFO] fit using parallel scheme")
                        loss, running_time = self.federated_learning.fit_parallel(Xa_batch, Y_batch,
                                                                                  party_X_train_batch_dict,
                                                                                  global_step)
                    else:
                        if verbose: print("[INFO] fit using sequential scheme")
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
                        # print("--- Validation: ---")
                        # print("--- neg：", pred_neg_count, "pos:", pred_pos_count)
                        # print("--- num of correct:", correct_count)
                        print("=== epoch: {0}, batch: {1}, loss: {2}, acc: {3}, auc: {4}".format(ep,
                                                                                                 batch_idx,
                                                                                                 loss,
                                                                                                 acc,
                                                                                                 auc))
                        # print("---", precision_recall_fscore_support(y_test, y_hat_lbls, average="weighted"))

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

            return result_dict
