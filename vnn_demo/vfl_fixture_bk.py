import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from datasets.data_util import convert_to_pos_neg_labels


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


class FederatedLearningFixture(object):

    def __init__(self):
        self.federatedLearning = None

    def set_federated_learning(self, federated_learning):
        self.federatedLearning = federated_learning

    def fit(self, train_data, test_data, epochs=50, batch_size=-1, show_fig=True):

        # TODO: add early stopping

        Xa_train, Xb_train, y_train = train_data
        Xa_test, Xb_test, y_test = test_data

        # only labels for training should be converted to {-1, 1}
        y_train_cvted = convert_to_pos_neg_labels(y_train)
        y_train_cvted = np.expand_dims(y_train_cvted, axis=1)

        N, _ = Xa_train.shape
        residual = N % batch_size
        if residual == 0:
            n_batches = N // batch_size
        else:
            n_batches = N // batch_size + 1

        print("number of samples:", N)
        print("batch size:", batch_size)
        print("number of batches:", n_batches)

        threshold = 0.5
        init = tf.global_variables_initializer()
        with tf.Session() as sess:
            self.federatedLearning.set_session(sess)

            sess.run(init)
            loss_list = []
            acc_list = []
            auc_list = []
            for ep in range(epochs):
                for i in range(n_batches):
                    Xa_batch = Xa_train[i * batch_size: i * batch_size + batch_size]
                    Xb_batch = Xb_train[i * batch_size: i * batch_size + batch_size]
                    Y_batch = y_train_cvted[i * batch_size: i * batch_size + batch_size]

                    loss = self.federatedLearning.fit(Xa_batch, Xb_batch, Y_batch)

                    if i % 5 == 0:
                        loss_list.append(loss)
                        print("epoch:", ep, "batch:", i, "loss:", loss)

                if ep % 1 == 0:
                    y_pred = self.federatedLearning.predict(Xa_test, Xb_test)
                    y_hat_lbls, stastics = compute_correct_prediction(y_targets=y_test,
                                                                      y_prob_preds=y_pred,
                                                                      threshold=threshold)
                    pred_pos_count, pred_neg_count, correct_count = stastics
                    acc = correct_count / len(y_test)
                    auc = roc_auc_score(y_test, y_hat_lbls, average="weighted")
                    acc_list.append(acc)
                    auc_list.append(auc)
                    print("--- neg：", pred_neg_count, "pos:", pred_pos_count)
                    print("--- num of correct:", correct_count)
                    print("--- epoch:", ep, "batch:", i, "loss:", loss, "acc:", acc, "auc:", auc)
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
