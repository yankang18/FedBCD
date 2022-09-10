import numpy as np
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

import config
from datasets.data_util import load_data, split_data_combined, series_plot
from models.autoencoder import Autoencoder
from vnn_demo.plain_ftl import PlainFTLHostModel, PlainFTLGuestModel, LocalPlainFederatedTransferLearning

if __name__ == '__main__':

    infile = config.data_dir + "UCI_Credit_Card/UCI_Credit_Card.csv"
    X, y = load_data(infile=infile, balanced=True)

    X_A, y_A, X_B, y_B, overlap_indexes = split_data_combined(X, y,
                                                              overlap_ratio=0.1,
                                                              ab_split_ratio=0.1,
                                                              n_feature_b=23)

    valid_ratio = 0.3
    non_overlap_indexes = np.setdiff1d(range(X_B.shape[0]), overlap_indexes)
    validate_indexes = non_overlap_indexes[:int(valid_ratio * len(non_overlap_indexes))]
    test_indexes = non_overlap_indexes[int(valid_ratio*len(non_overlap_indexes)):]
    x_B_valid = X_B[validate_indexes]
    y_B_valid = y_B[validate_indexes]
    x_B_test = X_B[test_indexes]
    y_B_test = y_B[test_indexes]

    print("X_A shape", X_A.shape)
    print("y_A shape", y_A.shape)
    print("X_B shape", X_B.shape)
    print("y_B shape", y_B.shape)

    print("overlap_indexes len", len(overlap_indexes))
    # print("overlap_indexes", overlap_indexes)
    print("non_overlap_indexes len", len(non_overlap_indexes))
    # print("non_overlap_indexes", non_overlap_indexes)

    print("validate_indexes len", len(validate_indexes))
    print("test_indexes len", len(test_indexes))

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    autoencoder_A = Autoencoder(1)
    autoencoder_B = Autoencoder(2)

    autoencoder_A.build(X_A.shape[-1], 200, learning_rate=0.01)
    autoencoder_B.build(X_B.shape[-1], 200, learning_rate=0.01)

    alpha = 100
    partyA = PlainFTLGuestModel(autoencoder_A, alpha)
    partyB = PlainFTLHostModel(autoencoder_B, alpha)

    federatedLearning = LocalPlainFederatedTransferLearning(partyA, partyB)

    print("################################ Train Federated Models ############################")

    epochs = 100
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        autoencoder_A.set_session(sess)
        autoencoder_B.set_session(sess)

        sess.run(init)
        losses = []
        fscores = []
        aucs = []
        for ep in range(epochs):
            loss = federatedLearning.fit(X_A, X_B, y_A, overlap_indexes, non_overlap_indexes)
            losses.append(loss)

            if ep % 5 == 0:
                print("ep", ep, "loss", loss)
                y_pred = federatedLearning.predict(x_B_test)
                y_pred_label = []
                pos_count = 0
                neg_count = 0
                for _y in y_pred:
                    if _y <= 0.5:
                        neg_count += 1
                        y_pred_label.append(-1)
                    else:
                        pos_count += 1
                        y_pred_label.append(1)
                y_pred_label = np.array(y_pred_label)
                print("neg：", neg_count, "pos:", pos_count)
                precision, recall, fscore, _ = precision_recall_fscore_support(y_B_test, y_pred_label, average="weighted")
                fscores.append(fscore)
                auc = roc_auc_score(y_B_test, y_pred, average="weighted")
                aucs.append(auc)
                print("fscore, auc:", fscore, auc)

        series_plot(losses, fscores, aucs)


