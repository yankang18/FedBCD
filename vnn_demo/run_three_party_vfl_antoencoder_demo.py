import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from models.autoencoder import Autoencoder
from datasets.medical_dataset import load_vertical_medical_data
from vnn_demo.vfl_fixture import FederatedLearningFixture
from vfl import VFLHostModel, VFLGuestModel, VerticalMultiplePartyLogisticRegressionFederatedLearning


def benchmark_test(X_train, X_test, y_train, y_test, party_name=""):
    lr = LogisticRegression(solver="lbfgs")
    lr.fit(X_train, y_train)
    y_pred = lr.predict(X_test)
    score = lr.score(X_test, y_test)
    print(party_name + " score:", score)
    print(precision_recall_fscore_support(y_test, y_pred, average="weighted"))
    print(party_name + " auc", roc_auc_score(y_test, y_pred, average="weighted"))


def balance_X_y(X, y, seed=5):
    np.random.seed(seed)
    num_pos = np.sum(y == 1)
    num_neg = np.sum(y == -1)
    # print("pos samples", num_pos)
    # print("neg samples", num_neg)
    pos_indexes = [i for (i, _y) in enumerate(y) if _y > 0]
    neg_indexes = [i for (i, _y) in enumerate(y) if _y < 0]

    if num_pos < num_neg:
        np.random.shuffle(neg_indexes)
        rand_indexes = neg_indexes[:num_pos]
        indexes = pos_indexes + rand_indexes
        y = [y[i] for i in indexes]
        X = [X[i] for i in indexes]
    return np.array(X), np.array(y)


if __name__ == '__main__':

    print("################################ Prepare Data ############################")
    # change to the directory where you store NUS_WIDE datasets
    # data_dir = "../../data/"
    # train, test = load_horizontal_medical_data(data_dir)
    # Xa_train, Xb_train, y_train = train
    # Xa_test, Xb_test, y_test = test

    data_dir = "../../data/vertical_medical_data/"
    train, test = load_vertical_medical_data(data_dir)
    Xa_train, Xb_train, Xc_train, y_train = train
    Xa_test, Xb_test, Xc_test, y_test = test

    print("pos lbls train", sum(y_train), len(y_train))
    print("pos lbls test", sum(y_test), len(y_test))

    print("################################ Run Benchmark Models ############################")
    y_train_1d = np.ravel(y_train)
    y_test_1d = np.ravel(y_test)
    benchmark_test(Xa_train, Xa_test, y_train_1d, y_test_1d, "party A")
    benchmark_test(Xb_train, Xb_test, y_train_1d, y_test_1d, "party B")
    benchmark_test(Xc_train, Xc_test, y_train_1d, y_test_1d, "party C")

    X_train = np.concatenate([Xa_train, Xb_train, Xc_train], axis=1)
    X_test = np.concatenate([Xa_test, Xb_test, Xc_test], axis=1)
    benchmark_test(X_train, X_test, y_train_1d, y_test_1d, "All")

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    autoencoder_A = Autoencoder(1)
    autoencoder_B = Autoencoder(2)
    autoencoder_C = Autoencoder(3)

    autoencoder_A.build(input_dim=Xa_train.shape[1], hidden_dim=48, learning_rate=0.001)
    autoencoder_B.build(input_dim=Xb_train.shape[1], hidden_dim=32, learning_rate=0.001)
    autoencoder_C.build(input_dim=Xc_train.shape[1], hidden_dim=48, learning_rate=0.001)

    partyA = VFLGuestModel(local_model=autoencoder_A, n_iter=3, learning_rate=0.001, lbda=0.01, optimizer="adam")
    partyB = VFLHostModel(local_model=autoencoder_B, n_iter=1, learning_rate=0.001, lbda=0.01, optimizer="adam")
    partyC = VFLHostModel(local_model=autoencoder_C, n_iter=1, learning_rate=0.001, lbda=0.01, optimizer="adam")

    party_B_id = "B"
    party_C_id = "C"
    federatedLearning = VerticalMultiplePartyLogisticRegressionFederatedLearning(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)
    federatedLearning.add_party(id=party_C_id, party_model=partyC)

    print("################################ Train Federated Models ############################")

    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train, party_C_id: Xc_train}}

    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test, party_C_id: Xc_test}}

    fl_fixture = FederatedLearningFixture(federatedLearning)
    fl_fixture.fit(train_data=train_data, test_data=test_data, epochs=150, batch_size=128)

