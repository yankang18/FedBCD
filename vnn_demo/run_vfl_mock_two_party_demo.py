import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score
from sklearn.utils import shuffle

from datasets.nus_wide_dataset import load_prepared_parties_data, get_data_folder_name
from models.mock_model import MockModel
from vfl import VFLHostModel, VFLGuestModel, VerticalMultiplePartyLogisticRegressionFederatedLearning
from vnn_demo.vfl_fixture import FederatedLearningFixture


def benchmark_test(X_train, X_test, y_train, y_test, party_name=""):
    print("------ {0} ------".format(party_name))
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


def run_experiment(train_data, test_data, output_directory_name, show_fig=True):
    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    mock_model_A = MockModel(1)
    mock_model_B = MockModel(2)

    mock_model_A.build(hidden_dim=Xa_train.shape[1])
    mock_model_B.build(hidden_dim=Xb_train.shape[1])

    lr = 0.01

    partyA = VFLGuestModel(local_model=mock_model_A, n_iter=1, learning_rate=lr, lbda=0.01, is_ave=False,
                           optimizer="adam")
    partyB = VFLHostModel(local_model=mock_model_B, n_iter=1, learning_rate=lr, lbda=0.01, optimizer="adam")

    party_B_id = "B"
    federatedLearning = VerticalMultiplePartyLogisticRegressionFederatedLearning(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)

    print("################################ Train Federated Models ############################")

    output_directory_full_name = "/result/" + output_directory_name
    fl_fixture = FederatedLearningFixture()
    fl_fixture.set_federated_learning(federatedLearning)
    fl_fixture.set_output_directory(output_directory_full_name)
    fl_fixture.set_local_model_name("vfl_mock")

    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train}}

    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test}}

    fl_fixture.fit(train_data=train_data, test_data=test_data, epochs=10, batch_size=256, show_fig=show_fig)


if __name__ == '__main__':

    print("################################ Prepare Data ############################")
    for_three_party = False
    data_dir = "../../data/"
    # class_lbls = ['person', 'water', 'animal', 'grass', 'buildings']
    # class_lbls = ['sky', 'clouds', 'person', 'water']
    class_lbls = ['person', 'animal']
    folder_name = get_data_folder_name(class_lbls, is_three_party=for_three_party)
    train, test = load_prepared_parties_data(data_dir, class_lbls, load_three_party=for_three_party)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    print("pos lbls train", sum(y_train), len(y_train))
    print("pos lbls test", sum(y_test), len(y_test))

    print("################################ Run Benchmark Models ############################")

    # y_train_1d = np.ravel(y_train)
    # y_test_1d = np.ravel(y_test)
    # benchmark_test(Xa_train, Xa_test, y_train_1d, y_test_1d, "party A")
    # benchmark_test(Xb_train, Xb_test, y_train_1d, y_test_1d, "party B")
    #
    # X_train = np.concatenate([Xa_train, Xb_train], axis=1)
    # X_test = np.concatenate([Xa_test, Xb_test], axis=1)
    # print("X_train shape {0}".format(X_train.shape))
    # print("X_test shape {0}".format(X_test.shape))
    # benchmark_test(X_train, X_test, y_train_1d, y_test_1d, "All")

    print("################################ Build Federated Models ############################")

    n_local_iter_list = [1, 3, 5, 10, 20]

    n_experiments = 1
    show_fig = False if n_experiments > 1 else True
    for i in range(n_experiments):
        print("communication_efficient_experiment: {0}".format(i))
        Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
        Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
        train = [Xa_train, Xb_train, y_train]
        test = [Xa_test, Xb_test, y_test]
        run_experiment(train_data=train, test_data=test, output_directory_name=folder_name, show_fig=show_fig)
