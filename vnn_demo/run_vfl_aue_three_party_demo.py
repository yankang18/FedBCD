import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from datasets.medical_dataset import load_vertical_medical_data
from models.autoencoder import Autoencoder
from models.learning_rate_decay import sqrt_learning_rate_decay
from store_utils import save_experimental_results
from vfl import VFLHostModel, VFLGuestModel, VerticalMultiplePartyFederatedLearning
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


def run_experiment(train_data, test_data, output_directory_name, n_local, batch_size, learning_rate, epoch, is_parallel,
                   apply_proximal=False, proximal_lbda=0.1, show_fig=True, is_debug=False, verbose=False):
    Xa_train, Xb_train, Xc_train, y_train = train_data
    Xa_test, Xb_test, Xc_test, y_test = test_data

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    autoencoder_A = Autoencoder(1)
    autoencoder_B = Autoencoder(2)
    autoencoder_C = Autoencoder(3)

    autoencoder_A.build(input_dim=Xa_train.shape[1], hidden_dim=50, reg_lbda=0.1, learning_rate=learning_rate)
    autoencoder_B.build(input_dim=Xb_train.shape[1], hidden_dim=30, reg_lbda=0.1, learning_rate=learning_rate)
    autoencoder_C.build(input_dim=Xc_train.shape[1], hidden_dim=30, reg_lbda=0.1, learning_rate=learning_rate)

    (guest_n_local, host_n_local) = n_local
    partyA = VFLGuestModel(local_model=autoencoder_A, n_iter=guest_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                           apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                           verbose=verbose)
    partyB = VFLHostModel(local_model=autoencoder_B, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)
    partyC = VFLHostModel(local_model=autoencoder_C, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)

    using_learning_rate_decay = False
    if using_learning_rate_decay:
        partyA.set_learning_rate_decay_func(sqrt_learning_rate_decay)
        partyB.set_learning_rate_decay_func(sqrt_learning_rate_decay)

    party_B_id = "B"
    party_C_id = "C"
    federatedLearning = VerticalMultiplePartyFederatedLearning(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)
    federatedLearning.add_party(id=party_C_id, party_model=partyC)

    print("################################ Train Federated Models ############################")

    fl_fixture = FederatedLearningFixture(federatedLearning)

    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train, party_C_id: Xc_train}}

    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test, party_C_id: Xc_test}}

    experiment_result = fl_fixture.fit(train_data=train_data,
                                       test_data=test_data,
                                       is_parallel=is_parallel,
                                       epochs=epoch,
                                       batch_size=batch_size,
                                       verbose=verbose,
                                       is_debug=is_debug)
    if output_directory_name is not None:
        task_name = "vfl_aue"
        save_experimental_results(experiment_result, output_directory_name, task_name, show_fig)


if __name__ == '__main__':

    print("################################ Prepare Data ############################")
    # for_three_party = True
    # data_dir = "../data/"
    # # class_lbls = ['person', 'water', 'animal', 'grass', 'buildings']
    # # class_lbls = ['sky', 'clouds', 'person', 'water']
    # class_lbls = ['person', 'animal']
    # folder_name = get_data_folder_name(class_lbls, is_three_party=for_three_party)
    # train, test = load_prepared_parties_data(data_dir, class_lbls, load_three_party=for_three_party)
    # # train, test = load_three_party_data(data_dir, ['person', 'animal'])
    # Xa_train, Xb_train, Xc_train, y_train = train
    # Xa_test, Xb_test, Xc_test, y_test = test

    data_dir = "../data/vertical_medical_data/"
    train, test = load_vertical_medical_data(data_dir)
    Xa_train, Xb_train, Xc_train, y_train = train
    Xa_test, Xb_test, Xc_test, y_test = test

    print("[INFO] pos lbls train", sum(y_train), len(y_train))
    print("[INFO] pos lbls test", sum(y_test), len(y_test))

    print("################################ Run Benchmark Models ############################")

    y_train_1d = np.ravel(y_train)
    y_test_1d = np.ravel(y_test)
    benchmark_test(Xa_train, Xa_test, y_train_1d, y_test_1d, "party A")
    benchmark_test(Xb_train, Xb_test, y_train_1d, y_test_1d, "party B")
    benchmark_test(Xc_train, Xc_test, y_train_1d, y_test_1d, "party C")

    X_train = np.concatenate([Xa_train, Xb_train, Xc_train], axis=1)
    X_test = np.concatenate([Xa_test, Xb_test, Xc_test], axis=1)
    print("X_train shape {0}".format(X_train.shape))
    print("X_test shape {0}".format(X_test.shape))
    benchmark_test(X_train, X_test, y_train_1d, y_test_1d, "All")

    print("################################ Build Federated Models ############################")
    output_dir_name = "/vnn_demo/result/auc_three_party/"

    lr = 0.001
    epoch = 10
    batch_size = 256

    apply_proximal = True
    proximal_lbda = 2.0
    n_local = (2, 1)
    # n_local = (1, 1)

    is_parallel = True
    n_experiments = 1
    show_fig = False if n_experiments > 1 else True
    is_debug = False
    verbose = False
    for i in range(n_experiments):
        print("[INFO] communication_efficient_experiment: {0}".format(i))
        # Xa_train, Xb_train, Xc_train, y_train = shuffle(Xa_train, Xb_train, Xc_train, y_train)
        # Xa_test, Xb_test, Xc_test, y_test = shuffle(Xa_test, Xb_test, Xc_test, y_test)
        train = [Xa_train, Xb_train, Xc_train, y_train]
        test = [Xa_test, Xb_test, Xc_test, y_test]
        run_experiment(train_data=train,
                       test_data=test,
                       output_directory_name=output_dir_name,
                       n_local=n_local,
                       batch_size=batch_size,
                       learning_rate=lr,
                       epoch=epoch,
                       is_parallel=is_parallel,
                       apply_proximal=apply_proximal, proximal_lbda=proximal_lbda,
                       show_fig=show_fig,
                       is_debug=is_debug,
                       verbose=verbose)
