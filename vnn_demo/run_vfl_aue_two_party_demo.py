import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

from config import data_dir
from datasets.nus_wide_dataset import load_prepared_parties_data
from models.autoencoder import Autoencoder
from models.learning_rate_decay import sqrt_learning_rate_decay
from store_utils import save_experimental_results
from vnn_demo.vfl import VFLHostModel, VFLGuestModel, VerticalMultiplePartyFederatedLearning
from vnn_demo.vfl_learner import VerticalFederatedLearningLearner


def benchmark_test(X_train, X_test, y_train, y_test, party_name=""):
    print("------ {0} ------".format(party_name))
    lr = LogisticRegression(C=2.0, solver="lbfgs")
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
                   apply_proximal=False, proximal_lbda=0.1, is_debug=False, verbose=False, show_fig=True):
    print("hyper-parameters:")
    print("# of local iterations: {0}".format(n_local))
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))
    print("is async {0}".format(is_parallel))
    print("apply proximal {0}".format(apply_proximal))
    print("proximal lbda {0}".format(proximal_lbda))
    print("show figure: {0}".format(show_fig))
    print("is debug: {0}".format(is_debug))
    print("verbose: {0}".format(verbose))

    Xa_train, Xb_train, y_train = train_data
    Xa_test, Xb_test, y_test = test_data

    print("################################ Build Federated Models ############################")

    tf.reset_default_graph()

    autoencoder_A = Autoencoder(1)
    autoencoder_B = Autoencoder(2)

    autoencoder_A.build(input_dim=Xa_train.shape[1], hidden_dim=50, reg_lbda=0.1, learning_rate=learning_rate,
                        proximal_lbda=proximal_lbda)
    autoencoder_B.build(input_dim=Xb_train.shape[1], hidden_dim=30, reg_lbda=0.1, learning_rate=learning_rate,
                        proximal_lbda=proximal_lbda)

    (guest_n_local, host_n_local) = n_local
    partyA = VFLGuestModel(local_model=autoencoder_A, n_iter=guest_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                           apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                           verbose=verbose)
    partyB = VFLHostModel(local_model=autoencoder_B, n_iter=host_n_local, learning_rate=learning_rate, reg_lbda=0.01,
                          apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, is_debug=is_debug,
                          verbose=verbose)

    using_learning_rate_decay = False
    if using_learning_rate_decay:
        partyA.set_learning_rate_decay_func(sqrt_learning_rate_decay)
        partyB.set_learning_rate_decay_func(sqrt_learning_rate_decay)

    party_B_id = "B"
    federatedLearning = VerticalMultiplePartyFederatedLearning(partyA)
    federatedLearning.add_party(id=party_B_id, party_model=partyB)

    print("################################ Train Federated Models ############################")

    fl_learner = VerticalFederatedLearningLearner(federatedLearning)

    train_data = {federatedLearning.get_main_party_id(): {"X": Xa_train, "Y": y_train},
                  "party_list": {party_B_id: Xb_train}}

    test_data = {federatedLearning.get_main_party_id(): {"X": Xa_test, "Y": y_test},
                 "party_list": {party_B_id: Xb_test}}

    experiment_result = fl_learner.fit(train_data=train_data,
                                       test_data=test_data,
                                       is_parallel=is_parallel,
                                       epochs=epoch,
                                       batch_size=batch_size,
                                       is_debug=is_debug,
                                       verbose=verbose)

    if output_directory_name is not None:
        task_name = "vfl_aue"
        save_experimental_results(experiment_result, output_directory_name, task_name, show_fig)


if __name__ == '__main__':

    print("################################ Prepare Data ############################")
    for_three_party = False
    # class_lbls = ['person', 'water', 'animal', 'grass', 'buildings']
    # class_lbls = ['sky', 'clouds', 'person', 'water']
    class_lbls = ['person', 'animal']
    # folder_name = get_data_folder_name(class_lbls, is_three_party=for_three_party)
    folder_name = None
    train, test = load_prepared_parties_data(data_dir, class_lbls, load_three_party=for_three_party)
    Xa_train, Xb_train, y_train = train
    Xa_test, Xb_test, y_test = test

    print("[INFO] pos lbls train", sum(y_train), len(y_train))
    print("[INFO] pos lbls test", sum(y_test), len(y_test))

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
    is_debug = False
    verbose = False
    show_fig = False

    output_dir_name = "/vnn_demo/result/auc_two_party/"
    n_experiments = 1
    apply_proximal = True
    proximal_lbda = 3.0
    batch_size = 256
    epoch = 10

    is_parallel_list = [True]

    # lr_list = [0.003]
    # n_local_iter_list = [(1, 1)]  # [(number_local_iterations_guest, number_local_iterations_host)]

    lr_list = [0.001]
    n_local_iter_list = [(6, 6)]  # [(number_local_iterations_guest, number_local_iterations_host)]

    for is_parallel in is_parallel_list:
        for lr in lr_list:
            for n_local in n_local_iter_list:
                if show_fig: show_fig = False if n_experiments > 1 else True
                for i in range(n_experiments):
                    print("[INFO] communication_efficient_experiment: {0} with is_async: {1}, lr: {2}, n_local: {3}"
                          .format(i, is_parallel, lr, n_local))
                    # Xa_train, Xb_train, y_train = shuffle(Xa_train, Xb_train, y_train)
                    # Xa_test, Xb_test, y_test = shuffle(Xa_test, Xb_test, y_test)
                    train = [Xa_train, Xb_train, y_train]
                    test = [Xa_test, Xb_test, y_test]
                    run_experiment(train_data=train, test_data=test, output_directory_name=output_dir_name,
                                   n_local=n_local, batch_size=batch_size, learning_rate=lr, epoch=epoch,
                                   is_parallel=is_parallel, apply_proximal=apply_proximal, proximal_lbda=proximal_lbda,
                                   is_debug=is_debug, show_fig=show_fig, verbose=verbose)
