import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from models.cnn import ConvolutionLayer, ReluActivationLayer, SimpleCNN
from models.learning_rate_decay import sqrt_learning_rate_decay
from models.regularization import EarlyStoppingCheckPoint
from vfl import VFLHostModel, VFLGuestModel, VerticalMultiplePartyLogisticRegressionFederatedLearning
from vnn_demo.vfl_fixture import FederatedLearningFixture
from vnn_demo.vfl_fixture import save_vfl_experiment_result

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
home_dir = os.path.split(os.path.realpath(__file__))[0]
sys.path.append(os.path.join(home_dir))


def getKaggleMINST():
    # MNIST datasets:
    # column 0 is labels
    # column 1-785 is datasets, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv('../../data/MINST/train.csv')
    # test = pd.read_csv('../../data/MINST/test.csv')
    train = train.as_matrix()
    # test = test.as_matrix()as_matrix
    print("train shape:{0}".format(train.shape))
    # print("test shape:{0}".formatßt(test.shape))
    train = shuffle(train)

    Xtrain = train[:-7500, 1:] / 255
    Ytrain = train[:-7500, 0].astype(np.int32)
    Xtest  = train[-7500:, 1:] / 255
    Ytest  = train[-7500:, 0].astype(np.int32)

    return Xtrain, Ytrain, Xtest, Ytest


def get_binary_labels(X, Y, binary=(9, 8)):
    """
    Convert two specified labels to 0 and 1
    :param X:
    :param Y:
    :param binary:
    :return:
    """
    X_b = []
    Y_b = []
    for index in range(X.shape[0]):
        lbl = Y[index]
        if lbl in binary:
            X_b.append(X[index])
            Y_b.append(0 if lbl == binary[0] else 1)
    return np.array(X_b), np.array(Y_b)


def split_in_half(imgs):
    """
    split input images in half vertically
    :param imgs: input images
    :return: left part of images and right part of images
    """
    left, right = [], []
    for index in range(len(imgs)):
        img = imgs[index]
        left.append(img[:, 0:14, :])
        right.append(img[:, 14:, :])
    return np.array(left), np.array(right)


def benchmark_test(X_train_left, X_train_right, Y_train, X_test_left, X_test_right, Y_test):

    # convert labels from [-1, 1] to [0, 1]
    Y_test  = (Y_test + 1) / 2
    Y_train = (Y_train + 1) / 2

    enc = OneHotEncoder()
    Y_test = enc.fit_transform(Y_test).toarray()
    Y_train = enc.fit_transform(Y_train).toarray()

    print("Y_test shape", Y_test.shape)
    print("Y_test", type(Y_test))
    # print("Y_test", Y_test)

    tf.reset_default_graph()

    conv_layer_1_for_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_2_A = ReluActivationLayer()
    conv_layer_3_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_4_A = ReluActivationLayer()

    simpleCNN = SimpleCNN(1)
    simpleCNN.add_layer(conv_layer_1_for_A)
    simpleCNN.add_layer(activation_layer_2_A)
    simpleCNN.add_layer(conv_layer_3_A)
    simpleCNN.add_layer(activation_layer_4_A)
    simpleCNN.build(input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=0.001)

    show_fig = True
    batch_size = 256
    N, D = Ytrain_b.shape
    residual = N % batch_size
    if residual == 0:
        n_batches = N // batch_size
    else:
        n_batches = N // batch_size + 1

    epochs = 5
    earlyStoppingCheckPoint = EarlyStoppingCheckPoint("acc", 100)
    earlyStoppingCheckPoint.set_model(simpleCNN)

    # merged = tf.summary.merge_all()
    init = tf.global_variables_initializer()
    with tf.Session() as sess:
        simpleCNN.set_session(sess)

        # log_dir = '../tensorboard/fl_cnn'
        # train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
        # train_writer_B = tf.summary.FileWriter(log_dir + '/train/model_B')

        sess.run(init)
        loss_list = []
        acc_list = []
        global_step = 0

        earlyStoppingCheckPoint.on_train_begin()
        for ep in range(epochs):
            for i in range(n_batches):
                global_step += 1
                # X_left = X_train_left[i * batch_size: i * batch_size + batch_size]
                X = X_train_right[i * batch_size: i * batch_size + batch_size]
                Y = Y_train[i * batch_size: i * batch_size + batch_size]
                loss, summary = simpleCNN.train(X, Y)
                # train_writer.add_summary(summary, global_step)

                # tf_summary = sess.run(merged)
                # train_writer.add_summary(tf_summary, global_step)

                if i % 1 == 0:
                    loss_list.append(loss)
                    # y_preds = federatedLearning.predict(X_test_b_left, X_test_b_right)
                    acc, summary = simpleCNN.evaluate(X_test_right, Y_test)
                    acc_list.append(acc)
                    print(ep, "batch", i, "loss:", loss, "acc", acc)

                    metrics = {"acc": acc}
                    earlyStoppingCheckPoint.on_iteration_end(ep, i, metrics)

                if simpleCNN.is_stop_training():
                    break

            if simpleCNN.is_stop_training():
                break

        if show_fig:
            plt.subplot(121)
            plt.plot(loss_list)
            plt.xlabel("loss")
            plt.subplot(122)
            plt.plot(acc_list)
            plt.xlabel("acc")
            plt.show()

        print("loss_list:", loss_list)
        print("acc_list:", acc_list)


def compute_accuracy(y_targets, y_preds):

    corr_count = 0
    total_count = len(y_preds)
    print("# of labels:", total_count)
    for y_p, y_t in zip(y_preds, y_targets):
        if (y_p <= 0.5 and y_t == -1) or (y_p > 0.5 and y_t == 1):
            corr_count += 1

    acc = float(corr_count) / float(total_count)
    return acc


def run_experiment(train_data, test_data, output_directory_name, n_local, batch_size, learning_rate, is_parallel,
                   epochs=5, apply_proximal=False, proximal_lbda=0.1, show_fig=True):

    print("hyper-parameters:")
    print("# of epochs: {0}".format(epochs))
    print("# of local iterations: {0}".format(n_local))
    print("batch size: {0}".format(batch_size))
    print("learning rate: {0}".format(learning_rate))
    print("show figure: {0}".format(show_fig))
    print("is async {0}".format(is_parallel))
    print("is apply_proximal {0}".format(apply_proximal))
    print("proximal_lbda {0}".format(proximal_lbda))

    X_train_b_left, X_train_b_right, Ytrain_b = train_data
    X_test_b_left, X_test_b_right, Ytest_b = test_data

    tf.reset_default_graph()

    conv_layer_1_for_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_2_A = ReluActivationLayer()
    conv_layer_3_A = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_4_A = ReluActivationLayer()

    simpleCNN_A = SimpleCNN(1)
    simpleCNN_A.add_layer(conv_layer_1_for_A)
    simpleCNN_A.add_layer(activation_layer_2_A)
    simpleCNN_A.add_layer(conv_layer_3_A)
    simpleCNN_A.add_layer(activation_layer_4_A)
    simpleCNN_A.build(input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=learning_rate,
                      proximal_lbda=proximal_lbda)

    conv_layer_1_for_B = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_2_B = ReluActivationLayer()
    conv_layer_3_B = ConvolutionLayer(filter_size=2, n_out_channels=64, stride_size=1, padding_mode="SAME")
    activation_layer_4_B = ReluActivationLayer()

    simpleCNN_B = SimpleCNN(2)
    simpleCNN_B.add_layer(conv_layer_1_for_B)
    simpleCNN_B.add_layer(activation_layer_2_B)
    simpleCNN_B.add_layer(conv_layer_3_B)
    simpleCNN_B.add_layer(activation_layer_4_B)
    simpleCNN_B.build(input_shape=(28, 14, 1), representation_dim=256, class_num=2, lr=learning_rate,
                      proximal_lbda=proximal_lbda)

    partyA = VFLGuestModel(local_model=simpleCNN_A, n_iter=n_local, learning_rate=learning_rate, lbda=0.01,
                           is_ave=False, optimizer="adam", apply_proximal=apply_proximal, proximal_lbda=proximal_lbda)
    partyB = VFLHostModel(local_model=simpleCNN_B, n_iter=n_local, learning_rate=learning_rate, lbda=0.01,
                          optimizer="adam", apply_proximal=apply_proximal, proximal_lbda=proximal_lbda)

    using_learning_rate_decay = False
    if using_learning_rate_decay:
        partyA.set_learning_rate_decay_func(sqrt_learning_rate_decay)
        partyB.set_learning_rate_decay_func(sqrt_learning_rate_decay)

    party_B_id = "B"
    federated_learning = VerticalMultiplePartyLogisticRegressionFederatedLearning(partyA)
    federated_learning.add_party(id=party_B_id, party_model=partyB)

    print("################################ Train Federated Models ############################")

    train_data = {federated_learning.get_main_party_id(): {"X": X_train_b_left, "Y": Ytrain_b},
                  "party_list": {party_B_id: X_train_b_right}}

    test_data = {federated_learning.get_main_party_id(): {"X": X_test_b_left, "Y": Ytest_b},
                 "party_list": {party_B_id: X_test_b_right}}

    fl_fixture = FederatedLearningFixture(federated_learning)
    experiment_result = fl_fixture.fit(train_data=train_data,
                                       test_data=test_data,
                                       is_parallel=is_parallel,
                                       epochs=epoch,
                                       batch_size=batch_size,
                                       show_fig=show_fig)

    output_directory_full_name = "/result/" + output_directory_name
    task_name = "vfl_cnn"
    save_vfl_experiment_result(output_directory=output_directory_full_name,
                               task_name=task_name,
                               experiment_result=experiment_result)


if __name__ == '__main__':

    # np.random.seed(2)
    Xtrain, Ytrain, Xtest, Ytest = getKaggleMINST()
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1)
    Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)

    print("Xtrain", Xtrain.shape)
    print("Xtest", Xtest.shape)

    # choose two labels from the 10 digit labels and convert the two labels to 0 and 1
    Xtrain_b, Ytrain_b = get_binary_labels(Xtrain, Ytrain, [3, 8])
    Xtest_b,  Ytest_b = get_binary_labels(Xtest, Ytest, [3, 8])

    # convert labels in the form of vector to matrix
    Ytrain_b = Ytrain_b.reshape(Ytrain_b.shape[0], 1)
    Ytest_b = Ytest_b.reshape(Ytest_b.shape[0], 1)

    print("Xtrain_b", Xtrain_b.shape)
    print("Xtest_b", Xtest_b.shape)
    print("Ytrain_b", Ytrain_b.shape)
    print("Ytest_b", Ytest_b.shape)

    # split each image in half vertically
    X_train_b_left, X_train_b_right = split_in_half(Xtrain_b)
    X_test_b_left, X_test_b_right = split_in_half(Xtest_b)

    print("X_train_b_left", X_train_b_left.shape)
    print("X_train_b_right", X_train_b_right.shape)
    print("X_test_b_left", X_test_b_left.shape)
    print("X_test_b_right", X_test_b_right.shape)

    print("################################ Build benchmark Models ############################")

    # benchmark_test(X_train_b_left, X_train_b_right, Ytrain_b, X_test_b_left, X_test_b_right, Ytest_b)

    print("################################ Build Federated Models ############################")

    '''
    Typically, a grid search involves picking values approximately on a logarithmic scale, e.g., a
    learning rate taken within the set {.1, .01, 10−3, 10−4, 10−5}
    '''
    # output_dir_name = "cnn_two_party_batch_512"
    output_dir_name = "cnn_lr_decay"
    n_experiments = 1
    apply_proximal = False
    proximal_lbda = 0.0
    batch_size = 256
    epoch = 5
    # try different local iteration for both parties,
    # e.g. [1, 3] means try 1 local iteration for both parties and then try 3 local iterations for both parties
    # is_async_list = [False, True]
    # for local 5
    # lr_list = [1e-05, 2e-05, 5e-05, 1e-04, 2e-04]
    # for local 3
    lr_list = [1e-04]
    is_parallel_list = [True]
    n_local_iter_list = [1]
    for is_parallel in is_parallel_list:
        for lr in lr_list:
            for n_local in n_local_iter_list:
                show_fig = False if n_experiments > 1 else True
                for i in range(n_experiments):
                    print("communication_efficient_experiment: {0} for is_asy {1}, lr {2}, n_local {3}".format(i, is_parallel, lr, n_local))
                    X_train_b_left, X_train_b_right, Ytrain_b = shuffle(X_train_b_left, X_train_b_right, Ytrain_b)
                    X_test_b_left, X_test_b_right, Ytest_b = shuffle(X_test_b_left, X_test_b_right, Ytest_b)
                    train = [X_train_b_left, X_train_b_right, Ytrain_b]
                    test = [X_test_b_left, X_test_b_right, Ytest_b]
                    run_experiment(train_data=train, test_data=test, output_directory_name=output_dir_name,
                                   n_local=n_local, batch_size=batch_size, learning_rate=lr, is_parallel=is_parallel,
                                   apply_proximal=apply_proximal, proximal_lbda=proximal_lbda, epochs=epoch,
                                   show_fig=show_fig)

