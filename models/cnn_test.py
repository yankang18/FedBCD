
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import shuffle

from cnn import MaxPoolingLayer, ReluActivationLayer, SimpleCNN, ConvolutionLayer


def getKaggleMNIST():
    # MNIST datasets:
    # column 0 is labels
    # column 1-785 is datasets, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv('../datasets/MINST/train.csv')
    train = train.as_matrix()
    train = shuffle(train)

    Xtrain = train[:1000, 1:] / 255.0
    Ytrain = train[:1000, 0].astype(np.int32)
    Xtest = train[-1000:, 1:] / 255.0
    Ytest = train[-1000:, 0].astype(np.int32)

    return Xtrain, Ytrain, Xtest, Ytest


if __name__ == '__main__':

    x_train, y_train, x_test, y_test = getKaggleMNIST()
    print(x_train.shape)
    print(x_test.shape)
    print(y_train.shape)
    print(y_test.shape)

    #
    # Xtrain = Xtrain.reshape(Xtrain.shape[0], 28, 28, 1)
    # Xtest = Xtest.reshape(Xtest.shape[0], 28, 28, 1)

    # mnist = tf.keras.datasets.mnist
    # (x_train, y_train), (x_test, y_test) = mnist.load_data()
    # (x_train, y_train), (x_test, y_test) = (x_train[:2000], y_train[:2000]), (x_test[:2000], y_test[:2000])

    enc = OneHotEncoder()
    y_train = enc.fit_transform(y_train.reshape(-1, 1)).toarray()
    y_test = enc.fit_transform(y_test.reshape(-1, 1)).toarray()

    x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
    x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

    # x_train, x_test = x_train / 255.0, x_test / 255.0

    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)

    tf.reset_default_graph()

    # bn_layer = BatchNormalizationLayer()
    # input_shape = x_train.shape
    # inputs = tf.placeholder(tf.float32, [None, input_shape[0], input_shape[1], input_shape[2]],
    #                         name="inputs")
    # bn_layer.build(inputs, "1", "2", is_training=True)
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     sess.run(init)
    #     bn_layer.get_layer_meta(sess)

    layer1 = ConvolutionLayer(filter_size=3, n_out_channels=32, stride_size=1, padding_mode="SAME")
    layer3 = ReluActivationLayer()
    layer4 = MaxPoolingLayer(filter_size=2, stride_size=1, padding_mode="VALID")
    layer5 = ConvolutionLayer(filter_size=3, n_out_channels=32, stride_size=1, padding_mode="SAME")
    layer7 = MaxPoolingLayer(filter_size=2, stride_size=1, padding_mode="VALID")
    layer8 = ReluActivationLayer()

    simpleCNN = SimpleCNN(1)
    simpleCNN.add_layer(layer1)
    simpleCNN.add_layer(layer3)
    simpleCNN.add_layer(layer4)
    simpleCNN.add_layer(layer5)
    simpleCNN.add_layer(layer7)
    simpleCNN.add_layer(layer8)
    simpleCNN.build(input_shape=(28, 28, 1), representation_dim=256, class_num=10, lr=0.01)

    show_fig = True
    batch_size = 32
    N = x_train.shape[0]
    n_batches = N // batch_size
    epochs = 2

    print("n_batches", n_batches)
    print("epochs", epochs)
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

        for ep in range(epochs):
            for i in range(n_batches):
                global_step += 1
                X = x_train[i * batch_size: i * batch_size + batch_size]
                Y = y_train[i * batch_size: i * batch_size + batch_size]
                loss = simpleCNN.train(X, Y)
                # train_writer.add_summary(summary, global_step)

                print(ep, "loss:", loss)
                loss_list.append(loss)

                # tf_summary = sess.run(merged)
                # train_writer.add_summary(tf_summary, global_step)

                if i % 5 == 0:
                    # y_preds = federatedLearning.predict(X_test_b_left, X_test_b_right)
                    acc = simpleCNN.evaluate(x_test, y_test)
                    acc_list.append(acc)
                    print("ep", ep, "batch", i, "loss:", loss, "acc", acc)
                    metrics = {"acc": acc}
                    # earlyStoppingCheckPoint.on_iteration_end(ep, i, metrics)

                # if simpleCNN.is_stop_training():
                #     break

            # if simpleCNN.is_stop_training():
            #     break

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

        model_meta = simpleCNN.get_model_parameters()
        # print("model_meta \n", model_meta)

    print("---------------------------")
    newSimpleCNN = SimpleCNN()
    tf.reset_default_graph()
    simpleCNN.restore_model(model_meta)
    init = tf.global_variables_initializer()
    sess = tf.Session()
    simpleCNN.set_session(sess)
    sess.run(init)

    acc = simpleCNN.evaluate(x_test, y_test)
    print("final acc:", acc)
    sess.close()

    # epochs = 2
    # batch_size = 128
    # test_valid_size = 256
    #
    # merged = tf.summary.merge_all()
    # init = tf.global_variables_initializer()
    # with tf.Session() as sess:
    #     simpleCNN.set_session(sess)
    #     sess.run(init)
    #
    #     log_dir = '../tensorboard'
    #     train_writer = tf.summary.FileWriter(log_dir + '/train', sess.graph)
    #     test_writer = tf.summary.FileWriter(log_dir + '/test')
    #
    #     global_step = 0
    #     for epoch in range(epochs):
    #         for batch in range(mnist.fit.num_examples // batch_size + 1):
    #             global_step += 1
    #             batch_x, batch_y = mnist.fit.next_batch(batch_size)
    #             loss, summary = simpleCNN.train(batch_x, batch_y)
    #             train_writer.add_summary(summary, global_step)
    #
    #             if batch % 20 == 0:
    #                 acc, summary = simpleCNN.predict(mnist.validation.images[:test_valid_size], mnist.validation.labels[:test_valid_size])
    #                 test_writer.add_summary(summary, global_step)
    #                 print("batch " + str(batch) + ", Minibatch Loss= " +
    #                       "{:.4f}".format(loss) + ", Training Accuracy= " +
    #                       "{:.3f}".format(acc))
    #
    #     test_acc, summary = simpleCNN.predict(mnist.validation.images[:test_valid_size],  mnist.validation.labels[:test_valid_size])
    #     test_writer.add_summary(summary, global_step+1)
    #     print('Testing Accuracy: {}'.format(test_acc))
    #
    #     train_writer.close()
    #     test_writer.close()
