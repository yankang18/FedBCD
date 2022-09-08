import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np
import pandas as pd

from models.autoencoder import Autoencoder


def getKaggleMNIST():
    # MNIST datasets:
    # column 0 is labels
    # column 1-785 is datasets, with values 0 .. 255
    # total size of CSV: (42000, 1, 28, 28)

    train = pd.read_csv('../../datasets/MINST/train.csv').values.astype(np.float32)
    train = shuffle(train)

    Xtrain = train[:-1000, 1:] / 255
    Ytrain = train[:-1000, 0].astype(np.int32)
    Xtest  = train[-1000:, 1:] / 255
    Ytest  = train[-1000:, 0].astype(np.int32)

    return Xtrain, Ytrain, Xtest, Ytest


def test_single_autoencoder():

    Xtrain, Ytrain, Xtest, Ytest = getKaggleMNIST()
    Xtrain = Xtrain.astype(np.float32)
    Xtest = Xtest.astype(np.float32)

    _, D = Xtrain.shape

    autoencoder = Autoencoder(0)
    autoencoder.build(D, 400)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        autoencoder.set_session(session)
        session.run(init_op)
        autoencoder.fit(Xtrain, epoch=1, show_fig=True)

        i = np.random.choice(len(Xtest))
        x = Xtest[i]
        y = autoencoder.predict([x])
        z = autoencoder.transform([x])
        # print("z shape", z.shape)
        plt.subplot(1, 2, 1)
        plt.imshow(x.reshape(28, 28), cmap='gray')
        plt.title('Original')

        plt.subplot(1, 2, 2)
        plt.imshow(y.reshape(28, 28), cmap='gray')
        plt.title('Reconstructed')
        plt.show()


if __name__ == '__main__':
    # test_single_autoencoder()
    test_func()
