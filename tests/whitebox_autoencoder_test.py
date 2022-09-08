import numpy as np
import tensorflow as tf

from models.autoencoder import Autoencoder


def test_func():

    X = np.array([[4, 2, 3],
                  [6, 5, 1],
                  [3, 4, 1],
                  [1, 2, 3]])
    y = np.array([[1], [-1], [1], [1]])

    overlap_indexes = [1, 2]
    non_overlap_indexes = [0]

    _, D = X.shape

    autoencoder = Autoencoder(0)
    autoencoder.build(D, 400)
    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        autoencoder.set_session(session)
        session.run(init_op)

        grad_w, grad_b = autoencoder.compute_gradients(X)

        print(len(grad_w))
        # print(grad_w)
        print(grad_w[0].shape)
        print(grad_w[1].shape)
        print(np.sum(grad_w[0]))
        print(np.sum(grad_w[1]))
        print(len(grad_b))
        print(grad_b[0].shape)
        print(grad_b[1].shape)

    print("finished")


if __name__ == '__main__':
    test_func()