import numpy as np
import tensorflow as tf


def func(x):
    # print(x[0].shape)
    # print(x[1].shape)
    # print(x[1][0].shape)
    # print(x[1][1].shape)
    return tf.sigmoid(tf.matmul(x[0], x[1]))


if __name__ == '__main__':

    # X = tf.placeholder(tf.float32, [None, 3], name="x")
    #
    # W = tf.Variable(tf.random_normal([3, 2]))
    # b = tf.Variable(tf.random_normal([2]))
    #
    # examples = tf.split(X, num_or_size_splits=4)
    # weight_copies = [tf.identity(W) for x in examples]
    # print(type(examples))
    # print(type(weight_copies))
    # # elems = [z for z in zip(examples, weight_copies)]
    # # print(len(elems[0]))
    # output = tf.stack([func(z) for z in zip(examples, weight_copies)])
    # per_example_gradients = tf.gradients(output, weight_copies)
    # #
    # # # grads_W, grads_b = tf.gradients(U, xs=[W, b])
    # datasets = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]])
    # sess = tf.Session()
    # with sess.graph.as_default():
    #     sess.run(tf.global_variables_initializer())
    #
    #     # grads_w_i = sess.run(per_example_gradients, feed_dict={X: datasets})
    #     # print(grads_w_i.shape)
    #
    #     # elems_1 = sess.run(elems, feed_dict={X: datasets})
    #     # print(elems_1[0])
    #
    #     output_1, per_example_gradients_1 = sess.run([output, per_example_gradients], feed_dict={X: datasets})
    #     print(output_1)
    #     print(per_example_gradients_1)

    input = tf.placeholder(tf.float32, [None, 3], name="x")

    W = tf.Variable(tf.random_normal([3, 2]))
    # b = tf.Variable(tf.random_normal([2]))
    output = func([input, W])

    print(output.shape)
    grads = tf.gradients(output, W)
    # # grads_W, grads_b = tf.gradients(U, xs=[W, b])
    # datasets = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9], [1, 2, 3]])
    data = np.array([[1, 2, 3]])
    sess = tf.Session()
    with sess.graph.as_default():
        sess.run(tf.global_variables_initializer())

        gradients_1 = sess.run(grads, feed_dict={input: data})
        # X.map(lambda x: sess.run(grads, feed_dict={input: x}))
        print(gradients_1)


