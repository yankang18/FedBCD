import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from models.base_model import BaseModel


class Autoencoder(BaseModel):

    def __init__(self, an_id):
        super(Autoencoder, self).__init__()
        self.id = str(an_id)

    def get_ID(self):
        return self.id

    def set_session(self, sess):
        self.sess = sess

    def build(self, input_dim, hidden_dim, learning_rate=1e-2, reg_lbda=0.01, proximal_lbda=1.0):
        """
        :param input_dim: the dimension of input
        :param hidden_dim: the dimension of hidden layer
        :param learning_rate: the learning rate
        :param reg_lbda: the regularization factor
        :param proximal_lbda: the lambda parameter for proximal loss
        :return:
        """

        self.lr = learning_rate
        self.lbda = reg_lbda
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.proximal_lbda = proximal_lbda

        self._add_input_placeholder()
        self._add_encoder_decoder_ops()
        self._add_forward_ops()
        self._add_representation_training_ops()
        self._add_e2e_training_ops()
        self._add_encrypt_grad_update_ops()

    def _add_input_placeholder(self):
        self.X_in = tf.placeholder(tf.float64, shape=(None, self.input_dim))
        self.batch_size = tf.placeholder(tf.int32)

        # self.apply_proximal = tf.placeholder(tf.bool)
        # self.proximal_Wh = tf.placeholder(tf.float64, shape=(self.input_dim, self.hidden_dim))

        # if self.apply_proximal:
        self.proximal_Wh = tf.placeholder(tf.float64, shape=(self.input_dim, self.hidden_dim))

    def _add_encoder_decoder_ops(self):
        self.encoder_vars_scope = self.id + "_encoder_vars"
        with tf.variable_scope(self.encoder_vars_scope):
            stddev = np.sqrt(2 / (self.input_dim + self.hidden_dim))
            # stddev = 1.0
            self.Wh = tf.get_variable("weights", initializer=tf.random_normal((self.input_dim, self.hidden_dim),
                                                                              stddev=stddev, dtype=tf.float64))
            self.bh = tf.get_variable("bias", initializer=np.zeros(self.hidden_dim).astype(np.float64))

        self.decoder_vars_scope = self.id + "_decoder_vars"
        with tf.variable_scope(self.decoder_vars_scope):
            stddev = np.sqrt(2 / (self.hidden_dim + self.input_dim))
            # stddev = 1.0
            self.Wo = tf.get_variable("weights", initializer=tf.random_normal((self.hidden_dim, self.input_dim),
                                                                              stddev=stddev, dtype=tf.float64))
            self.bo = tf.get_variable("bias", initializer=np.zeros(self.input_dim).astype(np.float64))

    def _add_forward_ops(self):
        self.Z = self._forward_hidden(self.X_in)
        self.logits = self._forward_logits(self.X_in)
        self.X_hat = self._forward_output(self.X_in)

    def _add_representation_training_ops(self):
        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.encoder_vars_scope)
        l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float64)) for v in vars_to_train])
        self.init_grad = tf.placeholder(tf.float64, shape=(None, self.hidden_dim))
        self.repr_training_reg_loss = self.lbda * l2_loss / tf.cast(self.batch_size, tf.float64)

        # def f1(): return self.Z + tf.nn.l2_loss(self.Wh - self.proximal_Wh)
        #
        # def f2(): return self.Z
        #
        # loss = tf.cond(self.apply_proximal, f1, f2)

        # loss = self.Z
        # if self.apply_proximal:
        #     loss = loss + self.proximal_lbda * tf.nn.l2_loss(self.Wh - self.proximal_Wh)

        loss1 = self.Z

        # print("## self.proximal_lbda in autoencoder: {0}".format(self.proximal_lbda))
        loss2 = self.Z + self.proximal_lbda * tf.nn.l2_loss(self.Wh - self.proximal_Wh)

        self.train_op_1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=loss1, var_list=vars_to_train,
                                                                                 grad_loss=self.init_grad)
        self.train_op_2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=loss2, var_list=vars_to_train,
                                                                                 grad_loss=self.init_grad)

        # self.train_op_1 = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(loss=loss1, var_list=vars_to_train,
        #                                                                             grad_loss=self.init_grad)
        # self.train_op_2 = tf.train.RMSPropOptimizer(learning_rate=self.lr).minimize(loss=loss2, var_list=vars_to_train,
        #                                                                             grad_loss=self.init_grad)

        # self.enc_reg_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(loss=self.repr_training_reg_loss,
        #                                                                          var_list=vars_to_train)

        # self.optimizer = Optimizer(learning_rate=learning_rate, opt_method_name=optimizer)
        # W_reg_grad = self.lbda * self.Wh / tf.cast(self.batch_size, tf.float64)
        W_reg_grad = self.lbda * self.Wh
        new_Wh = self.Wh - self.lr * W_reg_grad
        self.enc_reg_op = tf.assign(self.Wh, new_Wh)

    def _add_e2e_training_ops(self):
        self.loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits, labels=self.X_in))
        self.e2e_train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss)

    def _add_encrypt_grad_update_ops(self):
        # self.grads_W, self.grads_b = tf.gradients(self.Z, xs=[self.Wh, self.bh])
        self.Z_grads = tf.gradients(self.Z, xs=[self.Wh, self.bh])

        self.grads_W_new = tf.placeholder(tf.float64, shape=[self.input_dim, self.hidden_dim])
        self.grads_b_new = tf.placeholder(tf.float64, shape=[self.hidden_dim])
        self.new_Wh = self.Wh.assign(self.Wh - self.lr * self.grads_W_new)
        self.new_bh = self.bh.assign(self.bh - self.lr * self.grads_b_new)

    def _forward_hidden(self, X):
        return tf.sigmoid(tf.matmul(X, self.Wh) + self.bh)

    def _forward_logits(self, X):
        Z = self._forward_hidden(X)
        return tf.matmul(Z, self.Wo) + self.bo

    def _forward_output(self, X):
        return tf.sigmoid(self._forward_logits(X))

    def transform(self, X):
        return self.sess.run(self.Z, feed_dict={self.X_in: X})

    def get_proximal_model(self):
        return self.sess.run(self.Wh)

    def compute_gradients(self, X):
        grads_W_collector = []
        grads_b_collector = []
        for i in range(len(X)):
            grads_w_i, grads_b_i = self.sess.run(self.Z_grads, feed_dict={self.X_in: np.expand_dims(X[i], axis=0)})
            # print(grads_w_i)
            # print(grads_w_i.shape)
            grads_W_collector.append(grads_w_i)
            grads_b_collector.append(grads_b_i)
        # print("finished")
        return [np.array(grads_W_collector), np.array(grads_b_collector)]

    def apply_gradients(self, gradients):
        grads_W = gradients[0]
        grads_b = gradients[1]
        _, _ = self.sess.run([self.new_Wh, self.new_bh],
                             feed_dict={self.grads_W_new: grads_W, self.grads_b_new: grads_b})

    def backpropogate(self, X, y, in_grad, apply_proximal=False, proximal=None):
        if apply_proximal:
            if proximal is None:
                raise Exception("proximal should be provided but is None")
            # print("## autoencoder apply proximal")

            _, _, repr_training_reg_loss = self.sess.run(
                [self.train_op_2, self.enc_reg_op, self.repr_training_reg_loss],
                feed_dict={self.X_in: X,
                           self.init_grad: in_grad,
                           self.proximal_Wh: proximal,
                           self.batch_size: X.shape[0]})
        else:
            # print("## autoencoder does not apply proximal")
            _, _, repr_training_reg_loss = self.sess.run(
                [self.train_op_1, self.enc_reg_op, self.repr_training_reg_loss],
                feed_dict={self.X_in: X,
                           self.init_grad: in_grad,
                           self.batch_size: X.shape[0]})
        return repr_training_reg_loss

    def predict(self, X):
        return self.sess.run(self.X_hat, feed_dict={self.X_in: X})

    def get_features_dim(self):
        return self.hidden_dim

    def fit(self, X, batch_size=32, epoch=1, show_fig=False):

        N, D = X.shape

        n_batches = N // batch_size
        costs = []
        for ep in range(epoch):
            for i in range(n_batches):
                batch = X[i * batch_size: i * batch_size + batch_size]
                _, c = self.sess.run([self.e2e_train_op, self.loss], feed_dict={self.X_in: batch})

                if i % 10 == 0:
                    print(i, "/", n_batches, "cost:", c)
                costs.append(c)

        if show_fig:
            plt.plot(costs)
            plt.show()
