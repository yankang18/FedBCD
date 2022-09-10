from typing import List

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import init_ops

from models.base_model import BaseModel


class Layer(object):

    def build(self, inputs, owner_model_id, layer_id, is_training):
        pass

    def restore_layer(self, model_parameters, inputs, is_training):
        pass

    def get_layer_meta(self, session):
        return None, None

    def get_layer_proximal_parameters(self, session):
        return None

    def compute_difference_to_proximal(self):
        return None

    def can_apply_proximal(self):
        return False

    def get_proximal_placeholder(self):
        return None

    @staticmethod
    def generate_layer_identifier(owner_model_id, layer_id):
        return "model_" + owner_model_id + "_layer_" + layer_id


class ActivationLayer(Layer):

    def __init__(self, activation_fn, activation_type):
        self.layer_type = "activation"
        self.owner_model_id = None
        self.layer_id = None
        self.activation_fn = activation_fn
        self.activation_type = activation_type

    def build(self, inputs, owner_model_id, layer_id, is_training):
        self.owner_model_id = str(owner_model_id)
        self.layer_id = str(layer_id)
        return self.activation_fn(inputs)

    def restore_layer(self, model_parameters, inputs, is_training):
        owner_model_id = model_parameters["owner_model_id"]
        layer_id = model_parameters["layer_id"]
        return self.build(inputs, owner_model_id, layer_id, is_training)

    def get_layer_meta(self, session):
        layer_model_meta = {"layer_type": self.layer_type,
                            "activation_type": self.activation_type,
                            "owner_model_id": self.owner_model_id,
                            "layer_id": self.layer_id}
        identifier = self.generate_layer_identifier(self.owner_model_id, self.layer_id)
        return identifier, layer_model_meta


class ReluActivationLayer(ActivationLayer):
    def __init__(self):
        super(ReluActivationLayer, self).__init__(activation_fn=tf.nn.relu, activation_type="relu")


class BatchNormalizationLayer(Layer):

    def __init__(self):
        self.layer_type = "batch_normalization"
        self.owner_model_id = None
        self.layer_id = None
        self.conv_layer_vars_scope = None
        self.bn_full_name = None
        self.bn_layer = None

        self.beta_initializer = None
        self.gamma_initializer = None
        self.moving_mean_initializer = None
        self.moving_variance_initializer = None

    def build(self, inputs, owner_model_id, layer_id, is_training):
        self.owner_model_id = str(owner_model_id)
        self.layer_id = str(layer_id)
        self._set_layer_variable_initializer()
        self._create_layer_variables_scope(self.owner_model_id)
        return self._build_layer(inputs, is_training)

    def _build_layer(self, inputs, is_training):
        layer_name = "layer_" + self.layer_id
        bn_name = layer_name + "_bn_ops"
        self.bn_full_name = self.conv_layer_vars_scope + "/" + layer_name + "/" + bn_name + "/"
        with tf.variable_scope(self.conv_layer_vars_scope):
            with tf.variable_scope(layer_name):
                self.bn_layer = tf.layers.batch_normalization(inputs,
                                                              beta_initializer=self.beta_initializer,
                                                              gamma_initializer=self.gamma_initializer,
                                                              moving_mean_initializer=self.moving_mean_initializer,
                                                              moving_variance_initializer=self.moving_variance_initializer,
                                                              training=is_training,
                                                              name=bn_name)
        return self.bn_layer

    def restore_layer(self, model_parameters, inputs, is_training):
        self.owner_model_id = model_parameters["owner_model_id"]
        self.layer_id = model_parameters["layer_id"]
        self._create_layer_variables_scope(self.owner_model_id)
        self.beta_initializer = tf.constant_initializer(model_parameters["bn_beta"])
        self.gamma_initializer = tf.constant_initializer(model_parameters["bn_gamma"])
        self.moving_mean_initializer = tf.constant_initializer(model_parameters["bn_moving_mean"])
        self.moving_variance_initializer = tf.constant_initializer(model_parameters["bn_moving_variance"])

        return self._build_layer(inputs, is_training)

    def get_layer_meta(self, session):
        # for v in tf.global_variables():
        #     print(v.name)

        bn_gamma = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.bn_full_name + "gamma")[0]
        bn_beta = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.bn_full_name + "beta")[0]
        bn_moving_mean = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.bn_full_name + "moving_mean")[0]
        bn_moving_variance = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES,
                                               scope=self.bn_full_name + "moving_variance")[0]
        # print("bn_gamma-", bn_gamma)
        # print("bn_beta-", bn_beta)
        # print("bn_moving_mean-", bn_moving_mean)
        # print("bn_moving_variance-", bn_moving_variance)

        bn_gamma = session.run(bn_gamma)
        bn_beta = session.run(bn_beta)
        bn_moving_mean = session.run(bn_moving_mean)
        bn_moving_variance = session.run(bn_moving_variance)

        # print("bn_gamma", bn_gamma)
        # print("bn_beta", bn_beta)
        # print("bn_moving_mean", bn_moving_mean)
        # print("bn_moving_variance", bn_moving_variance)

        layer_model_meta = {"layer_type": self.layer_type,
                            "owner_model_id": self.owner_model_id,
                            "layer_id": self.layer_id,
                            "bn_gamma": bn_gamma,
                            "bn_beta": bn_beta,
                            "bn_moving_mean": bn_moving_mean,
                            "bn_moving_variance": bn_moving_variance
                            }
        identifier = self.generate_layer_identifier(self.owner_model_id, self.layer_id)
        return identifier, layer_model_meta

    def _create_layer_variables_scope(self, owner_model_id):
        self.conv_layer_vars_scope = "model_" + owner_model_id + "_conv_layer_vars"

    def _set_layer_variable_initializer(self):
        self.beta_initializer = init_ops.zeros_initializer()
        self.gamma_initializer = init_ops.ones_initializer()
        self.moving_mean_initializer = init_ops.zeros_initializer()
        self.moving_variance_initializer = init_ops.ones_initializer()


class ConvolutionLayer(Layer):

    def __init__(self, filter_size=None, n_out_channels=None, stride_size=None,
                 padding_mode=None, proximal_lbda=1.0):
        self.layer_type = "convolution"
        self.filter_size = filter_size
        self.n_out_channels = n_out_channels
        self.stride_size = stride_size
        self.padding_mode = padding_mode
        self.n_in_channels = None
        self.owner_model_id = None
        self.layer_id = None
        self.conv_layer_vars_scope = None
        # self.activation_fn = tf.nn.relu
        self.filter_initializer = None

        # self.can_apply_proximal = True
        self.proximal_lbda = proximal_lbda
        self.proximal_placeholder = None

    def build(self, inputs, owner_model_id, layer_id, is_training):
        self.owner_model_id = str(owner_model_id)
        self.layer_id = str(layer_id)
        self._set_layer_variable_initializer(inputs.shape)
        self._create_layer_variables_scope(self.owner_model_id)
        return self._build_layer(inputs)

    def compute_difference_to_proximal(self):
        if self.can_apply_proximal is False:
            raise Exception("This layer can not be applied with the proximal regularization")
        return self.filter_weights - self.proximal_placeholder

    def get_layer_proximal_parameters(self, session):
        if self.can_apply_proximal is False:
            raise Exception("This layer can not be applied with the proximal regularization")
        filter_weights = session.run(self.filter_weights)
        return filter_weights

    def can_apply_proximal(self):
        # print("here", self.layer_id, self.layer_type)
        return True

    def get_proximal_placeholder(self):
        return self.proximal_placeholder

    def _build_layer(self, inputs):
        layer_name = "layer_" + self.layer_id
        vars_name = layer_name + "_conv_vars"
        conv_name = layer_name + "_conv_ops"
        with tf.variable_scope(self.conv_layer_vars_scope):
            with tf.variable_scope(layer_name):
                self.filter_weights = tf.get_variable(name=vars_name, initializer=self.filter_initializer,
                                                      dtype=tf.float32)

                # record learned filters for testing/debugging purpose
                # filters = tf.reshape(self.filter_weights,
                #                   [self.filter_size, self.filter_size, -1, self.n_in_channels * self.n_out_channels])
                # filters = tf.transpose(filters, [3, 0, 1, 2])
                # tf.summary.image("filters", filters, 50)

                strides = [1, self.stride_size, self.stride_size, 1]
                layer = tf.nn.conv2d(inputs, filter=self.filter_weights, strides=strides, padding=self.padding_mode,
                                     name=conv_name)
        return layer

    def _set_layer_variable_initializer(self, input_feature_map_shape):
        self.n_in_channels = input_feature_map_shape[-1].value
        stddev = np.sqrt(2 / (self.n_in_channels + self.n_out_channels))
        filter_shape = [self.filter_size, self.filter_size, self.n_in_channels, self.n_out_channels]
        self.filter_initializer = tf.truncated_normal(
            shape=filter_shape, stddev=stddev, dtype=tf.float32)
        self.proximal_placeholder = tf.placeholder(dtype=tf.float32, shape=filter_shape)

    def _create_layer_variables_scope(self, owner_model_id):
        self.conv_layer_vars_scope = "model_" + owner_model_id + "_conv_layer_vars"

    def get_layer_variables_scope(self):
        return self.conv_layer_vars_scope

    def restore_layer(self, model_parameters, inputs, is_training):
        self.owner_model_id = model_parameters["owner_model_id"]
        self.layer_id = model_parameters["layer_id"]
        self.filter_size = model_parameters["filter_size"]
        self.n_in_channels = model_parameters["n_in_channels"]
        self.n_out_channels = model_parameters["n_out_channels"]
        self.padding_mode = model_parameters["padding_mode"]
        self.stride_size = model_parameters["stride_size"]
        self.filter_initializer = model_parameters["filter_weights"]
        self._create_layer_variables_scope(self.owner_model_id)
        return self._build_layer(inputs)

    def get_layer_meta(self, session):
        filter_weights = session.run(self.filter_weights)
        layer_model_meta = {"layer_type": self.layer_type,
                            "owner_model_id": self.owner_model_id,
                            "layer_id": self.layer_id,
                            "filter_size": self.filter_size,
                            "n_in_channels": self.n_in_channels,
                            "n_out_channels": self.n_out_channels,
                            "padding_mode": self.padding_mode,
                            "stride_size": self.stride_size,
                            "filter_weights": filter_weights}
        identifier = self.generate_layer_identifier(self.owner_model_id, self.layer_id)
        return identifier, layer_model_meta


class MaxPoolingLayer(Layer):

    def __init__(self, filter_size=None, stride_size=None, padding_mode=None):
        self.layer_type = "max_pooling"
        self.padding_mode = padding_mode
        self.filter_size = filter_size
        self.stride_size = stride_size
        self.owner_model_id = None
        self.layer_id = None

    def build(self, inputs, owner_model_id, layer_id, is_training):
        self.owner_model_id = owner_model_id
        self.layer_id = layer_id
        layer_name = "layer_" + self.layer_id
        conv_layer_ops_scope = "model_" + owner_model_id + "_max_pooling_layer_ops"
        ops_name = layer_name + "_max_pooling_ops"
        with tf.name_scope(conv_layer_ops_scope):
            with tf.name_scope(layer_name):
                kernel = [1, self.filter_size, self.filter_size, 1]
                strides = [1, self.stride_size, self.stride_size, 1]
                output = tf.nn.max_pool(inputs, ksize=kernel, strides=strides, padding=self.padding_mode,
                                        name=ops_name)
        return output

    def restore_layer(self, model_parameters, inputs, is_training):
        self.owner_model_id = model_parameters["owner_model_id"]
        self.layer_id = model_parameters["layer_id"]
        self.padding_mode = model_parameters["padding_mode"]
        self.filter_size = model_parameters["filter_size"]
        self.stride_size = model_parameters["stride_size"]
        return self.build(inputs, self.owner_model_id, self.layer_id, is_training)

    def get_layer_meta(self, session):
        layer_model_meta = {"layer_type": self.layer_type,
                            "owner_model_id": self.owner_model_id,
                            "layer_id": self.layer_id,
                            "filter_size": self.filter_size,
                            "stride_size": self.stride_size,
                            "padding_mode": self.padding_mode
                            }
        identifier = self.generate_layer_identifier(self.owner_model_id, self.layer_id)
        return identifier, layer_model_meta


class SimpleCNN(BaseModel):

    def __init__(self, an_id=None):
        super(SimpleCNN, self).__init__()
        self.layers: List[Layer] = []
        self.session = None
        self.model_id = an_id if an_id is None else str(an_id)
        self.feature_representation = None
        self.input_shape = None
        self.representation_dim = None
        self.class_num = None
        self.lr = None
        self.proximal_lbda = None
        self.repr_w_initializer = None
        self.repr_b_initializer = None
        self.logits_w_initializer = None
        self.logits_b_initializer = None

    def set_session(self, session):
        self.session = session

    def build(self, input_shape, representation_dim, class_num, lr, proximal_lbda=1.0):
        self.input_shape = input_shape
        self.representation_dim = representation_dim
        self.class_num = class_num
        self.lr = lr
        self.proximal_lbda = proximal_lbda

        self._set_variable_initializer()
        self._build_model()

    def restore_model(self, model_parameters):

        hyperparameters = model_parameters["hyperparameters"]
        layers_meta = model_parameters["layers_meta"]

        self.model_id = hyperparameters["model_id"]
        self.lr = hyperparameters["learning_rate"]
        self.class_num = hyperparameters["class_num"]
        self.representation_dim = hyperparameters["representation_dim"]
        self.input_shape = hyperparameters["input_shape"]

        repr_layer = layers_meta["repr_layer"]
        logits_layer = layers_meta["logits_layer"]
        self.repr_w_initializer = tf.constant_initializer(repr_layer["repr_kernel"])
        self.repr_b_initializer = tf.constant_initializer(repr_layer["repr_bias"])
        self.logits_w_initializer = tf.constant_initializer(logits_layer["logits_kernel"])
        self.logits_b_initializer = tf.constant_initializer(logits_layer["logits_bias"])

        self._add_input_placeholder()
        self._restore_feature_extraction_layers(layers_meta)
        self._build_feature_representation_layer()
        self._build_objective_layer()

    def _get_repr_logits_tensors(self):
        repr_kernel_name = self.representation_layer_vars_scope + '/dense/kernel:0'
        repr_kernel_tensor = tf.get_default_graph().get_tensor_by_name(repr_kernel_name)
        logits_kernel_name = self.logits_layer_vars_scope + '/dense/kernel:0'
        logits_kernel_tensor = tf.get_default_graph().get_tensor_by_name(logits_kernel_name)
        return repr_kernel_tensor, logits_kernel_tensor

    def _compute_distance_to_proximal(self):
        repr_kernel_tensor, logits_kernel_tensor = self._get_repr_logits_tensors()

        diff_collection = list()
        proximal_placeholder_collection = list()
        repr_diff = repr_kernel_tensor - self.repr_proximal_placeholder
        logits_diff = logits_kernel_tensor - self.logits_proximal_placeholder
        diff_collection.append(repr_diff)
        diff_collection.append(logits_diff)
        proximal_placeholder_collection.append(self.repr_proximal_placeholder)
        proximal_placeholder_collection.append(self.logits_proximal_placeholder)
        for layer in self.layers:
            # print("layer", layer)
            # print("layer.can_apply_proximal()", layer.can_apply_proximal())
            if layer.can_apply_proximal():
                proximal_difference = layer.compute_difference_to_proximal()
                proximal_placeholder = layer.get_proximal_placeholder()
                diff_collection.append(proximal_difference)
                proximal_placeholder_collection.append(proximal_placeholder)
        return diff_collection, proximal_placeholder_collection

    def get_proximal_model(self):
        repr_kernel_tensor, logits_kernel_tensor = self._get_repr_logits_tensors()
        repr_tensor = self.session.run(repr_kernel_tensor)
        logits_tensor = self.session.run(logits_kernel_tensor)
        proximal_model_param_collection = [repr_tensor, logits_tensor]
        for layer in self.layers:
            if layer.can_apply_proximal():
                proximal_tensor = layer.get_layer_proximal_parameters(self.session)
                proximal_model_param_collection.append(proximal_tensor)
        return proximal_model_param_collection

    def get_model_parameters(self):
        repr_kernel_name = self.representation_layer_vars_scope + '/dense/kernel:0'
        repr_bias_name = self.representation_layer_vars_scope + '/dense/bias:0'
        repr_kernel_tensor = tf.get_default_graph().get_tensor_by_name(repr_kernel_name)
        repr_bias_tensor = tf.get_default_graph().get_tensor_by_name(repr_bias_name)

        logits_kernel_name = self.logits_layer_vars_scope + '/dense/kernel:0'
        logits_bias_name = self.logits_layer_vars_scope + '/dense/bias:0'
        logits_kernel_tensor = tf.get_default_graph().get_tensor_by_name(logits_kernel_name)
        logits_bias_tensor = tf.get_default_graph().get_tensor_by_name(logits_bias_name)

        repr_kernel = self.session.run(repr_kernel_tensor)
        repr_bias = self.session.run(repr_bias_tensor)
        logits_kernel = self.session.run(logits_kernel_tensor)
        logits_bias = self.session.run(logits_bias_tensor)

        layers_meta = {}
        feature_extraction_layers_identifiers_list = []
        for layer in self.layers:
            identifier, layer_model = layer.get_layer_meta(self.session)
            feature_extraction_layers_identifiers_list.append(identifier)
            layers_meta[identifier] = layer_model

        layers_meta["feature_extraction_layers_identifiers_list"] = feature_extraction_layers_identifiers_list
        layers_meta["repr_layer"] = {"repr_kernel": repr_kernel, "repr_bias": repr_bias}
        layers_meta["logits_layer"] = {"logits_kernel": logits_kernel, "logits_bias": logits_bias}
        hyperparameters = {"model_id": self.model_id,
                           "learning_rate": self.lr,
                           "class_num": self.class_num,
                           "representation_dim": self.representation_dim,
                           "input_shape": self.input_shape,
                           "num_layers": len(self.layers) + 2}
        return {"hyperparameters": hyperparameters, "layers_meta": layers_meta}

    def _build_model(self):
        self._add_input_placeholder()
        self._build_feature_extraction_layers()
        self._build_feature_representation_layer()
        self._build_objective_layer()

    def _set_variable_initializer(self):
        self.repr_w_initializer = None
        self.repr_b_initializer = init_ops.zeros_initializer()
        self.logits_w_initializer = None
        self.logits_b_initializer = init_ops.zeros_initializer()

    def _add_input_placeholder(self):
        self.inputs = tf.placeholder(tf.float32, [None, self.input_shape[0], self.input_shape[1], self.input_shape[2]],
                                     name="inputs")
        tf.summary.image(self.model_id + '_inputs', self.inputs, 10)

        self.targets = tf.placeholder(tf.float32, [None, self.class_num], name="targets")
        self.is_training = tf.placeholder(tf.bool, name="is_training")

    def _build_feature_extraction_layers(self):
        layer_input = self.inputs
        for layer_index, layer in enumerate(self.layers):
            layer_input = layer.build(layer_input, self.model_id, str(layer_index), self.is_training)
        self.flattened_features = self._flatten_features(layer_input)

    def _restore_feature_extraction_layers(self, layers_meta):
        layer_input = self.inputs
        feature_extraction_layers_identifiers_list = layers_meta["feature_extraction_layers_identifiers_list"]
        for layer_identifier in feature_extraction_layers_identifiers_list:
            layer_meta = layers_meta[layer_identifier]
            layer = LayerFactory.create(layer_meta)
            layer_input = layer.restore_layer(layers_meta[layer_identifier], layer_input, self.is_training)
            self.layers.append(layer)
        self.flattened_features = self._flatten_features(layer_input)

    @staticmethod
    def _flatten_features(features):
        dim = features.shape[1] * features.shape[2] * features.shape[3]
        flattened_features = tf.reshape(features, [-1, dim])
        return flattened_features

    def _build_feature_representation_layer(self):
        self.representation_layer_vars_scope = "model_" + str(self.model_id) + "_representation_layer_vars"
        with tf.variable_scope(self.representation_layer_vars_scope):
            self.feature_representation = tf.layers.dense(inputs=self.flattened_features,
                                                          units=self.representation_dim,
                                                          kernel_initializer=self.repr_w_initializer,
                                                          bias_initializer=self.repr_b_initializer,
                                                          activation=tf.nn.relu)

        self.repr_proximal_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[self.flattened_features.shape[1], self.representation_dim])

        # record learned kernel and bias for testing/debugging purpose
        kernel_name = self.representation_layer_vars_scope + '/dense/kernel:0'
        bias_name = self.representation_layer_vars_scope + '/dense/bias:0'
        kernel = tf.get_default_graph().get_tensor_by_name(kernel_name)
        bias = tf.get_default_graph().get_tensor_by_name(bias_name)
        variable_summaries(kernel, "kernel")
        variable_summaries(bias, "bias")

    def _build_objective_layer(self):
        self.logits_layer_vars_scope = "model_" + str(self.model_id) + "_logits_layer_vars"
        with tf.variable_scope(self.logits_layer_vars_scope):
            self.logits = tf.layers.dense(inputs=self.feature_representation,
                                          units=self.class_num,
                                          kernel_initializer=self.logits_w_initializer,
                                          bias_initializer=self.logits_b_initializer)

        self.logits_proximal_placeholder = tf.placeholder(
            dtype=tf.float32, shape=[self.feature_representation.shape[1], self.class_num])

        self.prediction = tf.nn.softmax(self.logits, name="softmax_tensor")
        self.loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.targets))
        tf.summary.scalar(self.model_id + '_loss', self.loss_op)

        self.init_grad = tf.placeholder(tf.float32, shape=(None, self.representation_dim))

        conv_layer_vars_scope = "model_" + str(self.model_id) + "_conv_layer_vars"
        vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=conv_layer_vars_scope)
        vars_to_train += tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=self.representation_layer_vars_scope)

        # global_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        # for var in vars_to_train:
        #     print(var)
        # print("global variables")
        # for var in global_variables:
        #     print(var)

        # vars_to_train = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)
        # l2_loss = tf.add_n([tf.nn.l2_loss(tf.cast(v, tf.float32)) for v in vars_to_train])

        # the training operations are waiting for batch normalization to be computed
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope=conv_layer_vars_scope)
        with tf.control_dependencies(update_ops):
            loss1 = self.feature_representation
            self.back_train_op_1 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                loss=loss1,
                var_list=vars_to_train,
                grad_loss=self.init_grad)

            diff_collection, self.proximal_placeholder_collection = self._compute_distance_to_proximal()
            loss2 = self.feature_representation
            # for diff in diff_collection:
            #     print("diff shape: {0}".format(diff.shape))
            #     loss2 = loss2 + self.proximal_lbda * tf.nn.l2_loss(diff)
            #     loss2 = loss2 + 0.5 * self.proximal_lbda * tf.square(diff)
            proximal_loss = tf.add_n([tf.nn.l2_loss(v) for v in diff_collection])
            loss2 = loss2 + self.proximal_lbda * proximal_loss
            self.back_train_op_2 = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(
                loss=loss2,
                var_list=vars_to_train,
                grad_loss=self.init_grad)

            self.e2e_train_op = tf.train.AdamOptimizer(learning_rate=self.lr).minimize(self.loss_op)

        # accuracy calculation operations
        self.correct_pred = tf.equal(tf.argmax(self.prediction, 1), tf.argmax(self.targets, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
        tf.summary.scalar(self.model_id + '_accuracy', self.accuracy)

    def backpropogate(self, X, y, in_grad, apply_proximal=False, proximal_param_collection=None):
        if apply_proximal:
            if proximal_param_collection is None:
                raise Exception("proximal should be provided but is None")
            # print("## cnn apply proximal")
            feed_dictionary = {self.inputs: X, self.is_training: True, self.init_grad: in_grad}
            for placeholder, param in zip(self.proximal_placeholder_collection, proximal_param_collection):
                feed_dictionary[placeholder] = param
            self.session.run(self.back_train_op_2, feed_dict=feed_dictionary)
        else:
            # print("## cnn does not apply proximal")
            self.session.run(self.back_train_op_1,
                             feed_dict={self.inputs: X, self.is_training: True, self.init_grad: in_grad})

    def transform(self, X):
        return self.session.run(self.feature_representation, feed_dict={self.inputs: X, self.is_training: False})

    def train(self, X, y):
        _, loss = self.session.run([self.e2e_train_op, self.loss_op],
                                   feed_dict={self.inputs: X, self.is_training: True, self.targets: y})
        return loss

    def evaluate(self, X, y):
        return self.session.run(self.accuracy, feed_dict={self.inputs: X, self.is_training: False, self.targets: y})

    def add_layer(self, layer):
        self.layers.append(layer)

    def get_features_dim(self):
        return self.representation_dim

    def merge_summary(self, merged):
        return self.session.run(merged)


def variable_summaries(var, name):
    """Attach summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope(name + '_summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


class LayerFactory(object):

    @classmethod
    def create(cls, layer_meta):
        layer_type = layer_meta["layer_type"]
        if layer_type is "convolution":
            print("[INFO] add convolution layer:", )
            layer = ConvolutionLayer()
        elif layer_type is "max_pooling":
            print("[INFO] add max_pooling layer")
            layer = MaxPoolingLayer()
        elif layer_type is "batch_normalization":
            print("[INFO] add batch_normalization layer")
            layer = BatchNormalizationLayer()
        elif layer_type is "activation":
            layer = ActivationLayerFactory.create(layer_meta)
        else:
            raise TypeError("does not support {} layer".format(layer_type))
        return layer


class ActivationLayerFactory(object):

    @classmethod
    def create(cls, layer_meta):
        activation_type = layer_meta["activation_type"]
        if activation_type is "relu":
            print("[INFO] add relu activation layer:", )
            layer = ReluActivationLayer()
        else:
            raise TypeError("does not support {} activation".format(activation_type))
        return layer
