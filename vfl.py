import numpy as np
import time
from models.base_model import BaseModel
from models.optimizer import Optimizer

"""
    The following three classes are to mimic the federated learning procedure between two parties. They are solely for 
    developing and testing purpose. Their roles should be replaced by guest, host, arbiter/coordinator in a distributed 
    environment for integration testing and production phase.
"""


class PartyModelInterface(object):

    def send_components(self):
        pass

    def receive_components(self, components):
        pass

    def send_gradients(self):
        pass

    def receive_gradients(self, gradients):
        pass

    def predict(self, X, msg=None):
        pass


def glorot_normal(fan_in, fan_out):
    stddev = np.sqrt(2 / (fan_in + fan_out))
    return np.random.normal(0, stddev, (fan_in, fan_out))


# TODO: this function is not numerical stable. Fix it when having time.
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class VFLGuestModel(object):

    def __init__(self, local_model, learning_rate, n_iter=1, lbda=0.01, is_ave=False, optimizer="adam",
                 apply_proximal=False, proximal_lbda=0.1):
        super(VFLGuestModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_features_dim()
        self.learning_rate = learning_rate
        self.lbda = lbda
        self.n_iter = n_iter
        self.is_ave = is_ave
        self.optimizer = Optimizer(learning_rate=learning_rate, opt_method_name=optimizer)
        self.learning_rate_decay = None
        self.is_debug = False

        # state
        self.W = self._init_weights(self.feature_dim + 1, 1)
        self.merged = None
        self.parties_grad_component_list = []
        self.parties_loss_component_list = []
        self.parties_loss_reg_term_list = []
        self.collected_common_grad = None
        self.current_global_step = None

        # following hyper_parameters are about applying proximal
        self.apply_proximal = apply_proximal
        self.reserved_W = None
        self.reserved_local_model_param = None
        self._proximal_saved = False
        self.proximal_lbda = proximal_lbda

    def set_session(self, session):
        self.localModel.set_session(session)

    def set_batch(self, X, y, global_step):
        self.X = X
        self.y = y
        self.collected_common_grad = np.zeros((self.X.shape[0], 1))
        self.current_global_step = global_step

    def set_learning_rate_decay_func(self, learning_rate_decay_func):
        self.learning_rate_decay = learning_rate_decay_func

    @staticmethod
    def _init_weights(feature_dim, dim):
        return glorot_normal(feature_dim, dim)

    def fit(self, X, y):
        temp_K_Z = self.localModel.transform(X)
        bias = np.ones((temp_K_Z.shape[0], 1))
        self.K_Z = np.concatenate([temp_K_Z, bias], axis=1)

        self.K_U = np.dot(self.K_Z, self.W)
        self.K_U_2 = np.square(self.K_U)

        self._compute_common_gradient_and_loss(y)
        self._update_local_model(X, y)
        self._update_federated_layer_model()
        # self._update_federated_layer_model()
        # self._update_local_model(X, y)

    def predict(self, X, component_list):
        # self.K_Z = self.localModel.transform(X)
        temp_K_Z = self.localModel.transform(X)
        bias = np.ones((temp_K_Z.shape[0], 1))
        self.K_Z = np.concatenate([temp_K_Z, bias], axis=1)
        self.K_U = np.matmul(self.K_Z, self.W)
        U = self.K_U
        for comp in component_list:
            U = U + comp
        return sigmoid(np.sum(U, axis=1))

    def receive_components(self, component_list):
        self._set_proximal_saved(False)
        for party_component in component_list:
            self.parties_grad_component_list.append(party_component[0])
            self.parties_loss_component_list.append(party_component[1])
            self.parties_loss_reg_term_list.append(party_component[2])

    def fit_one(self):
        self.fit(self.X, self.y)

    def fit_additional_iterations(self):
        # n_iter = self.n_iter if is_seq else self.n_iter - 1
        # iter_for_reserve = 1 if is_seq else 0
        n_iter = self.n_iter - 1
        iter_for_reserve = 0
        print("@ number of local iterations: {0}".format(self.n_iter))
        for i in range(n_iter):

            # when applying proximal and it is the second iteration, we preserve the model of the first iteration
            if self.apply_proximal and i == iter_for_reserve:
                print("@ Guest local iteration {0} (actual {1}), and reserve model".format(i, i + 1))
                self._reserve_model_parameters()
                self._set_proximal_saved(True)

            self.fit(self.X, self.y)
        self.parties_grad_component_list = []
        self.parties_loss_component_list = []
        self.parties_loss_reg_term_list = []

    def _reserve_model_parameters(self):
        self.reserved_W = self.W.copy()
        self.reserved_local_model_param = self.localModel.get_proximal_model()

    def _is_proximal_saved(self):
        return self._proximal_saved

    def _set_proximal_saved(self, _proximal_saved):
        self._proximal_saved = _proximal_saved

    def _compute_common_gradient_and_loss(self, y):
        # if self.A_U.shape[0] != self.K_U.shape[0] or self.B_U.shape[0] != self.K_U.shape[0]:
        #     raise Exception("the first dim of K_U, A_U and B_U should be the same")
        U = self.K_U
        loss_reg_term_sum = np.squeeze(0.5 * self.lbda * np.dot(np.transpose(self.W), self.W))
        for grad_comp, loss_reg_term in zip(self.parties_grad_component_list, self.parties_loss_reg_term_list):
            U = U + grad_comp
            loss_reg_term_sum = loss_reg_term_sum + loss_reg_term

        yU = 0.5 * y * U
        # print("y*y {0}, sum {1}".format(y * y, np.sum(y * y)))
        self.common_grad = 0.5 * y * (yU - 1.0) / (y.shape[0] * self.feature_dim)

        self.partial_common_grad = 0.5 * (0.5 * self.K_U - y) / (y.shape[0] * self.feature_dim)
        # self.common_grad = 0.5 * y * (yU - 1.0) / y.shape[0]

        if self.is_debug:
            print("y shape: {0}".format(y.shape))
            print("U shape: {0}".format(U.shape))
            print("yU shape: {0}".format(yU.shape))
            print("common_grad shape: {0}".format(self.common_grad.shape))

        self.collected_common_grad += self.common_grad

        # TODO: this implementation is not compatible with HE
        loss_1 = np.mean(np.log(2) - 0.5 * yU + 0.125 * np.square(U))
        loss_2 = loss_reg_term_sum / y.shape[0]
        self.loss = loss_1 + loss_2

    def send_gradients(self):
        common_grad = self.common_grad
        # if self.is_ave:
        #     common_grad = self.collected_common_grad / self.n_iter
        return common_grad, self.partial_common_grad

    def _update_federated_layer_model(self):
        # TODO: apply vectorization
        W_grad = []
        for j in range(self.K_Z.shape[1]):
            grad_j = 0
            for i in range(self.K_Z.shape[0]):
                if np.fabs(self.K_Z[i, j]) < 1e-5:
                    continue
                # pitfall
                grad_j += self.common_grad[i, 0] * self.K_Z[i, j]
            W_grad.append(grad_j)

        W_grad = np.array(W_grad).reshape(self.W.shape) + self.lbda * self.W
        if self._is_proximal_saved():
            print("@ Guest is applying proximal")
            proximal_reg = self.proximal_lbda * (self.W - self.reserved_W)
            W_grad = W_grad + proximal_reg

        learning_rate = self.learning_rate_decay(self.learning_rate,
                                                 self.current_global_step) \
            if self.learning_rate_decay is not None \
            else self.learning_rate

        self.W = self.W - learning_rate * W_grad
        # self.W -= self.optimizer.apply_gradients(W_grad)

        # W_grad = np.array(W_grad).reshape(self.W.shape) + self.lbda * self.W
        # self.W = self.W - self.learning_rate * W_grad

        # print("common_grad: ", self.common_grad.shape[0])
        # W_grad = np.array(W_grad).reshape(self.W.shape) + self.lbda * self.W / self.common_grad.shape[0]
        # self.W -= self.optimizer.apply_gradients(W_grad)

    def _update_local_model(self, X, y):
        # print("self.common_grad shape", self.common_grad.shape)
        # print("self.W shape", self.W.shape)
        W = self.W[:self.feature_dim]
        back_grad = np.matmul(self.common_grad, np.transpose(W))
        # bp_loss = self.localModel.backpropogate(X, y, back_grad)
        if self._is_proximal_saved():
            self.localModel.backpropogate(X, y, back_grad, True, self.reserved_local_model_param)
        else:
            self.localModel.backpropogate(X, y, back_grad, False)

    def get_loss(self):
        return self.loss

    def get_local_iteration(self):
        return self.n_iter

    def get_is_average(self):
        return self.is_ave


class VFLHostModel(PartyModelInterface):

    def __init__(self, local_model, learning_rate=0.01, n_iter=1, lbda=0.01, optimizer="adam",
                 apply_proximal=False, proximal_lbda=0.1):
        super(VFLHostModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_features_dim()
        self.learning_rate = learning_rate
        self.lbda = lbda
        self.n_iter = n_iter
        self.optimizer = Optimizer(learning_rate=learning_rate, opt_method_name=optimizer)
        self.learning_rate_decay = None
        self.is_debug = False

        self.W = self._init_weights(self.feature_dim, 1)
        self.X = None
        self.merged = None
        self.common_grad = None
        self.partial_common_grad = None
        self.current_global_step = None

        # following hyper_parameters are about applying proximal
        self.apply_proximal = apply_proximal
        self.reserved_W = None
        self.reserved_local_model_param = None
        self._applying_proximal = False
        self.proximal_lbda = proximal_lbda

    def set_session(self, session):
        self.localModel.set_session(session)

    def set_batch(self, X, global_step):
        self.X = X
        self.current_global_step = global_step

    def set_learning_rate_decay_func(self, learning_rate_decay_func):
        self.learning_rate_decay = learning_rate_decay_func

    @staticmethod
    def _init_weights(feature_dim, dim):
        return glorot_normal(feature_dim, dim)

    def _forward_computation(self, X):
        self.A_Z = self.localModel.transform(X)
        # print("A_Z:", self.A_Z.shape)
        self.A_U = np.dot(self.A_Z, self.W)
        # print("A_U:", self.A_U.shape)
        self.A_U_2 = np.square(self.A_U)

    def _fit(self, X, y):
        self._update_local_model(X, y)
        self._update_federated_layer_model()
        # self._receive_common_gradients(common_grad)
        # self._update_federated_layer_model()
        # self._update_local_model(X, y)

    def receive_gradients(self, gradients):
        self.common_grad = gradients[0]
        self.partial_common_grad = gradients[1]

        if self.is_debug:
            print("common_grad : {0} with shape {1}".format(self.common_grad[0], self.common_grad.shape))
            print("partial_common_grad : {0} with shape {1}".format(self.partial_common_grad[0],
                                                                    self.partial_common_grad.shape))

        print("@ number of local iterations: {0}".format(self.n_iter))
        self._set_applying_proximal(False)
        for i in range(self.n_iter):

            # when applying proximal and it is the second iteration, we preserve the model of the first iteration
            if self.apply_proximal and i == 1:
                print("@ Host local iteration {0}, and reserve model".format(i))
                self._reserve_model_parameters()
                self._set_applying_proximal(True)

            self._fit(self.X, None)

            # if it is not the last iteration, computes forward pass and common grad
            if i != self.n_iter - 1:
                print("@ Host is not in last local iter, computes forward pass and common grad")
                self._forward_computation(self.X)
                self._compute_common_gradient()
            else:
                print("@ Host is in the last local iter")

    def _compute_common_gradient(self):
        # self.partial_common_grad = (0.5 * 0.5 * self.K_U - 0.5 * y) / (y.shape[0] * self.feature_dim)

        if self.is_debug:
            print("self.A_U.shape[0]:", self.A_U.shape[0])

        self.common_grad = self.partial_common_grad + 0.5 * 0.5 * self.A_U / (self.A_U.shape[0] * 50)

    def _reserve_model_parameters(self):
        self.reserved_W = self.W.copy()
        self.reserved_local_model_param = self.localModel.get_proximal_model()

    def _is_applying_proximal(self):
        return self._applying_proximal

    def _set_applying_proximal(self, is_applying_proximal):
        self._applying_proximal = is_applying_proximal

    def send_components(self):
        self._forward_computation(self.X)
        loss_reg_term = np.squeeze(0.5 * self.lbda * np.dot(np.transpose(self.W), self.W))
        # if self._is_applying_proximal():
        #     proximal_loss = 0.5 * self.proximal_lbda * np.sum(np.square(self.W - self.reserved_W))
        #     loss_reg_term = loss_reg_term + proximal_loss

        return self.A_U, self.A_U_2, loss_reg_term

    def _update_federated_layer_model(self):
        # TODO: apply vectorization
        W_grad = []
        for j in range(self.A_Z.shape[1]):
            grad_j = 0
            for i in range(self.A_Z.shape[0]):
                if np.fabs(self.A_Z[i, j]) < 1e-5:
                    continue
                grad_j += self.common_grad[i, 0] * self.A_Z[i, j]
            W_grad.append(grad_j)

        W_grad = np.array(W_grad).reshape(self.W.shape) + self.lbda * self.W
        if self._is_applying_proximal():
            print("@ Host is applying proximal")
            proximal_reg = self.proximal_lbda * (self.W - self.reserved_W)
            W_grad = W_grad + proximal_reg

        learning_rate = self.learning_rate_decay(self.learning_rate,
                                                 self.current_global_step) \
            if self.learning_rate_decay is not None \
            else self.learning_rate

        self.W = self.W - learning_rate * W_grad
        # self.W -= self.optimizer.apply_gradients(W_grad)

        # W_grad = np.array(W_grad).reshape(self.W.shape) + self.lbda * self.W / self.common_grad.shape[0]
        # self.W -= self.optimizer.apply_gradients(W_grad)

    def _update_local_model(self, X, y):
        back_grad = np.matmul(self.common_grad, np.transpose(self.W))
        if self._is_applying_proximal():
            self.localModel.backpropogate(X, y, back_grad, True, self.reserved_local_model_param)
        else:
            self.localModel.backpropogate(X, y, back_grad, False)

    def predict(self, X, msg=None):
        self._forward_computation(X)
        return self.A_U

    def get_local_iteration(self):
        return self.n_iter


class VerticalMultiplePartyLogisticRegressionFederatedLearning(BaseModel):

    def __init__(self, party_A: VFLGuestModel, main_party_id="K"):
        super(VerticalMultiplePartyLogisticRegressionFederatedLearning, self).__init__()
        self.main_party_id = main_party_id
        self.party_a = party_A
        self.party_dict = dict()

    def get_main_party_id(self):
        return self.main_party_id

    def add_party(self, *, id, party_model):
        self.party_dict[id] = party_model

    # def set_learning_rate(self, learning_rate):
    #     self.party_a.set_learning_rate(learning_rate)
    #     for _, party in self.party_dict.items():
    #         party.set_learning_rate(learning_rate)

    def fit(self, X_A, y, party_X_dict, global_step):
        print("------ fit using sequential scheme ------")

        self.party_a.set_batch(X_A, y, global_step)

        for id, party_X in party_X_dict.items():
            self.party_dict[id].set_batch(party_X, global_step)

        print("=> Guest receive immediate result from hosts")
        comp_list = []
        for party in self.party_dict.values():
            comp_list.append(party.send_components())
        self.party_a.receive_components(component_list=comp_list)

        start_time = time.time()
        print("=> Guest computes loss")
        self.party_a.fit_one()
        loss = self.party_a.get_loss()

        print("=> Guest does local update")
        self.party_a.fit_additional_iterations()

        print("=> Guest sends out common grad")
        grad_result = self.party_a.send_gradients()

        print("=> Hosts receive common grad from guest and do local updates")
        for party in self.party_dict.values():
            party.receive_gradients(grad_result)

        end_time = time.time()

        spend_time = end_time - start_time
        return loss, spend_time

    def fit_parallel(self, X_A, y, party_X_dict, global_step):
        print("------ fit using parallel scheme ------")

        self.party_a.set_batch(X_A, y, global_step)

        for id, party_X in party_X_dict.items():
            self.party_dict[id].set_batch(party_X, global_step)

        # collect intermediate gradient and loss components from all parties
        print("=> Guest receive immediate result from hosts")
        comp_list = []
        for party in self.party_dict.values():
            comp_list.append(party.send_components())
        self.party_a.receive_components(component_list=comp_list)

        start_time = time.time()
        print("=> Guest sends out common grad and computes loss")
        self.party_a.fit_one()
        grad_result = self.party_a.send_gradients()
        loss = self.party_a.get_loss()

        print("=> Guest does local update")
        self.party_a.fit_additional_iterations()

        end_time = time.time()

        print("=> Hosts receive common grad from guest and do local updates")
        for party in self.party_dict.values():
            party.receive_gradients(grad_result)

        spend_time = end_time - start_time
        return loss, spend_time

    def predict(self, X_A, party_X_dict):
        comp_list = []
        for id, party_X in party_X_dict.items():
            comp_list.append(self.party_dict[id].predict(party_X))
        return self.party_a.predict(X_A, component_list=comp_list)

    def set_session(self, session):
        self.party_a.set_session(session)
        for party in self.party_dict.values():
            party.set_session(session)

    def get_local_iteration_of_parties(self):
        local_iter_list = [self.party_a.get_local_iteration()]
        for party in self.party_dict.values():
            local_iter_list.append(party.get_local_iteration())
        return local_iter_list

    def is_average(self):
        return self.party_a.get_is_average()
