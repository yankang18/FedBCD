import time

import numpy as np

from models.base_model import BaseModel

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


# TODO: this function is not numerical stable.
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


class VFLGuestModel(object):

    def __init__(self, local_model, learning_rate, n_iter=1, reg_lbda=0.01, apply_proximal=False, proximal_lbda=0.1,
                 is_debug=False, verbose=False):
        super(VFLGuestModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_features_dim()
        self.learning_rate = learning_rate
        self.reg_lbda = reg_lbda
        self.n_iter = n_iter
        self.learning_rate_decay = None
        self.is_debug = is_debug
        self.verbose = verbose

        self.guest_id = self.localModel.get_ID()

        # state
        self.W = self._init_weights(self.feature_dim + 1, 1)
        self.merged = None
        self.parties_grad_component_list = []
        self.parties_loss_component_list = []
        self.parties_loss_reg_term_list = []
        self.current_global_step = None

        # following hyper_parameters are relevant to applying proximal term
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

    def predict(self, X, component_list):
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
        if self.verbose: print("=>    [INFO] Guest-{} local iteration: 1".format(self.guest_id))
        self.fit(self.X, self.y)

    def fit_additional_iterations(self):
        additional_n_iter = self.n_iter - 1
        iter_for_reserve = 0
        for i in range(additional_n_iter):
            if self.verbose: print("=>    [INFO] Guest-{} local iteration: {}".format(self.guest_id, i + 2))
            if self.apply_proximal and i == iter_for_reserve:
                if self.is_debug: print("[DEBUG] Guest reserves model at local iteration:{}".format(i + 2))
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
        U = self.K_U
        loss_reg_term_sum = np.squeeze(0.5 * self.reg_lbda * np.dot(np.transpose(self.W), self.W))
        for grad_comp, loss_reg_term in zip(self.parties_grad_component_list, self.parties_loss_reg_term_list):
            U = U + grad_comp
            loss_reg_term_sum = loss_reg_term_sum + loss_reg_term

        yU = 0.5 * y * U
        self.common_grad = 0.5 * y * (yU - 1.0) / (y.shape[0] * self.feature_dim)
        self.partial_common_grad = 0.5 * (0.5 * self.K_U - y) / (y.shape[0] * self.feature_dim)

        # print("y*y {0}, sum {1}".format(y * y, np.sum(y * y)))
        # self.common_grad = 0.5 * y * (yU - 1.0) / y.shape[0]

        if self.is_debug:
            print("[DEBUG] y shape: {0}".format(y.shape))
            print("[DEBUG] U shape: {0}".format(U.shape))
            print("[DEBUG] yU shape: {0}".format(yU.shape))
            print("[DEBUG] common_grad shape: {0}".format(self.common_grad.shape))

        # TODO: this implementation is not compatible with HE
        loss_1 = np.mean(np.log(2) - 0.5 * yU + 0.125 * np.square(U))
        loss_2 = loss_reg_term_sum / y.shape[0]
        self.loss = loss_1 + loss_2

    def send_gradients(self):
        common_grad = self.common_grad
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

        W_grad = np.array(W_grad).reshape(self.W.shape) + self.reg_lbda * self.W
        if self._is_proximal_saved():
            if self.is_debug: print("[DEBUG] Guest is applying proximal")
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
        W = self.W[:self.feature_dim]
        back_grad = np.matmul(self.common_grad, np.transpose(W))

        if self._is_proximal_saved():
            self.localModel.backpropogate(X, y, back_grad, True, self.reserved_local_model_param)
        else:
            self.localModel.backpropogate(X, y, back_grad, False)

    def get_loss(self):
        return self.loss

    def get_local_iteration(self):
        return self.n_iter


class VFLHostModel(PartyModelInterface):

    def __init__(self, local_model, learning_rate=0.01, n_iter=1, reg_lbda=0.01, apply_proximal=False,
                 proximal_lbda=0.1, is_debug=False, verbose=False):
        super(VFLHostModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_features_dim()
        self.learning_rate = learning_rate
        self.reg_lbda = reg_lbda
        self.n_iter = n_iter
        self.learning_rate_decay = None
        self.is_debug = is_debug
        self.verbose = verbose

        self.host_id = self.localModel.get_ID()

        self.W = self._init_weights(self.feature_dim, 1)
        self.X = None
        self.merged = None
        self.common_grad = None
        self.partial_common_grad = None
        self.current_global_step = None

        # following hyper_parameters are relevant to applying proximal
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
        self.A_U = np.dot(self.A_Z, self.W)
        self.A_U_2 = np.square(self.A_U)

    def _fit(self, X, y):
        self._update_local_model(X, y)
        self._update_federated_layer_model()

    def receive_gradients(self, gradients):
        self.common_grad = gradients[0]
        self.partial_common_grad = gradients[1]

        if self.is_debug:
            print("[DEBUG] common_grad : {0} with shape {1}".format(self.common_grad[0], self.common_grad.shape))
            print("[DEBUG] partial_common_grad : {0} with shape {1}".format(self.partial_common_grad[0],
                                                                            self.partial_common_grad.shape))

        self._set_applying_proximal(False)
        for i in range(self.n_iter):
            if self.verbose: print("=>    [INFO] Host-{} local iteration: {}".format(self.host_id, i + 1))
            # when applying proximal and it is the second iteration, we preserve the model of the first iteration
            if self.apply_proximal and i == 1:
                if self.is_debug: print("[DEBUG] Host reserves model at local iteration:{}".format(i + 1))
                self._reserve_model_parameters()
                self._set_applying_proximal(True)

            self._fit(self.X, None)

            # if it is not the last iteration, computes forward pass and common grad
            if i != self.n_iter - 1:
                if self.is_debug: print("[DEBUG] Host is not at last local iter, and"
                                        " computes forward pass and common grad.")
                self._forward_computation(self.X)
                self._compute_common_gradient()
            else:
                if self.is_debug: print("[DEBUG] Host is at the last local iter.")

    def _compute_common_gradient(self):
        if self.is_debug: print("[DEBUG] self.A_U.shape[0]:", self.A_U.shape[0])
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
        loss_reg_term = np.squeeze(0.5 * self.reg_lbda * np.dot(np.transpose(self.W), self.W))
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

        W_grad = np.array(W_grad).reshape(self.W.shape) + self.reg_lbda * self.W
        if self._is_applying_proximal():
            if self.is_debug: print("[DEBUG] Host is applying proximal")
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


class VerticalMultiplePartyFederatedLearning(BaseModel):

    def __init__(self, party_a: VFLGuestModel, main_party_id="K", verbose=False):
        super(VerticalMultiplePartyFederatedLearning, self).__init__()
        self.main_party_id = main_party_id
        self.party_a = party_a
        self.party_dict = dict()
        self.verbose = verbose

    def get_main_party_id(self):
        return self.main_party_id

    def add_party(self, *, id, party_model):
        self.party_dict[id] = party_model

    def fit(self, X_A, y, party_X_dict, global_step):

        self.party_a.set_batch(X_A, y, global_step)

        for id, party_X in party_X_dict.items():
            self.party_dict[id].set_batch(party_X, global_step)

        if self.verbose: print("=> [INFO] Guest receive immediate result from hosts")
        comp_list = []
        for party in self.party_dict.values():
            comp_list.append(party.send_components())
        self.party_a.receive_components(component_list=comp_list)

        start_time = time.time()
        if self.verbose: print("=> [INFO] Guest computes loss")
        self.party_a.fit_one()
        loss = self.party_a.get_loss()

        if self.verbose: print("=> [INFO] Guest performs additional local updates")
        self.party_a.fit_additional_iterations()

        if self.verbose: print("=> [INFO] Guest sends out common grad")
        grad_result = self.party_a.send_gradients()

        if self.verbose: print("=> [INFO] Hosts receive common grad from guest and do local updates")
        for party in self.party_dict.values():
            party.receive_gradients(grad_result)

        end_time = time.time()

        spend_time = end_time - start_time
        return loss, spend_time

    def fit_parallel(self, X_A, y, party_X_dict, global_step):

        self.party_a.set_batch(X_A, y, global_step)

        for id, party_X in party_X_dict.items():
            self.party_dict[id].set_batch(party_X, global_step)

        # collect intermediate gradient and loss components from all parties
        if self.verbose: print("=> [INFO] Guest receive immediate result from hosts")
        comp_list = []
        for party in self.party_dict.values():
            comp_list.append(party.send_components())
        self.party_a.receive_components(component_list=comp_list)

        start_time = time.time()
        if self.verbose: print("=> [INFO] Guest sends out common grad and computes loss")
        self.party_a.fit_one()
        grad_result = self.party_a.send_gradients()
        loss = self.party_a.get_loss()

        if self.verbose: print("=> [INFO] Guest performs additional local updates")
        self.party_a.fit_additional_iterations()

        end_time = time.time()

        if self.verbose: print("=> [INFO] Hosts receive common grad from guest and do local updates")
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
