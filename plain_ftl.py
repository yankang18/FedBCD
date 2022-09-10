import numpy as np

from models.base_model import BaseModel
from vnn_demo.vfl import PartyModelInterface


def sigmoid(x):
    return 1. / (1. + np.exp(-x))


class PlainFTLGuestModel(PartyModelInterface):

    def __init__(self, local_model, alpha):
        super(PlainFTLGuestModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_representation_dim()
        self.alpha = alpha
        self.merged = None
        self.is_trace = False

    def set_batch(self, X, y, non_overlap_indexes, overlap_indexes):
        self.X = X
        self.y = y
        self.non_overlap_indexes = non_overlap_indexes
        self.overlap_indexes = overlap_indexes

    def __compute_yA_uA(self, U_A, y):
        length_y = len(y)
        return np.expand_dims(np.sum(y * U_A, axis=0) / length_y, axis=0)

    def __compute_components(self):
        # y_A_u_A has shape (1, feature_dim)
        # y_A_u_A_2 has shape (feature_dim, feature_dim)
        self.y_A_u_A = self.__compute_yA_uA(self.U_A, self.y)
        self.y_A_u_A_2 = np.matmul(self.y_A_u_A.transpose(), self.y_A_u_A)

        # y_overlap and y_overlap2 have shape (len(overlap_indexes), 1)
        self.y_overlap = self.y[self.overlap_indexes]
        self.y_overlap2 = self.y_overlap * self.y_overlap

        if self.is_trace:
            print("y_A_u_A shape", self.y_A_u_A.shape)
            print("y_A_u_A_2 shape", self.y_A_u_A_2.shape)
            print("y_overlap shape", self.y_overlap.shape)
            print("y_overlap2 shape", self.y_overlap2.shape)

        # following two parameters will be sent to the other party for computing gradients
        # comp_A_beta1 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # comp_A_beta2 has shape (len(overlap_indexes), feature_dim)
        self.comp_A_beta1 = 0.25 * np.expand_dims(self.y_overlap2, axis=2) * self.y_A_u_A_2
        self.comp_A_beta2 = -0.5 * self.y_overlap * self.y_A_u_A

        if self.is_trace:
            print("comp_A_beta1 shape", self.comp_A_beta1.shape)
            print("comp_A_beta2 shape", self.comp_A_beta2.shape)

    def send_components(self):
        self.U_A = self.localModel.transform(self.X)
        self.U_A_overlap = self.U_A[self.overlap_indexes]
        self.__compute_components()
        # mapping_comp_A has shape (len(overlap_indexes), feature_dim)
        mapping_comp_A = - self.U_A_overlap / self.feature_dim

        if self.is_trace:
            print("comp_A_beta1 shape", self.comp_A_beta1.shape)
            print("comp_A_beta2 shape", self.comp_A_beta2.shape)
            print("mapping_comp_A shape", mapping_comp_A.shape)

        return [self.comp_A_beta1, self.comp_A_beta2, mapping_comp_A]

    def receive_components(self, components):
        self.U_B_overlap = components[0]
        self.U_B_2_overlap = components[1]
        self.mapping_comp_B = components[2]
        self.__update_gradients()
        self.__update_loss()

    def __update_gradients(self):

        # y_overlap2 have shape (len(overlap_indexes), 1),
        # y_A_u_A has shape (1, feature_dim),
        # y_overlap2_y_A_u_A has shape (len(overlap_indexes), 1, feature_dim)
        y_overlap2_y_A_u_A = np.expand_dims(self.y_overlap2 * self.y_A_u_A, axis=1)

        # U_B_2_overlap has shape (len(overlap_indexes), feature_dim, feature_dim)
        # tmp has shape (len(overlap_indexes), feature_dim)
        tmp = 0.25 * np.squeeze(np.matmul(y_overlap2_y_A_u_A, self.U_B_2_overlap), axis=1)

        if self.is_trace:
            print("tmp shape", tmp.shape)
            print("tmp", tmp)

            print("## self.y_overlap shape", self.y_overlap.shape)
            print("## self.y_overlap \n", self.y_overlap)
            print("## self.U_B_overlap shape", self.U_B_overlap.shape)
            print("## self.U_B_overlap \n", self.U_B_overlap)

            # const has shape (feature_dim,)
            print("*** self.y_overlap", self.y_overlap)
            print("*** self.U_B_overlap", self.U_B_overlap)
            print("*** self.y_overlap * self.U_B_overlap", self.y_overlap * self.U_B_overlap)
            print("*** self.y_overlap * self.U_B_overlap", 0.5 * (np.sum(self.y_overlap * self.U_B_overlap, axis=0)))

        const = np.sum(tmp, axis=0) - 0.5 * np.sum(self.y_overlap * self.U_B_overlap, axis=0)
        # grad_A_nonoverlap has shape (len(non_overlap_indexes), feature_dim)
        # grad_A_overlap has shape (len(overlap_indexes), feature_dim)
        grad_A_nonoverlap = self.alpha * const * self.y[self.non_overlap_indexes] / len(self.y)
        grad_A_overlap = self.alpha * const * self.y_overlap / len(self.y) + self.mapping_comp_B

        if self.is_trace:
            print("*** const:", const)
            print("*** grad_A_nonoverlap \n", grad_A_nonoverlap)
            print("*** grad_A_overlap \n", grad_A_overlap)

        grad_loss_A = np.zeros((len(self.y), self.U_B_overlap.shape[1]))
        grad_loss_A[self.non_overlap_indexes, :] = grad_A_nonoverlap
        grad_loss_A[self.overlap_indexes, :] = grad_A_overlap
        self.loss_grads = grad_loss_A
        self.localModel.backpropogate(self.X, self.y, grad_loss_A)

    def send_loss(self):
        return self.loss

    def receive_loss(self, loss):
        self.loss = loss

    def __update_loss(self):
        U_A_overlap_prime = - self.U_A_overlap / self.feature_dim
        loss_overlap = np.sum(U_A_overlap_prime * self.U_B_overlap)
        loss_Y = self.__compute_loss_y(self.U_B_overlap, self.y_overlap, self.y_A_u_A)
        if self.is_trace:
            print("loss_Y", loss_Y)
        self.loss = self.alpha * loss_Y + loss_overlap

    def __compute_loss_y(self, U_B_overlap, y_overlap, y_A_u_A):
        UB_yAuA = np.matmul(U_B_overlap, y_A_u_A.transpose())
        loss_Y = (-0.5 * np.sum(y_overlap * UB_yAuA) + 1.0 / 8 * np.sum(UB_yAuA * UB_yAuA)) + len(y_overlap) * np.log(2)
        if self.is_trace:
            print("UB_yAuA shape", UB_yAuA.shape)
            part1 = y_overlap * UB_yAuA
            part2 = UB_yAuA * UB_yAuA
            print("part1 shape", part1.shape)
            print("part2 shape", part2.shape)
        return loss_Y

    def get_loss_grads(self):
        return self.loss_grads

    def predict(self, U_B, msg=None):
        return sigmoid(np.matmul(U_B, self.y_A_u_A.transpose()))


class PlainFTLHostModel(PartyModelInterface):

    def __init__(self, local_model, alpha):
        super(PlainFTLHostModel, self).__init__()
        self.localModel = local_model
        self.feature_dim = local_model.get_representation_dim()
        self.alpha = alpha
        self.merged = None

    def set_batch(self, X, overlap_indexes):
        self.X = X
        self.overlap_indexes = overlap_indexes

    def send_components(self):
        self.U_B = self.localModel.transform(self.X)
        # U_B_overlap has shape (len(overlap_indexes), feature_dim)
        # U_B_overlap_2 has shape (len(overlap_indexes), feature_dim, feature_dim)
        # mapping_comp_B has shape (len(overlap_indexes), feature_dim)
        self.U_B_overlap = self.U_B[self.overlap_indexes]
        U_B_overlap_2 = np.matmul(np.expand_dims(self.U_B_overlap, axis=2), np.expand_dims(self.U_B_overlap, axis=1))
        mapping_comp_B = - self.U_B_overlap / self.feature_dim
        return [self.U_B_overlap, U_B_overlap_2, mapping_comp_B]

    def receive_components(self, components):
        self.comp_A_beta1 = components[0]
        self.comp_A_beta2 = components[1]
        self.mapping_comp_A = components[2]
        self.__update_gradients()

    def __update_gradients(self):
        U_B_overlap_ex = np.expand_dims(self.U_B_overlap, axis=1)
        U_B_comp_A_beta1 = np.matmul(U_B_overlap_ex, self.comp_A_beta1)
        grad_l1_B = np.squeeze(U_B_comp_A_beta1, axis=1) + self.comp_A_beta2
        grad_loss_B = self.alpha * grad_l1_B + self.mapping_comp_A
        self.loss_grads = grad_loss_B
        self.localModel.backpropogate(self.X[self.overlap_indexes], None, grad_loss_B)

    def get_loss_grads(self):
        return self.loss_grads

    def predict(self, X, msg=None):
        return self.localModel.transform(X)


class LocalPlainFederatedTransferLearning(BaseModel):

    def __init__(self, party_A: PlainFTLGuestModel, party_B: PlainFTLHostModel, private_key=None):
        super(LocalPlainFederatedTransferLearning, self).__init__()
        self.party_a = party_A
        self.party_b = party_B
        self.private_key = private_key

    def fit(self, X_A, X_B, y, overlap_indexes, non_overlap_indexes):
        self.party_a.set_batch(X_A, y, non_overlap_indexes, overlap_indexes)
        self.party_b.set_batch(X_B, overlap_indexes)
        comp_B = self.party_b.send_components()
        comp_A = self.party_a.send_components()
        self.party_a.receive_components(comp_B)
        self.party_b.receive_components(comp_A)
        loss = self.party_a.send_loss()
        return loss

    def predict(self, X_B):
        msg = self.party_b.predict(X_B)
        return self.party_a.predict(msg)
