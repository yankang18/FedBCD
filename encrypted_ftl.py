import numpy as np

from eggroll_computation.helper import compute_sum_XY, \
    compute_XY, encrypt_matrix, compute_XY_plus_Z, \
    encrypt_matmul_2, encrypt_matmul_3, compute_X_plus_Y
from encryption.encryption import decrypt_array, decrypt_matrix, decrypt_scalar
from models.base_model import BaseModel
from plain_ftl import PlainFTLGuestModel, PlainFTLHostModel


class EncryptedFTLGuestModel(PlainFTLGuestModel):

    def __init__(self, local_model, alpha, public_key=None, private_key=None):
        super(EncryptedFTLGuestModel, self).__init__(local_model, alpha)
        self.feature_dim = local_model.get_representation_dim()
        self.public_key = public_key
        self.private_key = private_key
        self.is_trace = False

    def set_public_key(self, public_key):
        self.public_key = public_key

    def set_private_key(self, private_key):
        self.private_key = private_key

    def send_components(self):
        components = super(EncryptedFTLGuestModel, self).send_components()
        return self.__encrypt_components(components)

    def __encrypt_components(self, components):
        print("comp_A_beta1")
        encrypt_comp_0 = encrypt_matrix(self.public_key, components[0])
        print("encrypt_comp_A2")
        encrypt_comp_1 = encrypt_matrix(self.public_key, components[1])
        print("encrypt_mapping_comp_A")
        encrypt_comp_2 = encrypt_matrix(self.public_key, components[2])
        return [encrypt_comp_0, encrypt_comp_1, encrypt_comp_2]

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
        tmp1 = encrypt_matmul_3(y_overlap2_y_A_u_A, self.U_B_2_overlap)
        tmp2 = 0.25 * np.squeeze(tmp1, axis=1)

        if self.is_trace:
            print("tmp1 shape", tmp1.shape)
            print("tmp2 shape", tmp2.shape)
            print("tmp2", decrypt_matrix(self.private_key, tmp2))
        # tmp = 0.25 * np.squeeze(encrypt_matmul_3(self.public_key, y_overlap2_y_A_u_A, self.U_B_2_overlap), axis=1)

        y_overlap = np.tile(self.y_overlap, (1, self.U_B_overlap.shape[-1]))

        if self.is_trace:
            print("*** y_overlap", y_overlap)
            print("*** self.y_overlap", self.y_overlap)
            print("*** self.U_B_overlap", decrypt_matrix(self.private_key, self.U_B_overlap))
        # part2 = distributed_calculate_avg_XY(y_overlap, self.U_B_overlap)
        part2 = compute_sum_XY(y_overlap * 0.5, self.U_B_overlap)

        if self.is_trace:
            print("part2 shape", part2.shape)
            print("*** part2 \n", decrypt_matrix(self.private_key, part2))

        encrypt_const = np.sum(tmp2, axis=0) - part2
        # encrypt_const = (np.sum(tmp2, axis=0) - 0.5 * np.sum(self.y_overlap * self.U_B_overlap, axis=0)) / len(self.y)
        # encrypt_const has shape (1, feature_dim)

        encrypt_const_overlap = np.tile(encrypt_const, (len(self.overlap_indexes), 1))
        encrypt_const_nonoverlap = np.tile(encrypt_const, (len(self.non_overlap_indexes), 1))
        y_non_overlap = np.tile(self.y[self.non_overlap_indexes], (1, self.U_B_overlap.shape[-1]))

        if self.is_trace:
            print("y_non_overlap shape", y_non_overlap.shape)
            print("encrypt_const_overlap shape", encrypt_const_overlap.shape)
            print("encrypt_const_nonoverlap shape", encrypt_const_nonoverlap.shape)

        encrypt_grad_A_nonoverlap = compute_XY(self.alpha * y_non_overlap / len(self.y), encrypt_const_nonoverlap)
        encrypt_grad_A_overlap = compute_XY_plus_Z(self.alpha * y_overlap / len(self.y), encrypt_const_overlap, self.mapping_comp_B)

        if self.is_trace:
            print("# encrypt_const shape:", encrypt_const.shape)
            print("# encrypt_grad_A_nonoverlap", encrypt_grad_A_nonoverlap.shape)
            print("# encrypt_grad_A_overlap", encrypt_grad_A_overlap.shape)
            decrypt_const = decrypt_array(self.private_key, encrypt_const)
            decrypt_grad_A_nonoverlap = decrypt_matrix(self.private_key, encrypt_grad_A_nonoverlap)
            decrypt_grad_A_overlap = decrypt_matrix(self.private_key, encrypt_grad_A_overlap)
            print("*** decrypt_const: \n", decrypt_const)
            print("*** decrypt_grad_A_nonoverlap \n", decrypt_grad_A_nonoverlap)
            print("*** decrypt_grad_A_overlap \n", decrypt_grad_A_overlap)

        encrypt_grad_loss_A = [[0 for _ in range(self.U_B_overlap.shape[1])] for _ in range(len(self.y))]
        # TODO: need more efficient way to do following task
        for i, j in enumerate(self.non_overlap_indexes):
            encrypt_grad_loss_A[j] = encrypt_grad_A_nonoverlap[i]
        for i, j in enumerate(self.overlap_indexes):
            encrypt_grad_loss_A[j] = encrypt_grad_A_overlap[i]

        encrypt_grad_loss_A = np.array(encrypt_grad_loss_A)

        if self.is_trace:
            print("encrypt_grad_loss_A shape", encrypt_grad_loss_A.shape)
            print("encrypt_grad_loss_A", encrypt_grad_loss_A)

        self.loss_grads = encrypt_grad_loss_A
        grads = self.localModel.compute_gradients(self.X)
        self.encrypt_grads_W, self.encrypt_grads_b = self.__compute_encrypt_grads(grads, encrypt_grad_loss_A)

    def __compute_encrypt_grads(self, grads, encrypt_grads):

        grads_W = grads[0]
        grads_b = grads[1]
        encrypt_grads_ex = np.expand_dims(encrypt_grads, axis=1)

        if self.is_trace:
            print("grads_W shape", grads_W.shape)
            print("grads_b shape", grads_b.shape)
            print("encrypt_grads_ex shape", encrypt_grads_ex.shape)

        encrypt_grads_W = compute_sum_XY(encrypt_grads_ex, grads_W)
        encrypt_grads_b = compute_sum_XY(encrypt_grads, grads_b)
        # encrypt_grads_W = np.sum(encrypt_grads_ex * grads_W, axis=0)
        # encrypt_grads_b = np.sum(encrypt_grads * grads_b, axis=0)

        if self.is_trace:
            print("encrypt_grads_W shape", encrypt_grads_W.shape)
            print("encrypt_grads_b shape", encrypt_grads_b.shape)

        return encrypt_grads_W, encrypt_grads_b

    def send_gradients(self):
        return self.encrypt_grads_W, self.encrypt_grads_b

    def receive_gradients(self, gradients):
        self.localModel.apply_gradients(gradients)

    def send_loss(self):
        return self.loss

    def receive_loss(self, loss):
        self.loss = loss

    def __update_loss(self):
        U_A_overlap_prime = - self.U_A_overlap / self.feature_dim
        loss_overlap = np.sum(compute_sum_XY(U_A_overlap_prime, self.U_B_overlap))
        loss_Y = self.__compute_encrypt_loss_y(self.U_B_overlap, self.U_B_2_overlap, self.y_overlap, self.y_A_u_A)

        if self.is_trace:
            print("loss_Y", loss_Y)

        self.loss = self.alpha * loss_Y + loss_overlap

    def __compute_encrypt_loss_y(self, encrypt_U_B_overlap, encrypt_U_B_2_overlap, y_overlap, y_A_u_A):

        if self.is_trace:
            print("_calculate_encrypt_loss_y")
            print("encrypt_U_B_overlap shape", encrypt_U_B_overlap.shape)
            # print("encrypt_U_B_overlap", encrypt_U_B_overlap)
            print("y_A_u_A shape", y_A_u_A.shape)
            # print("y_A_u_A", y_A_u_A)

        encrypt_UB_yAuA = encrypt_matmul_2(encrypt_U_B_overlap, y_A_u_A.transpose())
        encrypt_UB_T_UB = np.sum(encrypt_U_B_2_overlap, axis=0)
        wx2 = encrypt_matmul_2(encrypt_matmul_2(y_A_u_A, encrypt_UB_T_UB), y_A_u_A.transpose())
        encrypt_loss_Y = (-0.5 * compute_sum_XY(y_overlap, encrypt_UB_yAuA)[0] + 1.0 / 8 * np.sum(wx2)) + len(y_overlap) * np.log(2)
        return encrypt_loss_Y

    def get_loss_grads(self):
        return self.loss_grads

    def predict(self, U_B, msg=None):
        return encrypt_matmul_2(U_B, self.y_A_u_A.transpose())


class EncryptedFTLHostModel(PlainFTLHostModel):

    def __init__(self, local_model, alpha, public_key=None, private_key=None):
        super(EncryptedFTLHostModel, self).__init__(local_model, alpha)
        self.localModel = local_model
        self.feature_dim = local_model.get_representation_dim()
        self.alpha = alpha
        self.public_key = public_key
        self.private_key = private_key

    def set_batch(self, X, overlap_indexes):
        self.X = X
        self.overlap_indexes = overlap_indexes

    def send_components(self):
        components = super(EncryptedFTLHostModel, self).send_components()
        return self.__encrypt_components(components)

    def __encrypt_components(self, components):
        encrypt_UB_1 = encrypt_matrix(self.public_key, components[0])
        encrypt_UB_2 = encrypt_matrix(self.public_key, components[1])
        encrypt_mapping_comp_B = encrypt_matrix(self.public_key, components[2])
        return [encrypt_UB_1, encrypt_UB_2, encrypt_mapping_comp_B]

    def receive_components(self, components):
        self.comp_A_beta1 = components[0]
        self.comp_A_beta2 = components[1]
        self.mapping_comp_A = components[2]
        self.__update_gradients()

    def __update_gradients(self):
        U_B_overlap_ex = np.expand_dims(self.U_B_overlap, axis=1)
        grads = self.localModel.compute_gradients(self.X[self.overlap_indexes])

        encrypted_U_B_comp_A_beta1 = encrypt_matmul_3(U_B_overlap_ex, self.comp_A_beta1)
        encrypted_grad_l1_B = compute_X_plus_Y(np.squeeze(encrypted_U_B_comp_A_beta1, axis=1), self.comp_A_beta2)
        encrypted_grad_loss_B = compute_X_plus_Y(self.alpha * encrypted_grad_l1_B, self.mapping_comp_A)

        self.loss_grads = encrypted_grad_loss_B
        self.encrypt_grads_W, self.encrypt_grads_b = self.__compute_encrypt_grads(grads, encrypted_grad_loss_B)

    def __compute_encrypt_grads(self, grads, encrypt_grads):
        grads_W = grads[0]
        grads_b = grads[1]
        encrypt_grads_W = compute_sum_XY(np.expand_dims(encrypt_grads, axis=1), grads_W)
        encrypt_grads_b = compute_sum_XY(encrypt_grads, grads_b)
        return encrypt_grads_W, encrypt_grads_b

    def send_gradients(self):
        return self.encrypt_grads_W, self.encrypt_grads_b

    def receive_gradients(self, gradients):
        self.localModel.apply_gradients(gradients)

    def get_loss_grads(self):
        return self.loss_grads

    def predict(self, X, msg=None):
        logits = self.localModel.transform(X)
        return encrypt_matrix(self.public_key, logits)


class LocalEncryptedFederatedTransferLearning(BaseModel):

    def __init__(self, guest: EncryptedFTLGuestModel, host: EncryptedFTLHostModel, private_key=None):
        super(LocalEncryptedFederatedTransferLearning, self).__init__()
        self.guest = guest
        self.host = host
        self.private_key = private_key

    def fit(self, X_A, X_B, y, overlap_indexes, non_overlap_indexes):

        self.guest.set_batch(X_A, y, non_overlap_indexes, overlap_indexes)
        self.host.set_batch(X_B, overlap_indexes)

        # For training the federated model.
        comp_B = self.host.send_components()
        comp_A = self.guest.send_components()

        self.guest.receive_components(comp_B)
        self.host.receive_components(comp_A)

        encrypt_gradients_A = self.guest.send_gradients()
        encrypt_gradients_B = self.host.send_gradients()

        self.guest.receive_gradients(self.__decrypt_gradients(encrypt_gradients_A))
        self.host.receive_gradients(self.__decrypt_gradients(encrypt_gradients_B))

        encrypt_loss = self.guest.send_loss()
        loss = self.__decrypt_loss(encrypt_loss)

        return loss

    def predict(self, X_B):
        msg = self.host.predict(X_B)
        return self.__decrypt_prediction(self.guest.predict(msg))

    def __decrypt_gradients(self, encrypt_gradients):
        return decrypt_matrix(self.private_key, encrypt_gradients[0]), decrypt_array(self.private_key, encrypt_gradients[1])

    def __decrypt_loss(self, encrypt_loss):
        return decrypt_scalar(self.private_key, encrypt_loss)

    def __decrypt_prediction(self, encrypt_prediction):
        return decrypt_matrix(self.private_key, encrypt_prediction)
