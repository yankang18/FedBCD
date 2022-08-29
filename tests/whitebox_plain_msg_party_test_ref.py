import numpy as np
from federated_learning.encryption import encrypt_matrix, encrypt_matmul_3, decrypt_matrixes, decrypt_matrix, decrypt_array
from federated_learning.encryption import paillier

import tensorflow as tf


def party_A_compute_components(U_A, y, overlap_indexes):

    print("####> party A compute components")
    y_A_u_A = np.expand_dims(np.sum(y * U_A, axis=0) / len(y), axis=0)
    y_A_u_A_2 = np.matmul(y_A_u_A.transpose(), y_A_u_A)

    print("y_A_u_A shape:", y_A_u_A.shape)
    # print("y_A_u_A: \n", y_A_u_A)

    print("y_A_u_A_2 shape:", y_A_u_A_2.shape)
    # print("y_A_u_A_2: \n", y_A_u_A_2)

    y_overlap = y[overlap_indexes]
    y_overlap2 = y_overlap * y_overlap

    print("y_overlap2 shape:", y_overlap2.shape)
    # print("y_overlap2: \n", y_overlap2)

    y_overlap2_ex = np.expand_dims(y_overlap2, axis=2)
    print("y_overlap2_ex shape:", y_overlap2_ex.shape)
    # print("y_overlap2_ex: \n", y_overlap2_ex)

    print("--------> party A pass following three components to party B")
    comp_A_beta1 = 0.25 * y_overlap2_ex * y_A_u_A_2
    comp_A_beta2 = -0.5 * y_overlap * y_A_u_A
    U_A_overlap = U_A[overlap_indexes]
    print("comp_A_beta1 shape:", comp_A_beta1.shape)
    print("comp_A_beta1: \n", comp_A_beta1)
    print("comp_A_beta2 shape:", comp_A_beta2.shape)
    print("comp_A_beta2: \n", comp_A_beta2)
    mapping_comp_A = - U_A_overlap / U_A.shape[1]
    print("mapping_comp_A shape:", mapping_comp_A.shape)
    print("mapping_comp_A:", mapping_comp_A)
    return comp_A_beta1, comp_A_beta2, mapping_comp_A, y_overlap, y_overlap2, y_A_u_A


def party_B_compute_components(U_B, overlap_indexes):
    print("####> party B compute components")
    # party B received components from party A
    U_B_overlap = U_B[overlap_indexes]

    U_B_overlap_left = np.expand_dims(U_B_overlap, axis=2)
    print("U_B_overlap_left shape:", U_B_overlap_left.shape)
    # print("U_B_overlap_left: \n", U_B_overlap_left)
    U_B_overlap_right = np.expand_dims(U_B_overlap, axis=1)
    print("U_B_overlap_right shape:", U_B_overlap_right.shape)
    # print("U_B_overlap_right: \n", U_B_overlap_right)
    U_B_overlap_2 = np.matmul(U_B_overlap_left, U_B_overlap_right)

    print("--------> party B pass following three components to party A")
    print("U_B_overlap shape:", U_B_overlap.shape)
    # print("U_B_overlap: \n", U_B_overlap)
    print("U_B_overlap_2 shape:", U_B_overlap_2.shape)
    # print("U_B_overlap_2: \n", U_B_overlap_2)
    mapping_comp_B = - U_B_overlap / U_B.shape[1]
    print("mapping_comp_B shape:", mapping_comp_B.shape)
    return U_B_overlap, U_B_overlap_2, mapping_comp_B


def party_A_received_components_from_party_B_and_update_gradients(U_B_overlap, U_B_2_overlap, mapping_comp_B, y_overlap, y_overlap2, y_non_overlap, y_A_u_A, y):
    tmp = 0.25 * np.squeeze(np.matmul(np.expand_dims(y_overlap2 * y_A_u_A, axis=1), U_B_2_overlap), axis=1)
    const = (np.sum(tmp, axis=0) - 0.5 * np.sum(y_overlap * U_B_overlap, axis=0)) / len(y)
    grad_A_nonoverlap = const * y_non_overlap
    grad_A_overlap = const * y_overlap + mapping_comp_B
    grad_A = np.vstack((grad_A_overlap, grad_A_nonoverlap))
    print("grad_A shape:", grad_A.shape)
    print("grad_A: \n", grad_A)


def party_B_received_components_from_party_A_and_update_gradients(comp_A_beta1, comp_A_beta2, mapping_comp_A, U_B_overlap):

    print("####> party B received components from party A and update gradients")
    U_B_overlap_ex = np.expand_dims(U_B_overlap, axis=1)
    print("U_B_overlap_ex shape:", U_B_overlap_ex.shape)
    print("U_B_overlap_ex: \n", U_B_overlap_ex)

    # (2, 1, 3) x (2, 3, 3)
    U_B_grad_A_beta1 = np.matmul(U_B_overlap_ex, comp_A_beta1)
    print("U_B_grad_A_beta1 shape:", U_B_grad_A_beta1.shape)
    # print("U_B_grad_A_beta1: \n", U_B_grad_A_beta1)

    grad_B_part_1 = np.squeeze(U_B_grad_A_beta1, axis=1)
    print("grad_B_part_1 shape:", grad_B_part_1.shape)
    # print("grad_B_part_1: \n", grad_B_part_1)

    grad_B = grad_B_part_1 + comp_A_beta2 + mapping_comp_A
    print("grad_B shape:", grad_B.shape)
    print("grad_B: \n", grad_B)


if __name__ == '__main__':

    # pubkey, privatekey = paillier.generate_paillier_keypair(n_length=1024)

    U_A = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    U_B = np.array([[4, 2, 3],
                    [6, 5, 1],
                    [3, 4, 1]])
    y = np.array([[1],
                  [-1],
                  [1]])

    overlap_indexes = [1, 2]
    non_overlap_indexes = [0]
    print("----> Data")
    print("U_A shape:", U_A.shape)
    print("U_A: \n", U_A)
    print("U_B shape:", U_B.shape)
    print("U_B: \n", U_B)
    print("y：", y.shape)
    print("y： \n", y)

    comp_A_beta1, comp_A_beta2, mapping_comp_A, y_overlap, y_overlap2, y_A_u_A = party_A_compute_components(U_A, y, overlap_indexes)

    print("* " * 100)

    U_B_overlap, U_B_2, mapping_comp_B = party_B_compute_components(U_B, overlap_indexes)

    print("* " * 100)
    y_non_overlap = y[non_overlap_indexes]
    party_A_received_components_from_party_B_and_update_gradients(U_B_overlap, U_B_2, mapping_comp_B, y_overlap, y_overlap2, y_non_overlap, y_A_u_A, y)

    print("* " * 100)

    party_B_received_components_from_party_A_and_update_gradients(comp_A_beta1, comp_A_beta2, mapping_comp_A, U_B_overlap)

    print("* " * 100)


    # encrypt_grad_A_beta1 = encrypt_matrix(pubkey, comp_A_beta1)
    # encrypt_grad_A_beta2 = encrypt_matrix(pubkey, comp_A_beta2)
    # print("encrypt_grad_A_beta1 shape:", encrypt_grad_A_beta1.shape)
    # print("encrypt_grad_A_beta1: \n", encrypt_grad_A_beta1)
    # print("encrypt_grad_A_beta2 shape:", encrypt_grad_A_beta2.shape)
    # print("encrypt_grad_A_beta2: \n", encrypt_grad_A_beta2)
    #
    # encrypt_grad_B_part_1 = np.squeeze(np.array(encrypt_matmul_3(pubkey, U_B_overlap_ex, encrypt_grad_A_beta1)), axis=1)
    # print("encrypt_grad_B_part_1 shape:", encrypt_grad_B_part_1.shape)
    # print("encrypt_grad_B_part_1: \n", encrypt_grad_B_part_1)
    #
    # encrypt_grad_B = encrypt_grad_B_part_1 + encrypt_grad_A_beta2
    #
    # decrypt_grad_B_part_1 = np.array(decrypt_matrix(privatekey, encrypt_grad_B_part_1))
    # print("decrypt_grad_B_part_1 shape:", decrypt_grad_B_part_1.shape)
    # print("decrypt_grad_B_part_1: \n", decrypt_grad_B_part_1)
    #
    # encrypt_grad_B_part_1_sum = np.sum(encrypt_grad_B_part_1, axis=0)
    # print("encrypt_grad_B_part_1_sum shape:", encrypt_grad_B_part_1_sum.shape)
    # print("encrypt_grad_B_part_1_sum: \n", encrypt_grad_B_part_1_sum)
    #
    # decrypt_grad_B_part_1_sum = np.array(decrypt_array(privatekey, encrypt_grad_B_part_1_sum))
    # print("decrypt_grad_B_part_1_sum shape:", decrypt_grad_B_part_1_sum.shape)
    # print("decrypt_grad_B_part_1_sum: \n", decrypt_grad_B_part_1_sum)
    #
    # decrypy_grad_B = np.array(decrypt_matrix(privatekey, encrypt_grad_B))
    # print("decrypy_grad_B shape:", decrypy_grad_B.shape)
    # print("decrypy_grad_B: \n", decrypy_grad_B)




