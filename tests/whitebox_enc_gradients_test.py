from federated_learning.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel
from federated_learning.tests.fake_models import FakeAutoencoder
from federated_learning.tests.whitebox_plain_gradients_test import run_one_party_msg_exchange
from federated_learning.encryption import paillier
from federated_learning.encryption import encryption
import numpy as np

public_key, private_key = paillier.generate_paillier_keypair(n_length=1024)


def party_b_gradient_checking_test():

    # U_A = np.array([[1, 2, 3],
    #                 [4, 5, 6],
    #                 [7, 8, 9],
    #                 [1, 3, 4]])
    # U_B = np.array([[4, 2, 3],
    #                 [6, 5, 1],
    #                 [3, 4, 1],
    #                 [1, 3, 4]])
    # y = np.array([[1], [-1], [1], [1]])
    #
    # overlap_indexes = [1, 2]
    # non_overlap_indexes = [0, 3]

    U_A = np.array([[1, 2, 3, 4, 5],
                    [4, 5, 6, 7, 8],
                    [7, 8, 9, 10, 11],
                    [4, 5, 6, 7, 8]])
    U_B = np.array([[4, 2, 3, 1, 2],
                    [6, 5, 1, 4, 5],
                    [7, 4, 1, 9, 10],
                    [6, 5, 1, 4, 5]])
    y = np.array([[1], [-1], [1], [-1]])

    overlap_indexes = [1, 2]
    non_overlap_indexes = [0, 3]

    Wh = np.ones((4, U_A.shape[1]))
    bh = np.zeros(U_A.shape[1])

    autoencoderA = FakeAutoencoder(0)
    autoencoderA.build(U_A.shape[1], Wh, bh)
    autoencoderB = FakeAutoencoder(1)
    autoencoderB.build(U_B.shape[1], Wh, bh)

    partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes, public_key, private_key, True)
    loss_grads_B_1 = partyB.get_loss_grads()
    loss1 = partyA.send_loss()
    encrypt_grads_W1, encrypt_grads_b1 = partyB.send_gradients()
    print("encrypt_grads_W1 shape", encrypt_grads_W1.shape)
    print(encrypt_grads_W1)
    print("encrypt_grads_b1 shape", encrypt_grads_b1.shape)
    print(encrypt_grads_b1)

    # U_B_prime = np.array([[4, 2, 3],
    #                       [6, 5, 1.001],
    #                       [3, 4, 1]])
    U_B_prime = np.array([[4, 2, 3, 1, 2],
                          [6, 5, 1.001, 4, 5],
                          [7, 4, 1, 9, 10],
                          [6, 5, 1, 4, 5]])
    partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B_prime, y, overlap_indexes, non_overlap_indexes, public_key, private_key, True)
    loss_grads_B_2 = partyB.get_loss_grads()
    loss2 = partyA.send_loss()
    encrypt_grads_W2, encrypt_grads_b2 = partyB.send_gradients()
    print("encrypt_grads_W2 shape", encrypt_grads_W2.shape)
    print(encrypt_grads_W2)
    print("encrypt_grads_b2 shape", encrypt_grads_b2.shape)
    print(encrypt_grads_b2)

    loss_grads_B_1 = np.array(encryption.decrypt_matrix(private_key, loss_grads_B_1))
    loss_grads_B_2 = np.array(encryption.decrypt_matrix(private_key, loss_grads_B_2))

    print("loss_grads_B_1", loss_grads_B_1)
    print("loss_grads_B_2", loss_grads_B_2)

    print("encrypt loss1", loss1)
    print("encrypt loss2", loss2)

    loss1 = encryption.decrypt(private_key, loss1)
    loss2 = encryption.decrypt(private_key, loss2)

    print("loss2", loss2)
    print("loss1", loss1)

    print("(loss2 - loss1)", (loss2 - loss1))
    grad_approx = (loss2 - loss1) / 0.001
    grad_real = loss_grads_B_1[0, 2]
    grad_diff = np.abs(grad_approx - grad_real)
    print("grad_approx", grad_approx)
    print("grad_real", grad_real)
    print("grad diff", grad_diff)
    assert grad_diff < 0.001


def party_a_gradient_checking_test():

    # U_A = np.array([[1, 2, 3],
    #                 [4, 5, 6],
    #                 [7, 8, 9]])
    # U_B = np.array([[4, 2, 3],
    #                 [6, 5, 1],
    #                 [3, 4, 1]])
    # y = np.array([[1], [-1], [1]])
    #
    # overlap_indexes = [1, 2]
    # non_overlap_indexes = [0]

    U_A = np.array([[1, 2, 3, 4, 5],
                    [4, 5, 6, 7, 8],
                    [7, 8, 9, 10, 11],
                    [4, 5, 6, 7, 8]])
    U_B = np.array([[4, 2, 3, 1, 2],
                    [6, 5, 1, 4, 5],
                    [7, 4, 1, 9, 10],
                    [6, 5, 1, 4, 5]])
    y = np.array([[1], [-1], [1], [-1]])

    overlap_indexes = [1, 2]
    non_overlap_indexes = [0, 3]

    Wh = np.ones((4, U_A.shape[1]))
    bh = np.zeros(U_A.shape[1])

    autoencoderA = FakeAutoencoder(0)
    autoencoderA.build(U_A.shape[1], Wh, bh)
    autoencoderB = FakeAutoencoder(1)
    autoencoderB.build(U_B.shape[1], Wh, bh)

    partyA, _ = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes, public_key, private_key, True)
    loss_grads_A_1 = partyA.get_loss_grads()
    loss1 = partyA.send_loss()
    encrypt_grads_W1, encrypt_grads_b1 = partyA.send_gradients()
    print("encrypt_grads_W1 shape", encrypt_grads_W1.shape)
    print(encrypt_grads_W1)
    print("encrypt_grads_b1 shape", encrypt_grads_b1.shape)
    print(encrypt_grads_b1)

    # U_A_prime = np.array([[1, 2, 3],
    #                       [4, 5.001, 6],
    #                       [7, 8, 9]])
    U_A_prime = np.array([[1, 2, 3, 4, 5],
                          [4, 5.001, 6, 7, 8],
                          [7, 8, 9, 10, 11],
                          [4, 5, 6, 7, 8]])
    partyA, _ = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A_prime, U_B, y, overlap_indexes, non_overlap_indexes, public_key, private_key, True)
    loss_grads_A_2 = partyA.get_loss_grads()
    loss2 = partyA.send_loss()
    encrypt_grads_W2, encrypt_grads_b2 = partyA.send_gradients()
    print("encrypt_grads_W2 shape", encrypt_grads_W2.shape)
    print(encrypt_grads_W2)
    print("encrypt_grads_b2 shape", encrypt_grads_b2.shape)
    print(encrypt_grads_b2)

    loss_grads_A_1 = np.array(encryption.decrypt_matrix(private_key, loss_grads_A_1))
    loss_grads_A_2 = np.array(encryption.decrypt_matrix(private_key, loss_grads_A_2))

    print("loss_grads_A_1 \n", loss_grads_A_1)
    print("loss_grads_A_2 \n", loss_grads_A_2)

    loss1 = encryption.decrypt(private_key, loss1)
    loss2 = encryption.decrypt(private_key, loss2)

    print("loss2", loss2)
    print("loss1", loss1)
    print("(loss2 - loss1)", (loss2 - loss1))
    grad_approx = (loss2 - loss1) / 0.001
    grad_real = loss_grads_A_1[1, 1]
    grad_diff = np.abs(grad_approx - grad_real)
    print("grad_approx", grad_approx)
    print("grad_real", grad_real)
    print("grad diff", grad_diff)
    assert grad_diff < 0.001


if __name__ == '__main__':

    # print("party a checking: \n")
    # party_a_gradient_checking_test()
    print("party b checking: \n")
    party_b_gradient_checking_test()
