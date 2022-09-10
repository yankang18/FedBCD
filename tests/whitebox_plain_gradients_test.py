from vnn_demo.plain_ftl import PlainFTLGuestModel, PlainFTLHostModel
from tests.fake_models import FakeAutoencoder

import numpy as np


def msg_exchange_text():

    U_A = np.array([[1, 2, 3],
                    [4, 5, 6],
                    [7, 8, 9]])
    U_B = np.array([[4, 2, 3],
                    [6, 5, 1],
                    [3, 4, 1]])
    y = np.array([[1], [-1], [1]])

    overlap_indexes = [1, 2]
    non_overlap_indexes = [0]

    autoencoderA = FakeAutoencoder(0)
    autoencoderA.build(U_A.shape[1])
    autoencoderB = FakeAutoencoder(1)
    autoencoderB.build(U_B.shape[1])

    partyA = PlainFTLGuestModel(autoencoderA, 1)
    partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
    partyB = PlainFTLHostModel(autoencoderB, 1)
    partyB.set_batch(U_B, overlap_indexes)

    comp_A_beta1, comp_A_beta2, mapping_comp_A = partyA.send_components()

    print("comp_A_beta1 shape", comp_A_beta1.shape)
    print(comp_A_beta1)
    print("comp_A_beta2 shape", comp_A_beta2.shape)
    print(comp_A_beta2)
    print("mapping_comp_A shape", mapping_comp_A.shape)
    print(mapping_comp_A)

    U_B_overlap, U_B_overlap_2, mapping_comp_B = partyB.send_components()

    print("U_B_overlap shape", U_B_overlap.shape)
    print("U_B_overlap_2 shape", U_B_overlap_2.shape)
    print("mapping_comp_B shape", mapping_comp_B.shape)

    partyA.receive_components([U_B_overlap, U_B_overlap_2, mapping_comp_B])

    partyB.receive_components([comp_A_beta1, comp_A_beta2, mapping_comp_A])


def run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes):
    """

    :param autoencoderA:
    :param autoencoderB:
    :param U_A:
    :param U_B:
    :param y:
    :param overlap_indexes:
    :param non_overlap_indexes:

    :return:
    """

    partyA = PlainFTLGuestModel(autoencoderA, 1)
    partyA.set_batch(U_A, y, non_overlap_indexes, overlap_indexes)
    partyB = PlainFTLHostModel(autoencoderB, 1)
    partyB.set_batch(U_B, overlap_indexes)

    comp_A_beta1, comp_A_beta2, mapping_comp_A = partyA.send_components()

    print("comp_A_beta1 shape", comp_A_beta1.shape)
    # print(comp_A_beta1)
    print("comp_A_beta2 shape", comp_A_beta2.shape)
    # print(comp_A_beta2)
    print("mapping_comp_A shape", mapping_comp_A.shape)
    # print(mapping_comp_A)

    U_B_overlap, U_B_overlap_2, mapping_comp_B = partyB.send_components()

    print("U_B_overlap shape", U_B_overlap.shape)
    # print(U_B_overlap)
    print("U_B_overlap_2 shape", U_B_overlap_2.shape)
    # print(U_B_overlap_2)
    print("mapping_comp_A shape", mapping_comp_B.shape)
    # print(mapping_comp_B)

    partyA.receive_components([U_B_overlap, U_B_overlap_2, mapping_comp_B])
    # loss_grads_A = partyA.get_loss_grads()
    # loss = partyA.send_loss()

    partyB.receive_components([comp_A_beta1, comp_A_beta2, mapping_comp_A])
    # loss_grads_B = partyB.get_loss_grads()

    # return loss, loss_grads_A, loss_grads_B
    return partyA, partyB


def party_b_gradient_checking_test():

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

    Wh = np.ones((5, U_A.shape[1]))
    bh = np.zeros(U_A.shape[1])

    autoencoderA = FakeAutoencoder(0)
    autoencoderA.build(U_A.shape[1], Wh, bh)
    autoencoderB = FakeAutoencoder(1)
    autoencoderB.build(U_B.shape[1], Wh, bh)

    partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes)
    loss_grads_B_1 = partyB.get_loss_grads()
    loss1 = partyA.send_loss()

    # U_B_prime = np.array([[4, 2, 3],
    #                       [6, 5, 1.001],
    #                       [3, 4, 1]])
    U_B_prime = np.array([[4, 2, 3, 1, 2],
                          [6, 5, 1.001, 4, 5],
                          [7, 4, 1, 9, 10],
                          [6, 5, 1, 4, 5]])

    partyA, partyB = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B_prime, y, overlap_indexes, non_overlap_indexes)
    loss_grads_B_2 = partyB.get_loss_grads()
    loss2 = partyA.send_loss()

    print("loss_grads_B_1", loss_grads_B_1)
    print("loss_grads_B_2", loss_grads_B_2)
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

    Wh = np.ones((5, U_A.shape[1]))
    bh = np.zeros(U_A.shape[1])

    autoencoderA = FakeAutoencoder(0)
    autoencoderA.build(U_A.shape[1], Wh, bh)
    autoencoderB = FakeAutoencoder(1)
    autoencoderB.build(U_B.shape[1], Wh, bh)

    partyA, _ = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A, U_B, y, overlap_indexes, non_overlap_indexes)
    loss_grads_A_1 = partyA.get_loss_grads()
    loss1 = partyA.send_loss()

    U_A_prime = np.array([[1, 2, 3, 4, 5],
                          [4, 5.001, 6, 7, 8],
                          [7, 8, 9, 10, 11],
                          [4, 5, 6, 7, 8]])

    partyA, _ = run_one_party_msg_exchange(autoencoderA, autoencoderB, U_A_prime, U_B, y, overlap_indexes, non_overlap_indexes)
    loss_grads_A_2 = partyA.get_loss_grads()
    loss2 = partyA.send_loss()

    print("loss_grads_A_1 \n", loss_grads_A_1)
    print("loss_grads_A_2 \n", loss_grads_A_2)

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

    print("party a checking: \n")
    party_a_gradient_checking_test()
    # print("party b checking: \n")
    # party_b_gradient_checking_test()

