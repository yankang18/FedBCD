import time

import numpy as np

from federated_learning.encryption import encryption
# from federated_learning.eggroll_computation.helper import encrypt_matrix
from federated_learning.encryption import paillier


class Example(object):

    def __init__(self):
        self.publickey, self.privatekey = paillier.generate_paillier_keypair(n_length=2048)

    # def encrypt_2d_matrix(self, X):
    #     encrypt_X = [[0 for _ in range(X.shape[1])] for _ in range(X.shape[0])]
    #     for i in range(len(encrypt_X)):
    #         temp = []
    #         for j in range(X.shape[-1]):
    #             temp.append(self.publickey.encrypt(X[i, j]))
    #         encrypt_X[i] = temp
    #
    #     encrypt_X = np.array(encrypt_X)
    #     print(encrypt_X.shape)
    #     print(encrypt_X)
    #     return encrypt_X
    #
    # def encrypt_3d_matrix(self, X):
    #     encrypt_X = [[[0 for _ in range(X.shape[-1])] for _ in range(X.shape[1])] for _ in range(X.shape[0])]
    #     for i in range(X.shape[0]):
    #         second_dim_list = []
    #         for j in range(X.shape[1]):
    #             third_dim_list = []
    #             for z in range(X.shape[-1]):
    #                 third_dim_list.append(self.publickey.encrypt(X[i, j, z]))
    #             second_dim_list.append(third_dim_list)
    #         encrypt_X[i] = second_dim_list
    #     return np.array(encrypt_X)

    def test1(self):
        # X = np.ones((50, 50))

        # var = 10
        # curr_time1 = time.time()
        # self.publickey.encrypt(var)
        # encryption.encrypt_matrix(self.publickey, X)
        # curr_time2 = time.time()
        # print("sequence running time:", (curr_time2 - curr_time1))

        # encrypt_matrix(self.publickey, X)
        # curr_time3 = time.time()
        # print("distributed running time:", (curr_time3 - curr_time2))

        left = np.ones((100, 10))
        mid = np.ones((10, 10))
        right = np.ones((10, 100))
        mat = np.matmul(np.matmul(left, mid), right)

        temp = np.ones((10, 10))

        #
        start = time.time()
        mat = encryption.encrypt_matrix(self.publickey, mat)
        end = time.time()
        print(mat.shape)
        print("big table enc time:", (end - start))
        print("enc mat:", mat.shape)

        #
        start = time.time()
        enc_mid = encryption.encrypt_matrix(self.publickey, mid)
        for i in range(7):
            temp * enc_mid
        end = time.time()
        print("mid table enc time:", (end - start))

    # def test2(self):
    #
    #     X = np.array([[[1, 2, 3]],
    #                   [[10, 11, 12]]], dtype=np.float64)
    #     Y = np.array([[[10, 11, 12],
    #                   [13, 14, 15],
    #                   [16, 17, 18]],
    #                   [[19, 20, 21],
    #                   [22, 23, 24],
    #                   [25, 26, 27]]], dtype=np.float64)
    #
    #     Z = np.matmul(X, Y)
    #     print("Z shape \n", Z.shape)
    #     print("Z \n", Z)
    #
    #     # encrypt_X = self.encrypt(X)
    #     encrypt_Y = encrypt_matrix(Y)
    #
    #     # res = self.encrypt_matmul_3(encrypt_X, Y)
    #     res = encrypt_matmul_3(X, encrypt_Y)
    #     print(res)
    #
    #     # decrypt res
    #     decrypt_res = decrypt_matrix(self.privatekey, res)
    #     print("res shape", res.shape)
    #     print("res \n", res)
    #     print("decrypt_res \n", decrypt_res)

    # def test3(self):
    #
    #     X = np.array([[[1, 2, 3],
    #                   [4, 5, 6],
    #                   [7, 8, 9]],
    #                   [[10, 11, 12],
    #                   [13, 14, 15],
    #                   [16, 17, 18]]], dtype=np.float64)
    #     Y = np.array([[[10, 11, 12],
    #                   [13, 14, 15],
    #                   [16, 17, 18]],
    #                   [[19, 20, 21],
    #                   [22, 23, 24],
    #                   [25, 26, 27]]], dtype=np.float64)
    #
    #     print("X shape", X.shape)
    #     print("Y shape", Y.shape)
    #
    #     Z = np.matmul(X, Y)
    #     print("Z shape \n", Z.shape)
    #     print("Z \n", Z)
    #
    #     # encrypt_X = self.encrypt(X)
    #     encrypt_Y = self.encrypt_3d_matrix(Y)
    #
    #     # res = self.encrypt_matmul_3(encrypt_X, Y)
    #     res = encrypt_matmul_3(X, encrypt_Y)
    #     print(res)
    #
    #     # decrypt res
    #     decrypt_res = decrypt_matrix(self.privatekey, res)
    #     print("res shape", res.shape)
    #     print("res \n", res)
    #     print("decrypt_res \n", decrypt_res)


if __name__ == '__main__':
    example = Example()
    print("-" * 20)
    example.test1()



