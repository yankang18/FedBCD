import numpy as np
from federated_learning.encryption import decrypt_matrix
from federated_learning.eggroll_computation.helper import encrypt_matmul_2, encrypt_matmul_3, compute_avg_XY
from federated_learning.encryption import paillier


class Example(object):

    def __init__(self):
        self.publickey, self.privatekey = paillier.generate_paillier_keypair(n_length=1024)

    def encrypt_2d_matrix(self, X):
        encrypt_X = [[0 for _ in range(X.shape[1])] for _ in range(X.shape[0])]
        for i in range(len(encrypt_X)):
            temp = []
            for j in range(X.shape[-1]):
                temp.append(self.publickey.encrypt(X[i, j]))
            encrypt_X[i] = temp

        encrypt_X = np.array(encrypt_X)
        print(encrypt_X.shape)
        print(encrypt_X)
        return encrypt_X

    def encrypt_3d_matrix(self, X):
        encrypt_X = [[[0 for _ in range(X.shape[-1])] for _ in range(X.shape[1])] for _ in range(X.shape[0])]
        for i in range(X.shape[0]):
            second_dim_list = []
            for j in range(X.shape[1]):
                third_dim_list = []
                for z in range(X.shape[-1]):
                    third_dim_list.append(self.publickey.encrypt(X[i, j, z]))
                second_dim_list.append(third_dim_list)
            encrypt_X[i] = second_dim_list
        return np.array(encrypt_X)

    def test1(self):

        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float64)
        Y = np.array([[10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]], dtype=np.float64)

        Z = np.matmul(X, Y)
        print("Z \n", Z)

        encrypt_Y = self.encrypt_2d_matrix(Y)
        res = encrypt_matmul_2(X, encrypt_Y)

        # decrypt res
        decrypt_res = decrypt_matrix(self.privatekey, res)
        print("res shape", res.shape)
        print("res \n", res)
        print("decrypt_res \n", decrypt_res)

    def test2(self):

        X = np.array([[[1, 2, 3]],
                      [[10, 11, 12]]], dtype=np.float64)
        Y = np.array([[[10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]],
                      [[19, 20, 21],
                      [22, 23, 24],
                      [25, 26, 27]]], dtype=np.float64)

        Z = np.matmul(X, Y)
        print("Z shape \n", Z.shape)
        print("Z \n", Z)

        # encrypt_X = self.encrypt(X)
        encrypt_Y = self.encrypt_3d_matrix(Y)

        # res = self.encrypt_matmul_3(encrypt_X, Y)
        res = encrypt_matmul_3(X, encrypt_Y)
        print(res)

        # decrypt res
        decrypt_res = decrypt_matrix(self.privatekey, res)
        print("res shape", res.shape)
        print("res \n", res)
        print("decrypt_res \n", decrypt_res)

    def test3(self):

        X = np.array([[[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]],
                      [[10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]]], dtype=np.float64)
        Y = np.array([[[10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]],
                      [[19, 20, 21],
                      [22, 23, 24],
                      [25, 26, 27]]], dtype=np.float64)

        print("X shape", X.shape)
        print("Y shape", Y.shape)

        Z = np.matmul(X, Y)
        print("Z shape \n", Z.shape)
        print("Z \n", Z)

        # encrypt_X = self.encrypt(X)
        encrypt_Y = self.encrypt_3d_matrix(Y)

        # res = self.encrypt_matmul_3(encrypt_X, Y)
        res = encrypt_matmul_3(X, encrypt_Y)
        print(res)

        # decrypt res
        decrypt_res = decrypt_matrix(self.privatekey, res)
        print("res shape", res.shape)
        print("res \n", res)
        print("decrypt_res \n", decrypt_res)

    def test4(self):

        X = np.array([[[1, 2, 3]],
                      [[10, 11, 12]]], dtype=np.float64)
        Y = np.array([[[10, 11, 12],
                      [13, 14, 15],
                      [16, 17, 18]],
                      [[19, 20, 21],
                      [22, 23, 24],
                      [25, 26, 27]]], dtype=np.float64)

        print("X shape", X.shape)
        print("Y shape", Y.shape)

        Z = np.matmul(X, Y)
        print("Z shape \n", Z.shape)
        print("Z \n", Z)

        # encrypt_X = self.encrypt(X)
        encrypt_Y = self.encrypt_3d_matrix(Y)

        # res = self.encrypt_matmul_3(encrypt_X, Y)
        res = encrypt_matmul_3(X, encrypt_Y)
        print(res)

        # decrypt res
        decrypt_res = decrypt_matrix(self.privatekey, res)
        print("res shape", res.shape)
        print("res \n", res)
        print("decrypt_res \n", decrypt_res)

    def test5(self):
        X = np.array([[1, 2, 3],
                      [4, 5, 6],
                      [7, 8, 9]], dtype=np.float64)
        Y = np.array([[1], [-1], [1]])
        print("X shape", X.shape)
        print("Y shape", Y.shape, Y)

        Y = np.tile(Y, (1, X.shape[-1]))
        print("Y shape", Y.shape, Y)

        # np.sum(self.y_overlap * self.U_B_overlap, axis=0)) / len(self.y)
        actual1 = np.sum(Y * X, axis=0)/len(Y)
        actual2 = np.sum(X * Y, axis=0)/len(Y)
        predict1 = compute_avg_XY(X, Y)
        predict2 = compute_avg_XY(Y, X)
        print(actual1)
        print(actual2)
        print(predict1)
        print(predict2)


if __name__ == '__main__':
    example = Example()
    print("-"*20)
    example.test1()
    print("-"*20)
    example.test2()
    print("-"*20)
    example.test3()
    print("-"*20)
    example.test4()
    print("-"*20)
    example.test5()


