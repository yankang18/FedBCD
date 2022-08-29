import numpy as np
from federated_learning.encryption import paillier
from federated_learning.eggroll_computation.helper import encrypt_matrix
from eggroll_encryption.encryption_tests import PaillierEncrypt


class Example(object):

    def __init__(self):
        self.publickey, self.privatekey = paillier.generate_paillier_keypair(n_length=1024)
        self.paillierEncrypt = PaillierEncrypt()
        self.paillierEncrypt.set_public_key(self.publickey)
        self.paillierEncrypt.set_privacy_key(self.privatekey)

    def test1(self):
        matrix = np.array([[1, 2, 3],
                           [4, 5, 6],
                           [7, 8, 9],
                           [10, 11, 12],
                           [13, 14, 15],
                           [16, 17, 18],
                           [19, 20, 21]], dtype=np.float64)

        print("matrix shape", matrix.shape)
        print("matrix: \n", matrix)

        self._test(matrix)

    def test2(self):
        matrix = np.array([[[33, 22, 31],
                            [14, 15, 16],
                            [17, 18, 19]],
                           [[10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]]])

        print("matrix shape", matrix.shape)
        print("matrix: \n", matrix)

        self._test(matrix)

    def test3(self):
        matrix = np.array([[[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9]],
                           [[10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]],
                           [[11, 14, 12],
                            [13, 12, 15],
                            [17, 19, 20]]
                           ])

        print("matrix shape", matrix.shape)
        print("matrix: \n", matrix)

        self._test(matrix)

    def test4(self):
        matrix = np.ones((8, 50, 50))

        # print("matrix shape", matrix.shape)
        # print("matrix: \n", matrix)

        self._test(matrix)

    def _test(self, matrix):

        result = encrypt_matrix(self.publickey, matrix)
        # X = prepare_table(matrix, 1, 1)
        # result = self.paillierEncrypt.distribute_encrypt(X)
        print("encrypted_result shape", result.shape)
        print("encrypted_result: \n", result)
        print("encrypted_result type: \n", type(result))

        # decrypted_result = decrypt_matrix(self.privatekey, result)
        # decrypted_result = encrypt_matrix(self.privatekey, result, is_encryption=False)
        decrypted_result = self.paillierEncrypt.distribute_decrypt(result)
        print("decrypted_result shape", decrypted_result.shape)
        print("decrypted_result \n", decrypted_result)


if __name__ == '__main__':
    example = Example()

    print("-"*100, "test1")
    example.test1()

    # print("-"*100, "test2")
    # example.test2()

    # print("-"*100, "test3")
    # example.test3()

    # print("-"*100, "test4")
    # example.test4()


