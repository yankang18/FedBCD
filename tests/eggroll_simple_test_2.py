import numpy as np
from api.eggroll import parallelize
from federated_learning.encryption import paillier
from federated_learning.encryption import encrypt_matrix, decrypt_matrix, encrypt_array


if __name__ == '__main__':
    public_key, privatekey = paillier.generate_paillier_keypair(n_length=1024)
    X = np.array([[1, 2, 3],
                       [4, 5, 6],
                       [7, 8, 9],
                       [10, 11, 12],
                       [13, 14, 15],
                       [16, 17, 18],
                       [19, 20, 21]])

    print("matrix shape", X.shape)
    print("matrix: \n", X)

    print("r matrix shape", X.shape)
    print("r matrix \n", X)

    X = parallelize(X, partition=2)

    for i, item in enumerate(X.collect()):
        print(item, type(item))
        # encrypt_matrix(public_key, item)
        print("--")

    print("here 2")
    # X2 = X.mapValues(lambda x: encrypt_array(public_key, x))
    X2 = X.mapValues(lambda x: encrypt_matrix(public_key, x))
    # X2 = X.mapValues(lambda x: x)
    val = X2.collect()
    val = dict(val)
    print(val)

    # public_key, privatekey = paillier.generate_paillier_keypair(n_length=1024)
    # encrypt_matrix(public_key, np.array([2, 2, 2]))


