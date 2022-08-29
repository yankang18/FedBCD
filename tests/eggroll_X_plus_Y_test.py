import numpy as np
from federated_learning.eggroll_computation.helper import compute_X_plus_Y
from federated_learning.tests.utils import assert_matrix
import unittest


class TestSum(unittest.TestCase):

    def test_distributed_calculate_X_plus_Y_1(self):
        X = np.array([[1., 2., 3.],
                      [14., 5., 6.],
                      [7., 8., 9.]])

        Y = np.array([[1], [-1], [1]])

        real_X_plus_Y = X + Y
        X_plus_Y = compute_X_plus_Y(X, Y)
        print("X_plus_Y \n", X_plus_Y.shape)
        print("X_plus_Y", X_plus_Y)
        assert_matrix(real_X_plus_Y, X_plus_Y)

    def test_distributed_calculate_X_plus_Y_2(self):
        X = np.array([[1., 2., 3.],
                      [4., 5., 6.],
                      [7., 8., 9.]])

        Z = np.array([[1., 2., 3.],
                      [1., 2., 3.],
                      [1., 2., 3.]])

        real_X_plus_Z = X + Z
        X_plus_Z = compute_X_plus_Y(X, Z)
        print("X_plus_Z \n", X_plus_Z.shape)
        print("X_plus_Z", X_plus_Z)
        assert_matrix(real_X_plus_Z, X_plus_Z)


if __name__ == '__main__':
    unittest.main()

