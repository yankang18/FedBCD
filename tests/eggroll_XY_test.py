import numpy as np
from federated_learning.eggroll_computation.helper import compute_avg_XY, compute_sum_XY, compute_XY, compute_XY_plus_Z


if __name__ == '__main__':

    U_A = np.array([[1., 2., 3.],
                    [4., 5., 6.],
                    [7., 8., 9.]])

    Y = np.array([[1], [-1], [1]])

    Z = np.array([[1., 2., 3.],
                  [1., 2., 3.],
                  [1., 2., 3.]])

    # example = Example()
    # example.distributed_calculate_avg_XY(U_A, Y)
    print("---distributed_calculate_XY---")
    XY = compute_XY(U_A, Y)
    print("XY \n", XY.shape)
    print("XY", XY)

    print("---distributed_calculate_avg_XY---")
    XY = compute_avg_XY(U_A, Y)
    print("XY \n", XY.shape)
    print("XY", XY)

    print("---distributed_calculate_sum_XY---")
    XY = compute_sum_XY(U_A, Y)
    print("XY \n", XY.shape)
    print("XY", XY)

    XY_plus_Z = compute_XY_plus_Z(U_A, Y, Z)
    print("XY_plus_Z \n", XY_plus_Z.shape)
    print("XY_plus_Z", XY_plus_Z)







