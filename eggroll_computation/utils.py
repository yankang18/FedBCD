from encryption.encryption import encrypt_matrix, decrypt_matrix
import numpy as np


def distribute_compute_XY(X, Y):
    """

    :param X: DTable, with shape (feature_dim, sample_dim)
    :param Y: DTable, with shape (feature_dim, sample_dim)
    :return:
    """
    R = X.join(Y, lambda x, y: x * y)
    val = R.collect()
    table = dict(val)
    return table


def distribute_compute_X_plus_Y(X, Y):
    """

    :param X: DTable, with shape (feature_dim, sample_dim)
    :param Y: DTable, with shape (feature_dim, sample_dim)
    :return:
    """

    R = X.join(Y, lambda x, y: x + y)
    val = R.collect()
    table = dict(val)
    return table


def distribute_compute_hSum_XY(X, Y):
    """

    :param X: DTable, with shape (feature_dim, sample_dim)
    :param Y: DTable, with shape (feature_dim, sample_dim)
    :return:
    """
    R = X.join(Y, lambda x, y: np.sum(x * y))
    val = R.collect()
    table = dict(val)
    return table


def distribute_compute_vAvg_XY(X, Y, sample_dim):
    """

    :param X: DTable, with shape (feature_dim, sample_dim)
    :param Y: DTable, with shape (feature_dim, sample_dim) or (1, sample_dim)
    :param feature_dim:
    :param sample_dim:
    :return:
    """

    R = X.join(Y, lambda x, y: y * x / sample_dim)
    result = R.reduce(lambda agg_val, v: agg_val + v)
    return result


def distribute_encrypt(public_key, X):
    """
    :param X: DTable
    :return:
    """

    X2 = X.mapValues(lambda x: encrypt_matrix(public_key, x))
    val = X2.collect()
    val = dict(val)
    return val


def distribute_decrypt(private_key, X):
    """
    :param X: DTable
    :return:
    """

    X2 = X.mapValues(lambda x: decrypt_matrix(private_key, x))
    val = X2.collect()
    val = dict(val)
    return val
