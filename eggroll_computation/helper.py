from api.eggroll import parallelize, table
import numpy as np
import math
from eggroll_computation.utils import distribute_compute_vAvg_XY, distribute_compute_hSum_XY, distribute_encrypt, \
    distribute_decrypt, distribute_compute_XY, distribute_compute_X_plus_Y
import uuid


def prepare_table(matrix, batch_size=1, id=None):
    """

    :param matrix: 2D matrix
    :param batch_size: batch size for sample space
    :return:
    """
    m_length = len(matrix)
    n_batches = math.ceil(m_length / batch_size)
    # print("# of partition: ", n_batches)
    X = parallelize(matrix, partition=n_batches)

    # eggroll = EggRoll(flow_id='testaxsdfsfsdfc')
    # X = eggroll.table('test', 'testX12' + str(id))
    # X.destroy()
    #
    # m_length = len(matrix)
    # n_batches = math.ceil(m_length / batch_size)
    #
    # print("data_length", m_length)
    # print("batch_size", batch_size)
    # print("number_of_batch", n_batches)
    #
    # for i in range(n_batches):
    #     batch_m = matrix[i * batch_size: i * batch_size + batch_size]
    #     print(i, batch_m, batch_m.shape)
    #     print("batch_m type", type(batch_m))
    #     X.put([(i, batch_m)])
    #
    # print("prepare table finished")

    return X


def create_empty_table(table_name, namespace, partition=1):
    return table(name=table_name, namespace=namespace, partition=partition)


def compute_sum_XY(X, Y):
    batch = 1
    XT = prepare_table(X, batch, 1)
    YT = prepare_table(Y, batch, 2)
    return distribute_compute_vAvg_XY(XT, YT, 1)


# def compute_sum_XY_3(X, Y):
#     batch = 1
#     XT = prepare_table(X, batch, 1)
#     YT = prepare_table(Y, batch, 2)
#     return distribute_compute_avg_XY_3(XT, YT, X.shape[1], X.shape[-1], 1)


def compute_avg_XY(X, Y):
    length = len(X)
    batch = 1
    XT = prepare_table(X, batch, 1)
    YT = prepare_table(Y, batch, 2)
    return distribute_compute_vAvg_XY(XT, YT, length)


# def compute_XY(X, Y):
#     batch = 1
#     XT = prepare_table(X, batch, 1)
#     YT = prepare_table(Y, batch, 2)
#
#     R = XT.join(YT, lambda x, y: y * x)
#     val = R.collect()
#     val = dict(val)
#
#     result = []
#     for i in range(len(val)):
#         result.append(val[i])
#     return np.array(result)


def compute_XY(X, Y):
    batch = 1
    XT = prepare_table(X, batch, 1)
    YT = prepare_table(Y, batch, 2)

    val = distribute_compute_XY(XT, YT)

    result = []
    for i in range(len(val)):
        result.append(val[i])
    return np.array(result)


def compute_XY_plus_Z(X, Y, Z):
    batch = 1
    XT = prepare_table(X, batch, 1)
    YT = prepare_table(Y, batch, 2)
    ZT = prepare_table(Z, batch, 3)

    R = XT.join(YT, lambda x, y: y * x).join(ZT, lambda x, y: x + y)
    val = R.collect()
    val = dict(val)

    result = []
    for i in range(len(val)):
        result.append(val[i])
    return np.array(result)


def compute_X_plus_Y(X, Y):
    batch = 1
    XT = prepare_table(X, batch, 1)
    YT = prepare_table(Y, batch, 2)

    val = distribute_compute_X_plus_Y(XT, YT)

    result = []
    for i in range(len(val)):
        result.append(val[i])
    return np.array(result)


def _convert_3d_to_2d_matrix(matrix):
    dim1, dim2, dim3 = matrix.shape
    ddim1 = dim1 * dim2
    matrix = matrix.reshape((ddim1, dim3))
    return matrix


def encrypt_matrix(encryption_key, matrix, is_encryption=True):

    _shape = matrix.shape
    if len(_shape) == 3:
        matrix = _convert_3d_to_2d_matrix(matrix)

    # print("r matrix shape", matrix.shape)
    # print("r matrix \n", matrix)

    X = prepare_table(matrix, 1, 1)

    # for item in X.collect():
    #     print("item:", item)

    if is_encryption:
        val = distribute_encrypt(encryption_key, X)
    else:
        val = distribute_decrypt(encryption_key, X)

    # print("val", val)
    # print("val 0 type", type(val[0]))
    # print("val length", len(val))

    result = []
    last_index = len(val) - 1
    for i in range(last_index):
        result.append(val[i])

    # result = np.array(result)
    # return result
    # print("result[0].shape", result[0].shape)
    # print("val[last_index].shape", val[last_index].shape)
    # print("result len", len(result))

    # print("result shape", len(result))
    # print("result", result)
    # print("last_index", last_index)

    if len(result) == 0:
        result = val[last_index]
    elif len(result[0]) == len(val[last_index]):
        result.append(val[last_index])
        result = np.array(result)
        result = result.reshape((result.shape[0] * result.shape[1], result.shape[-1]))
    else:
        result = np.array(result)
        result = result.reshape((result.shape[0] * result.shape[1], result.shape[-1]))
        result = np.vstack((result, val[last_index]))

    if len(_shape) == 3:
        result = result.reshape(_shape)
    return result


def encrypt_matmul_2(X, Y):
    # batch = 1
    # XT = prepare_table(X, batch, 1)
    # YT = prepare_table(Y, batch, 2)
    XT = create_empty_table(str(uuid.uuid1()), str(uuid.uuid1()), partition=2)
    YT = create_empty_table(str(uuid.uuid1()), str(uuid.uuid1()), partition=2)

    for m in range(len(X)):
        for k in range(Y.shape[1]):
            key = str(m) + "_" + str(k)
            # print(key)
            # print(X[m], X[m].shape)
            # print(Y[:, k], Y[:, k].shape)
            XT.put(key, X[m])
            YT.put(key, Y[:, k])

    dictionary = distribute_compute_hSum_XY(XT, YT)

    res = [[0 for _ in range(Y.shape[1])] for _ in range(len(X))]
    for m in range(len(X)):
        row_list = []
        for k in range(Y.shape[1]):
            key = str(m) + "_" + str(k)
            row_list.append(dictionary[key])
        res[m] = row_list

    return np.array(res)


def encrypt_matmul_3(X, Y):
    assert X.shape[0] == Y.shape[0]

    XT = create_empty_table(str(uuid.uuid1()), str(uuid.uuid1()), partition=2)
    YT = create_empty_table(str(uuid.uuid1()), str(uuid.uuid1()), partition=2)

    for i in range(X.shape[0]):
        for m in range(X.shape[1]):
            for k in range(Y.shape[-1]):
                key = str(i) + "_" + str(m) + "_" + str(k)
                XT.put(key, X[i, m, :])
                YT.put(key, Y[i, :, k])

    dictionary = distribute_compute_hSum_XY(XT, YT)

    res = [[[0 for _ in range(Y.shape[-1])] for _ in range(X.shape[1])] for _ in range(X.shape[0])]
    for i in range(X.shape[0]):
        second_dim_list = []
        for m in range(X.shape[1]):
            third_dim_list = []
            for k in range(Y.shape[-1]):
                key = str(i) + "_" + str(m) + "_" + str(k)
                third_dim_list.append(dictionary[key])
            second_dim_list.append(third_dim_list)
        res[i] = second_dim_list

    return np.array(res)
