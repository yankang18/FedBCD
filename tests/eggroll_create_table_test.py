import numpy as np
from api.eggroll import parallelize,table
from federated_learning.datasets.data_util import load_data


class Instance(object):
    def __init__(self, inst_id=None, weight=1.0, features=None, label=None):
        self.inst_id = inst_id
        self.weight = weight
        self.features = features
        self.label = label

    def set_weight(self, weight=1.0):
        self.weight = weight

    def set_label(self, label=1):
        self.label = label

    def set_feature(self, features):
        self.features = features


def split_data_combined(X, y, overlap_ratio=0.2, guest_split_ratio=0.5, guest_feature_num=16):
    data_size = X.shape[0]
    # n_feature_a = 16
    # n_feature_b = X.shape[1] - n_feature_b
    overlap_size = int(data_size * overlap_ratio)
    overlap_indexes = np.array(range(overlap_size))
    print("overlap_indexes", overlap_indexes)
    print("len overlap_indexes", len(overlap_indexes))
    print("(data_size - overlap_size)", (data_size - overlap_size))
    guest_size = int((data_size - overlap_size) * guest_split_ratio)
    print("guest_size", guest_size)
    # B_size = (data_size - overlap_size) - guest_size
    X_A = X[:guest_size + overlap_size, guest_feature_num:]
    y_A = y[:guest_size + overlap_size, :]

    guest_data = table("tl1", "ns1", partition=2)
    # instances = {}
    for i in range(0, guest_size + overlap_size):
        # print(i, X[i, n_feature_b:], y[i, :])
        # Instance(inst_id=None, features=X[i, :n_feature_b].reshape(1, -1), label=y[i, 0])
        # instances[i] = Instance(inst_id=None, features=X[i, :n_feature_b].reshape(1, -1), label=y[i, 0])
        guest_data.put(i, Instance(inst_id=None, weight=1.0, features=X[i, :guest_feature_num].reshape(1, -1), label=y[i, 0]))
        # guest_data.put(i, i)

    host_data = table("tl3", "ns3", partition=2)
    count = 0
    for i in range(0, overlap_size):
        count += 1
        # print(i, X[i, :n_feature_b], y[i, :])
        # Instance(inst_id=None, features=X[i, :n_feature_b].reshape(1, -1), label=y[i, 0])
        # instances[i] = Instance(inst_id=None, features=X[i, :n_feature_b].reshape(1, -1), label=y[i, 0])
        host_data.put(i, Instance(inst_id=None, weight=1.0, features=X[i, guest_feature_num:].reshape(1, -1), label=y[i, 0]))
        # host_data.put(i, i)
    print("1g", count)
    count = 0
    for i in range(guest_size + overlap_size, len(X)):
        count += 1
        # print(i, X[i, :n_feature_b], y[i, :])
        # instances[i] = Instance(inst_id=None, features=X[i, :n_feature_b].reshape(1, -1), label=y[i, 0])
        # Instance(inst_id=None, features=X[i, :n_feature_b].reshape(1, -1), label=y[i, 0])
        # host_data.put(i, i)
        host_data.put(i, Instance(inst_id=None, weight=1.0, features=X[i, guest_feature_num:].reshape(1, -1), label=y[i, 0]))
    print("1h", count)
    # print("guest_data len:", len(guest_data))
    # print("host_data len:", len(host_data))

    # join_table = guest_data.join(host_data, lambda a, b: 0)
    # datasets = join_table.collect()
    # for d in datasets:
    #     print("datasets", d)

    count = 0
    for g in guest_data.collect():
        print("gg", g)
        count = count + 1
    print("g", count)
    count = 0
    for h in host_data.collect():
        count = count + 1
    print("h", count)
    X_B = np.vstack((X[:overlap_size, :guest_feature_num], X[guest_size + overlap_size:, :guest_feature_num]))
    y_B = np.vstack((y[:overlap_size, :], y[guest_size + overlap_size:, :]))
    # print("X shape:", X.shape)
    # print("X_A shape:", X_A.shape)
    # print("X_B shape:", X_B.shape)
    # print("overlap size:", overlap_size)
    # print(np.sum(y_A[overlap_indexes]>0))
    return X_A, y_A, X_B, y_B, overlap_indexes


if __name__ == '__main__':

    # X = np.array([[[10, 11, 12],
    #                [13, 14, 15],
    #                [16, 17, 18]],
    #               [[19, 20, 21],
    #                [22, 23, 24],
    #                [25, 26, 27]]], dtype=np.float64)
    #
    # X = parallelize(X).collect()
    # # print(X.shape)
    # print(X)
    # for item in X:
    #     print(item)

    # infile = "C:/Users/yangkang/Documents/FederatedLearningProject/UCI_Credit_Card/UCI_Credit_Card.csv"
    infile = "/home/maxhuang/eggroll_test/python/src/federated_learning/tests/UCI_Credit_Card/UCI_Credit_Card.csv"
    X, y = load_data(infile=infile, balanced=True)

    X = X[:1000]
    y = y[:1000]

    X_A, y_A, X_B, y_B, overlap_indexes = split_data_combined(X, y,
                                                              overlap_ratio=0.2,
                                                              guest_split_ratio=0.1,
                                                              guest_feature_num=18)



