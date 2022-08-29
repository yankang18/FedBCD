import numpy as np
from api.eggroll import parallelize

def add(x):
    return x + 2


if __name__ == '__main__':
    print("eggroll simple test")

    # # 修改flow_id 否则内存表可能被覆盖
    # eggroll = EggRoll(flow_id='testaxsdfsfsdfc')
    # X = eggroll.table('test', 'testX1')
    # X.destroy()
    #
    # matrix_a = np.ones((3, 3))
    # matrix_b = np.ones((3, 3))
    #
    # X.put([(0, matrix_a)])
    # X.put([(1, matrix_b)])
    #
    # X2 = X.map(lambda x: add(x))
    # print("here")
    # val = X2.get_all()
    # print("finished")
    # print(val)

    # y = np.array([[1], [-1], [-1], [1]])
    # print(y.shape, y)
    # y_ = np.tile(y, (1, 4))
    # print(y_.shape, y_)

    # X = np.array([1, 2, 3], dtype=np.float64)
    # X1 = np.tile(np.expand_dims(X, 0), (3, 1))
    # print("X", X, X.shape)
    # print("X1", X1, X1.shape)

    X = np.array([[[10, 11, 12],
                   [13, 14, 15],
                   [16, 17, 18]],
                  [[19, 20, 21],
                   [22, 23, 24],
                   [25, 26, 27]]], dtype=np.float64)

    # X = np.array([[[1,2,3]], [[4,5,6]], [[7,8,9]]])
    print(X.shape)
    print(X)

    Y = np.array([[[10, 11, 12],
                   [13, 14, 15],
                   [16, 17, 18]],
                  [[19, 20, 21],
                   [22, 23, 24],
                   [25, 26, 27]]], dtype=np.float64)

    # X1 = np.tile(X, (3, 1))
    print(Y.shape)
    print(Y)

    print(X.shape[1], X.shape[-1])

    X = parallelize(X, partition=2)
    Y = parallelize(Y, partition=2)

    R = X.join(Y, lambda x, y: y * x)
    result = R.reduce(lambda agg_val, v: agg_val + v)
    print(result)




