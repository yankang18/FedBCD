import numpy as np
from api.eggroll import table


if __name__ == '__main__':

    X = table("table1", "namespace1", partition=1)
    Y = table("table2", "namespace2", partition=1)

    matrix_a = np.array([[1, 2, 3]])
    matrix_b = np.array([[4, 5, 6]])
    matrix_c = np.array([[7, 8, 9]])
    matrix_d = np.array([[1, 1, 1]])

    y_a = np.ones((1,))
    y_b = np.ones((1,))
    y_c = np.array([-1])
    y_d = np.zeros((1,))

    X.put(0, matrix_a)
    X.put(1, matrix_b)
    X.put(2, matrix_c)
    X.put(3, matrix_d)

    Y.put(0, y_a)
    Y.put(1, y_b)
    Y.put(2, y_c)
    Y.put(3, y_d)

    print("here 1")
    R = X.join(Y, lambda x, y: y * x)
    print("here 2")
    result = R.reduce(lambda agg_val, v: agg_val + v)
    print(result)






