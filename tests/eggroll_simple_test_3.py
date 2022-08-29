import numpy as np

if __name__ == '__main__':
    X = np.array([[1, 2, 3],
                 [4, 5, 6],
                 [7, 8, 9],
                 [10, 11, 12],
                 [13, 14, 15],
                 [16, 17, 18],
                 [19, 20, 21]])

    Y = np.array([[[10, 11, 12],
                   [13, 14, 15],
                   [16, 17, 18]],
                  [[19, 20, 21],
                   [22, 23, 24],
                   [25, 26, 27]]], dtype=np.float64)

    for item in X:
        print(item, type(item), item.shape, len(item.shape))

    print("-------")
    for item in Y:
        print(item, type(item), item.shape, len(item.shape))

