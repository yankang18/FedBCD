import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def load_numpy_medical_data(file_full_path):
    return np.load(file_full_path)


def load_pandas_medical_data(file_full_path):
    return pd.read_csv(file_full_path)


def load_horizontal_medical_data(data_dir):
    file_x_train = data_dir + "x_train.npy"
    file_x_test = data_dir + "x_test.npy"
    file_y_train = data_dir + "y_train.npy"
    file_y_test = data_dir + "y_test.npy"
    x_train = load_numpy_medical_data(file_x_train)
    y_train = load_numpy_medical_data(file_y_train)
    x_test = load_numpy_medical_data(file_x_test)
    y_test = load_numpy_medical_data(file_y_test)
    print("x_train", x_train.shape)
    print("x_test", x_test.shape)
    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    # y_train = np.expand_dims(y_train, axis=1)
    # y_test = np.expand_dims(y_test, axis=1)

    print("y_train", y_train.shape)
    print("y_test", y_test.shape)

    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.fit_transform(x_test)

    x_A_train, x_B_train = x_train[:, :336], x_train[:, 336:]
    print("x_B_train", x_B_train.shape)
    print("x_A_train", x_A_train.shape)

    x_A_test, x_B_test = x_test[:, :336], x_test[:, 336:]
    print("x_B_test", x_B_test.shape)
    print("x_A_test", x_A_test.shape)

    return [x_A_train, x_B_train, y_train], [x_A_test, x_B_test, y_test]


def load_vertical_medical_data(data_dir, training_sample_rate=0.8):
    file_A = data_dir + "vertical_with_label.csv"
    file_B = data_dir + "vertical_without_label_1.csv"
    file_C = data_dir + "vertical_without_label_2.csv"

    A_samples = load_pandas_medical_data(file_A)
    B_samples = load_pandas_medical_data(file_B)
    C_samples = load_pandas_medical_data(file_C)

    print("A_samples", A_samples.shape)
    print("B_samples", B_samples.shape)
    print("C_samples", C_samples.shape)

    # print("Party A data: \n")
    # print(A_train.head(5))
    # print("Party B data: \n")
    # print(B_samples.head(5))
    # print("Party C data: \n")
    # print(C_samples.head(5))

    party_A = A_samples.values
    party_B = B_samples.values
    party_C = C_samples.values

    print("party_A", party_A.shape, type(party_A))
    print("party_B", party_B.shape, type(party_B))
    print("party_C", party_C.shape, type(party_C))

    Y = party_A[:, 1]
    Xa = party_A[:, 2:]
    Xb = party_B[:, 1:]
    Xc = party_C[:, 1:]

    print("Xa", Xa.shape)
    print("Xb", Xb.shape)
    print("Xc", Xc.shape)
    print("Y", Y.shape)

    num_train = int(training_sample_rate * Xa.shape[0])

    Y_train, Y_test = Y[:num_train], Y[num_train:]
    Xa_train, Xa_test = Xa[:num_train], Xa[num_train:]
    Xb_train, Xb_test = Xb[:num_train], Xb[num_train:]
    Xc_train, Xc_test = Xc[:num_train], Xc[num_train:]

    return (Xa_train, Xb_train, Xc_train, Y_train), (Xa_test, Xb_test, Xc_test, Y_test)
