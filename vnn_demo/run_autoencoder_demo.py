import tensorflow as tf

from models.autoencoder import Autoencoder
from datasets.data_util import load_data, split_data_combined

if __name__ == '__main__':

    infile = "./datasets/UCI_Credit_Card/UCI_Credit_Card.csv"
    X, y = load_data(infile=infile, balanced=True)

    print("################################ Generate Data for Two Parties ############################")

    X_A, y_A, X_B, y_B, overlap_indexes = split_data_combined(X, y,
                                                              overlap_ratio=0.1,
                                                              ab_split_ratio=0.1,
                                                              n_feature_b=23)

    print("X_A shape", X_A.shape)
    print("y_A shape", y_A.shape)
    print("X_B shape", X_B.shape)
    print("y_B shape", y_B.shape)

    print("################################ Build Autoencoders for Two Parties ############################")

    tf.reset_default_graph()

    autoencoder_A = Autoencoder(1)
    autoencoder_B = Autoencoder(2)

    autoencoder_A.build(X_A.shape[-1], 200, learning_rate=0.01)
    autoencoder_B.build(X_B.shape[-1], 200, learning_rate=0.01)

    print("################################ Run Autoencoders ############################")

    # Note, here is just running autoencoders working as a feature extractor that
    # transforms raw datasets input into more representative features.
    # Here is NOT training the autoencoders

    init_op = tf.global_variables_initializer()
    with tf.Session() as session:
        autoencoder_A.set_session(session)
        autoencoder_B.set_session(session)
        session.run(init_op)

        y_A_hat = autoencoder_A.predict(X_A)
        y_B_hat = autoencoder_B.predict(X_B)

    print("y_A_hat", y_A_hat.shape)
    print("y_B_hat", y_B_hat.shape)



