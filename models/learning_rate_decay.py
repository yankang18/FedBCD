import numpy as np


def sqrt_learning_rate_decay(learning_rate, global_step):
    decay_factor = 1 / np.sqrt(global_step + 1)
    decayed_learning_rate = learning_rate * decay_factor
    # print("learning rate is decayed from {0} to {1}".format(learning_rate, decayed_learning_rate))
    return decayed_learning_rate
