import numpy as np
from models.base_model import BaseModel


class FakeAutoencoder(BaseModel):

    def __init__(self, an_id):
        super(FakeAutoencoder, self).__init__()
        self.id = str(an_id)

    def build(self, encode_dim, Wh=None, bh=None):
        self.encode_dim = encode_dim
        self.Wh = Wh
        self.bh = bh

    def transform(self, X):
        return X

    def compute_gradients(self, X):
        N = len(X)
        Whs = []
        bhs = []
        for i in range(N):
            Whs.append(self.Wh.copy())
            bhs.append(self.bh.copy())
        return [np.array(Whs), np.array(bhs)]

    def apply_gradients(self, gradients):
        pass

    def backpropogate(self, X, y, in_grad):
        print("in backpropogate with model ", self.id)
        print("X shape", X.shape)
        if y is None:
            print("y is None")
        else:
            print("y shape", y.shape)
        print("in_grad shape", in_grad.shape)
        print("in_grad", in_grad)

    def predict(self, X):
        return 0.0

    def get_features_dim(self):
        return self.encode_dim


