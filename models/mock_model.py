from base_model import BaseModel


class MockModel(BaseModel):

    def __init__(self, an_id):
        super(MockModel, self).__init__()
        self.id = str(an_id)
        self.hidden_dim = None

    def build(self, hidden_dim):
        self.hidden_dim = hidden_dim

    def set_session(self, sess):
        pass

    def transform(self, X):
        return X

    def backpropogate(self, X, y, in_grad):
        pass

    def get_features_dim(self):
        return self.hidden_dim
