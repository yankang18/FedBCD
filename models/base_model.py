class BaseModel(object):

    def __init__(self):
        self.stop_training = False

    def is_stop_training(self):
        return self.stop_training

    def set_session(self, sess):
        self.sess = sess

    def get_ID(self):
        return 0
