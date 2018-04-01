
class BaseModel(object):

    def __init__(self, input_size):
        raise NotImplementedError("error message")

    def get_layers_info(self):
        raise NotImplementedError("error message")

    def get_layers_feauture(self):
        raise NotImplementedError("error message")

