import numpy as np

class Tensor:
    def __init__(self, data, requires_grad=False):
        # forces the type of data of 'data' to always be a float64
        if isinstance(data, (int, float, list)):
            data = np.array(data, dtype = np.float64) 

        self.data = data
        self.grad = None
        self.creator = None
        self.requires_grad = requires_grad

    def __repr__(self):
        return f"tensor(data={self.data}, grad={self.grad})"

    def shape(self):
        return self.data.shape
    