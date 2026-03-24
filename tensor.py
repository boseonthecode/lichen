import numpy as np

class Function:
    def __init__(self, *tensors):
        self.tensors = tensors

    def forward(self, *args):
        raise NotImplementedError
    
    def backward(self, grad):
        raise NotImplementedError
    
    @classmethod
    def apply(cls, *tensors):
        func = cls(*tensors)
        raw = [t.data for t in tensors]
        out_data = func.forward(*raw)

        needs_grad = any(t.requires_grad for t in tensors)
        out = Tensor(out_data, requires_grad=needs_grad)
        out.creator = func
        return out
    

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
    
