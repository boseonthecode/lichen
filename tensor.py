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

class Add(Function):
    def forward(self, x, y):
        return x+y
    
    def backward(self, grad):
        x,y = self.tensors[0],self.tensors[1]
        grad_x = grad if x.requires_grad else None
        grad_y = grad if y.requires_grad else None
        return grad_x,grad_y
    
class Mul(Function):
    def forward(self, x, y):
        return x*y
    
    def backward(self, grad):
        x,y = self.tensors[0], self.tensors[1]
        grad_x = grad * y.data if x.requires_grad else None
        grad_y = grad * x.data if y.requires_grad else None
        return grad_x, grad_y

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
    
    def __add__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Add.apply(self, other)
    
    def __mul__(self, other):
        if not isinstance(other, Tensor):
            other = Tensor(other)
        return Mul.apply(self, other)
    
    def __radd__(self, other):
        return self.__add__(other)
    
    def __rmul__(self, other):
        return self.__mul__(other)
    
    def backward(self, grad=None):
        if grad is None:
            grad = np.ones_like(self.data)
        self.grad=grad

        order = []
        visited = set()
        stack = [(self,False)]

        while(stack):
            node, children_done = stack.pop()
            if node in visited:
                continue
            if not children_done:
                stack.append((node,True))
                if node.creator is not None:
                    for parent in node.creator.tensors:
                        if parent.requires_grad:
                            stack.append((parent,False))
            else:
                visited.add(node)
                order.append(node)

        for tensor in reversed(order):
            if tensor.creator is None:
                continue

            grads = tensor.creator.backward(tensor.grad)
            for inp, g in zip(tensor.creator.tensors, grads):
                if g is None or not inp.requires_grad:
                    continue
                if inp.grad is None:
                    inp.grad = g.copy()
                else:
                    inp.grad += g