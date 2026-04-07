from tensor import Tensor, Function
import numpy as np

class _Relu(Function):
    def forward(self, x):
        self.x = x
        return np.maximum(x,0)
    
    def backward(self, grad):
        return(grad*(self.x>0),)

class _MSELoss(Function):
    def forward(self, pred, target):
        self.diff = pred-target
        return (np.array(np.mean(self.diff**2)))
    
    def backward(self, grad):
        n = self.diff.size
        grad_pred = grad * (self.diff * 2 / n)
        grad_target = None
        return grad_pred,grad_target
    
def relu(x):
    return _Relu.apply(x)

def mse_loss(pred, target):
    return _MSELoss.apply(pred,target)

class _Sigmoid(Function):
    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backward(self, grad):
        return (grad*self.out*(1-self.out),)
    
def sigmoid(x):
    return _Sigmoid.apply(x)