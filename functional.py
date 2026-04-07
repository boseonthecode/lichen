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
    
class _Sigmoid(Function):
    def forward(self, x):
        self.out = 1/(1+np.exp(-x))
        return self.out
    
    def backward(self, grad):
        return (grad*self.out*(1-self.out),)
    
class _Softmax(Function):
    def forward(self, x):
        e = np.exp(x - x.max(axis=-1, keepdims=True))
        self.out = e/e.sum(axis=-1, keepdims=True)
        return self.out
    
    def backward(self, grad):
        s = self.out
        dot = (grad*s).sum(axis=-1, keepdims=True)
        return(s*(grad-dot),)
    
class _CrossEntropyLoss(Function):
    def forward(self, logits, targets):
        batch = logits.shape[0]
        e = np.exp(logits - logits.max(axis=1, keepdims = True))
        self.probs = e/e.sum(axis=1, keepdims=True)
        self.targets = targets.astype(int)
        correct_probs = self.probs[np.arrange(batch), self.targets]
        return np.array(-np.mean(np.log(correct_probs + 1e-9)))  
    
    def backward(self, grad):
        batch = self.probs.shape[0]
        d = self.probs.copy()
        d[np.arrange(batch), self.targets] -=1
        d = grad*d/batch
        return d, None
    
def relu(x):
    return _Relu.apply(x)

def mse_loss(pred, target):
    return _MSELoss.apply(pred,target)

def sigmoid(x):
    return _Sigmoid.apply(x)

def softmax(x):
    return _Softmax.apply(x)

def cross_entropy(logits, targets):
    return _CrossEntropyLoss.apply(logits, targets)