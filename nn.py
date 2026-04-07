from tensor import Tensor
from functional import relu, sigmoid
import numpy as np

class Module:
    def __init__(self):
        self._parameters = {}
        self._modules = {}
        self._is_training = True

    def forward(self, x):
        raise NotImplementedError
    
    def __call__(self, *args):
        return self.forward(*args)
    
    def parameters(self):
        params = list(self._parameters.values())
        for module in self._parameters.values():
            params.extend(module.parameters())
        return params
    
    def zero_grad(self):
        for p in self.parameters():
            p.grad = None

    def train(self):
        self._is_training = True
        for m in self._modules.values():
            m.train()

    def eval(self):
        for m in self._modules.values():
            m.eval()
    
class Linear(Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        scale = np.sqrt(2.0 / in_features)
        self._parameters['W'] = Tensor(
            np.random.randn(out_features, in_features)*scale,
            requires_grad=True
        )
        self._parameters['b'] = Tensor(
            np.zeros(out_features),
            requires_grad = True
        )

    def forward(self, x):
        W = self._parameters['W']
        b = self._parameters['b']
        return x @ W.T() + b
    
class MLP(Module):
    def __init__(self, in_features, hidden, out_features):
        super().__init__()
        self._modules['l1'] = Linear(in_features, hidden)
        self._modules['l2'] = Linear(hidden, out_features)

    def forward(self, x):
        x = self._modules['l1'](x)
        x = relu(x)
        x = self._modules['l2'](x)
        x = sigmoid(x)
        return x