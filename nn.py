from tensor import Tensor
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