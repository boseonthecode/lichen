# Lichen 🌱

A minimalistic, PyTorch-like machine learning library built from scratch in Python and NumPy. It features a custom autograd engine, common neural network modules, and standard optimizers. 

## Features

- **Autograd Engine**: A reverse-mode automatic differentiation engine (`tensor.py`).
- **Neural Network API**: Base `Module` classes, `Linear` layers, and a sample `MLP` (`nn.py`).
- **Functional API**: Common activations (`relu`, `sigmoid`, `softmax`) and loss functions (`mse_loss`, `cross_entropy`).
- **Optimizers**: Basic implementations of `SGD` and `Adam`.

## Installation

Clone the repository and ensure you have `numpy` installed:

```bash
git clone https://github.com/yourusername/lichen.git
cd lichen
pip install numpy
```

## Quick Start
Here is a small example of calculating cross-entropy loss and backpropagating the gradients.

```python
import numpy as np
from tensor import Tensor
from functional import cross_entropy

# 3 samples, 4 classes
logits = Tensor(
    np.array([
        [2., 1., 0., 0.], 
        [0., 0., 3., 0.], 
        [0., 2., 1., 0.]
    ]),
    requires_grad=True
)
# Intended targets
targets = Tensor(np.array([0, 2, 1]))

# Forward pass
loss = cross_entropy(logits, targets)
print(f"Loss: {loss.data}")

# Backward pass
loss.backward()
print("Gradients:\n", logits.grad)
```

## Architecture

* `tensor.py`: The core computational graph and tensor operations (Addition, Multiplication, MatMul).
* `nn.py`: High-level network layers and weight initializations.
* `functional.py`: Stateless operations, activations, and loss functions.
* `optim.py`: Parameter optimization steps.

## License
MIT
