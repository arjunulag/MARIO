import numpy as np
from Tensor import Tensor

def build_weights(d_model=64):          # match their config
    kernels = [
        Tensor(np.random.randn(8, 4, 8, 8) * np.sqrt(2/(4*8*8))),
        Tensor(np.random.randn(16, 8, 4, 4) * np.sqrt(2/(8*4*4))),
    ]
    W1 = Tensor(np.random.randn(256, 5776) * np.sqrt(2/5776))
    b1 = Tensor(np.zeros(256))
    W2 = Tensor(np.random.randn(d_model, 256) * np.sqrt(2/256))  # d_model out
    b2 = Tensor(np.zeros(d_model))
    return kernels, W1, b1, W2, b2

def forward(state, kernels, W1, b1, W2, b2):
    x = Tensor(state, label='input')   # (4, 84, 84)
    x = x.conv2d(kernels[0]).relu()
    x = x.conv2d(kernels[1]).relu()
    x = x.flatten()
    x = x.linear(W1, b1).relu()
    x = x.linear(W2, b2)              # outputs (d_model,) = (64,) not Q-values
    return x