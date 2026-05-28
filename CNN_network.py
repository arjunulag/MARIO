import numpy as np
from Tensor import Tensor

def get_flat_size(kernels):
    dummy = Tensor(np.zeros((4, 84, 84)))
    x = dummy.conv2d(kernels[0]).relu()
    x = x.conv2d(kernels[1]).relu()
    return x.flatten().data.shape[0]

def build_weights(d_model=64):
    kernels = [
        Tensor(np.random.randn(8, 4, 8, 8) * np.sqrt(2/(4*8*8))),
        Tensor(np.random.randn(16, 8, 4, 4) * np.sqrt(2/(8*4*4))),
    ]
    flat_size = get_flat_size(kernels)  # calculate real size
    print(f"Flat size: {flat_size}")

    W1 = Tensor(np.random.randn(256, flat_size) * np.sqrt(2/flat_size))  # use flat_size here
    b1 = Tensor(np.zeros(256))
    W2 = Tensor(np.random.randn(d_model, 256) * np.sqrt(2/256))
    b2 = Tensor(np.zeros(d_model))
    return kernels, W1, b1, W2, b2

def forward(state, kernels, W1, b1, W2, b2):
    x = Tensor(state, label='input')
    x = x.conv2d(kernels[0]).relu()
    x = x.conv2d(kernels[1]).relu()
    x = x.flatten()
    x = x.linear(W1, b1).relu()
    x = x.linear(W2, b2)
    return x
