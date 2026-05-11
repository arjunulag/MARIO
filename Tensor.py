class Tensor:
    def __init__(self, data, _parents=(), label=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = set(_parents)
        self.label = label

    def conv2d(self, kernel, stride=1):
        # kernel is a Tensor of shape (n_filters, in_channels, kh, kw)
        n_filters, in_ch, kh, kw = kernel.data.shape
        _, H, W = self.data.shape
        outH = (H - kh) // stride + 1
        outW = (W - kw) // stride + 1
        out_data = np.zeros((n_filters, outH, outW), dtype=np.float32)

        # --- forward ---
        for f in range(n_filters):
            for i in range(outH):
                for j in range(outW):
                    patch = self.data[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
                    out_data[f, i, j] = np.sum(patch * kernel.data[f])

        out = Tensor(out_data, (self, kernel), 'conv2d')

        # --- backward ---
        def _backward():
            for f in range(n_filters):
                for i in range(outH):
                    for j in range(outW):
                        patch = self.data[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
                        g = out.grad[f, i, j]
                        kernel.grad[f] += g * patch
                        self.grad[:, i*stride:i*stride+kh, j*stride:j*stride+kw] += g * kernel.data[f]

        out._backward = _backward
        return out

    def relu(self, alpha=0.01):
        # your existing leaky relu, wrapped into the graph
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), (self,), 'relu')

        def _backward():
            self.grad += np.where(self.data > 0, 1, alpha) * out.grad

        out._backward = _backward
        return out

    def linear(self, W, b):
        # W is a Tensor of shape (out_features, in_features)
        # b is a Tensor of shape (out_features,)
        out = Tensor(self.data @ W.data.T + b.data, (self, W, b), 'linear')

        def _backward():
            self.grad += out.grad @ W.data
            W.grad += np.outer(out.grad, self.data)
            b.grad += out.grad

        out._backward = _backward
        return out

    def flatten(self):
        original_shape = self.data.shape
        out = Tensor(self.data.flatten(), (self,), 'flatten')

        def _backward():
            self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def backward(self):
        topo = []
        visited = set()

        def build(node):
            if id(node) not in visited:
                visited.add(id(node))
                for p in node._parents:
                    build(p)
                topo.append(node)

        build(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
