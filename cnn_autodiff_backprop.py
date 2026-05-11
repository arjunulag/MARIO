import numpy as np


class Tensor:
    def __init__(self, data, _parents=(), label=''):
        self.data = np.array(data, dtype=np.float32)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._parents = set(_parents)
        self.label = label

    def conv2d(self, kernel, bias=None, stride=1, padding=0):
        n_filters, in_ch, kh, kw = kernel.data.shape
        _, H, W = self.data.shape

        if padding > 0:
            x_pad = np.pad(
                self.data,
                ((0, 0), (padding, padding), (padding, padding)),
                mode='constant',
            )
        else:
            x_pad = self.data

        Hp, Wp = x_pad.shape[1], x_pad.shape[2]
        outH = (Hp - kh) // stride + 1
        outW = (Wp - kw) // stride + 1
        out_data = np.zeros((n_filters, outH, outW), dtype=np.float32)

        for f in range(n_filters):
            for i in range(outH):
                for j in range(outW):
                    patch = x_pad[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
                    out_data[f, i, j] = np.sum(patch * kernel.data[f])
            if bias is not None:
                out_data[f] += bias.data[f]

        parents = (self, kernel) + ((bias,) if bias is not None else ())
        out = Tensor(out_data, parents, 'conv2d')

        def _backward():
            dx_pad = np.zeros_like(x_pad)
            for f in range(n_filters):
                for i in range(outH):
                    for j in range(outW):
                        g = out.grad[f, i, j]
                        patch = x_pad[:, i*stride:i*stride+kh, j*stride:j*stride+kw]
                        kernel.grad[f] += g * patch
                        dx_pad[:, i*stride:i*stride+kh, j*stride:j*stride+kw] += g * kernel.data[f]
                if bias is not None:
                    bias.grad[f] += np.sum(out.grad[f])

            if padding > 0:
                self.grad += dx_pad[:, padding:padding+H, padding:padding+W]
            else:
                self.grad += dx_pad

        out._backward = _backward
        return out

    def maxpool2d(self, size=2, stride=2):
        C, H, W = self.data.shape
        outH = (H - size) // stride + 1
        outW = (W - size) // stride + 1
        out_data = np.zeros((C, outH, outW), dtype=np.float32)
        argmax_idx = np.zeros((C, outH, outW, 2), dtype=np.int32)

        for c in range(C):
            for i in range(outH):
                for j in range(outW):
                    region = self.data[c, i*stride:i*stride+size, j*stride:j*stride+size]
                    flat = np.argmax(region)
                    di, dj = np.unravel_index(flat, region.shape)
                    out_data[c, i, j] = region[di, dj]
                    argmax_idx[c, i, j] = (di, dj)

        out = Tensor(out_data, (self,), 'maxpool2d')

        def _backward():
            for c in range(C):
                for i in range(outH):
                    for j in range(outW):
                        di, dj = argmax_idx[c, i, j]
                        self.grad[c, i*stride+di, j*stride+dj] += out.grad[c, i, j]

        out._backward = _backward
        return out

    def relu(self, alpha=0.01):
        out = Tensor(np.where(self.data > 0, self.data, alpha * self.data), (self,), 'relu')

        def _backward():
            self.grad += np.where(self.data > 0, 1.0, alpha) * out.grad

        out._backward = _backward
        return out

    def flatten(self):
        original_shape = self.data.shape
        out = Tensor(self.data.flatten(), (self,), 'flatten')

        def _backward():
            self.grad += out.grad.reshape(original_shape)

        out._backward = _backward
        return out

    def linear(self, W, b):
        out = Tensor(self.data @ W.data.T + b.data, (self, W, b), 'linear')

        def _backward():
            self.grad += out.grad @ W.data
            W.grad += np.outer(out.grad, self.data)
            b.grad += out.grad

        out._backward = _backward
        return out

    def softmax_cross_entropy(self, target_idx):
        shifted = self.data - np.max(self.data)
        exp = np.exp(shifted)
        probs = exp / np.sum(exp)
        loss_val = -np.log(probs[target_idx] + 1e-12)
        out = Tensor(loss_val, (self,), 'softmax_ce')

        def _backward():
            dx = probs.copy()
            dx[target_idx] -= 1.0
            self.grad += dx * out.grad

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


def he_init(shape, fan_in):
    return np.random.randn(*shape).astype(np.float32) * np.sqrt(2.0 / fan_in)


if __name__ == "__main__":
    np.random.seed(0)

    in_channels = 1
    img_h, img_w = 8, 8
    n_filters_1 = 4
    kh1, kw1 = 3, 3
    n_classes = 3

    x = Tensor(np.random.randn(in_channels, img_h, img_w))
    target = 1

    K1 = Tensor(he_init((n_filters_1, in_channels, kh1, kw1), in_channels * kh1 * kw1))
    b1 = Tensor(np.zeros(n_filters_1))

    conv_out_h = (img_h - kh1) + 1
    conv_out_w = (img_w - kw1) + 1
    pool_out_h = conv_out_h // 2
    pool_out_w = conv_out_w // 2
    fc_in = n_filters_1 * pool_out_h * pool_out_w

    W_fc = Tensor(he_init((n_classes, fc_in), fc_in))
    b_fc = Tensor(np.zeros(n_classes))

    params = [K1, b1, W_fc, b_fc]
    lr = 0.01

    for step in range(20):
        for p in params:
            p.grad = np.zeros_like(p.data)

        a = x.conv2d(K1, b1, stride=1, padding=0)
        a = a.relu()
        a = a.maxpool2d(size=2, stride=2)
        a = a.flatten()
        logits = a.linear(W_fc, b_fc)
        loss = logits.softmax_cross_entropy(target)

        loss.backward()

        for p in params:
            p.data -= lr * p.grad

        print(f"step {step:02d}  loss={float(loss.data):.4f}")
