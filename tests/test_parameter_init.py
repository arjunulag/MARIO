import numpy as np

def sigmoid(z):
    """
    Numerically stable sigmoid.
 
    we make z values between -500 and 500 to prevent any overflow error
    while making sure return values still between 0 and 1.
    """
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))
 

def relu(z, alpha=0.01):
    """
    Prevents the "dead neuron" problem (if only given -z to train, gradient just becomes 0).
    Relu doesn't treat negative and positive equally sometimes which slows down learning.
    This solves it by giving a small negative slope if z is smaller than or equal to 0.
    """
    return np.where(z > 0, z, alpha * z)

ACTIVATION_FNS = {
    "relu":    relu,
    "sigmoid": sigmoid,
}

class Parameter_init:
    def __init__(self, config, l1=0.0, l2=0.0):
        self.config = config
        self.l1     = l1
        self.l2     = l2
        self.layers = self._build_layers(config)
 
    """
    build layers for the network
    returns the layers
    """
    def _build_layers(self, config):
        layers = []
        for layer_config in config:
            layer = self._create_layer(layer_config)
            self._init_weights(layer)
            layers.append(layer)
        return layers
 
    """
    creates a layer with mock weights and bias
    """
    def _create_layer(self, layer_config):
        layer_type = layer_config.get("type")
 
        if layer_type == "linear":
            in_dim  = layer_config["in"]
            out_dim = layer_config["out"]
            #creates mock values
            return {
                "type": "linear",
                "W":    np.zeros((in_dim, out_dim)),
                "b":    np.zeros((out_dim,)),
                "activation_hint": layer_config.get("activation_hint", "relu"),
            }
 
        elif layer_type in ACTIVATION_FNS:
            layer = {"type": layer_type, "fn": ACTIVATION_FNS[layer_type]}
            if layer_type == "relu" and "alpha" in layer_config:
                alpha = layer_config["alpha"]
                layer["fn"] = lambda z, a=alpha: relu(z, a)
            return layer
 
        else:
            raise ValueError(f"Unknown layer type: '{layer_type}'. Valid types: 'linear', {list(ACTIVATION_FNS.keys())}")
 
    """
    Creates weights for function
    """
    def _init_weights(self, layer):
        if layer["type"] != "linear":
            return
        fan_in = layer["W"].shape[0]
        hint   = layer.get("activation_hint", "relu")
        std    = np.sqrt(2.0 / fan_in) if hint == "relu" else np.sqrt(1.0 / fan_in)
        layer["W"] = np.random.randn(*layer["W"].shape) * std
 
    """
    brings it forward
    """
    def forward(self, x):
        if x.ndim == 1:
            x = x[np.newaxis, :]
        for layer in self.layers:
            if layer["type"] == "linear":
                x = x @ layer["W"] + layer["b"]
            elif "fn" in layer:
                x = layer["fn"](x)
        return x
 
    def regularization_loss(self):
        """
        Computes L1 and L2 penalty across all linear layers.
        Biases are excluded (standard practice).
 
        L1 penalty:  l1 * sum(|W|)
        L2 penalty:  l2 * sum(W^2)
 
        Add the result on top of your prediction loss during training.
        """
        l1_penalty = 0.0
        l2_penalty = 0.0
 
        for layer in self.layers:
            if layer["type"] != "linear":
                continue
 
            W = layer["W"]
            l1_penalty += np.sum(np.abs(W))
            l2_penalty += np.sum(W ** 2)
 
        return self.l1 * l1_penalty + self.l2 * l2_penalty
 
    def __repr__(self):
        lines = ["Parameter_init("]
        for i, layer in enumerate(self.layers):
            if layer["type"] == "linear":
                in_d, out_d = layer["W"].shape
                lines.append(f"  [{i}] Linear({in_d} -> {out_d})")
            else:
                lines.append(f"  [{i}] {layer['type'].capitalize()}()")
        lines.append(f"  l1={self.l1}, l2={self.l2}")
        lines.append(")")
        return "\n".join(lines)

#test written by Claude Sonnet 4.6
def forward_verbose(model, x):
    """
    Same logic as model.forward(), but prints every step so you can
    see exactly what z = xW + b and the activation functions are doing.
    """
    print("=" * 55)
    print("  MULTILAYER FORWARD PASS — STEP BY STEP")
    print("=" * 55)
 
    if x.ndim == 1:
        x = x[np.newaxis, :]
 
    print(f"\n📥  Input x:\n    {x}\n    shape: {x.shape}")
 
    layer_num = 1
    for layer in model.layers:
 
        if layer["type"] == "linear":
            W, b = layer["W"], layer["b"]
            print(f"\n{'─'*55}")
            print(f"  Layer {layer_num}: Linear  ({W.shape[0]} → {W.shape[1]} neurons)")
            print(f"{'─'*55}")
            print(f"  Math:  z = x · W + b")
            print(f"  W shape: {W.shape}   b shape: {b.shape}")
            z = x @ W + b
            print(f"  z (pre-activation):\n    {np.round(z, 4)}")
            x = z
            layer_num += 1
 
        elif "fn" in layer:
            act_name = layer["type"].capitalize()
            print(f"\n{'─'*55}")
            print(f"  Activation: {act_name}")
            print(f"{'─'*55}")
            before = x.copy()
            x = layer["fn"](x)
            print(f"  Before: {np.round(before, 4)}")
            print(f"  After:  {np.round(x, 4)}")
            if layer["type"] == "relu":
                negatives = np.sum(before < 0)
                print(f"  ({negatives} value(s) were negative → leaky slope applied)")
 
    print(f"\n{'='*55}")
    print(f"  ✅  Final output:  {np.round(x, 6)}")
    if x.shape[-1] == 1:
        prob = float(x.flat[0])
        pred = 1 if prob >= 0.5 else 0
        print(f"  Probability: {prob:.4f}  →  Predicted class: {pred}")
    print("=" * 55)
    return x
 
 
# ─────────────────────────────────────────────
#  Demo
# ─────────────────────────────────────────────
 
if __name__ == "__main__":
 
    np.random.seed(42)   # makes weights reproducible every run
 
    # Define the network: 3 inputs → 4 hidden neurons → 1 output
    config = [
        {"type": "linear",  "in": 3, "out": 4, "activation_hint": "relu"},
        {"type": "relu"},
        {"type": "linear",  "in": 4, "out": 1, "activation_hint": "sigmoid"},
        {"type": "sigmoid"},
    ]
 
    model = Parameter_init(config)
    print(model)
    print()
 
    # One sample: 3 features
    x = np.array([0.5, -0.3, 1.2])
 
    output = forward_verbose(model, x)
 
    # ── Batch demo ────────────────────────────
    print("\n\n── Batch of 4 samples ──")
    X_batch = np.array([
        [ 0.5, -0.3,  1.2],
        [-1.0,  0.8,  0.1],
        [ 2.0,  0.0, -0.5],
        [ 0.3,  1.5,  0.9],
    ])
    out_batch = model.forward(X_batch)
    print(f"Input shape:  {X_batch.shape}   →   Output shape: {out_batch.shape}")
    for i, (row, prob) in enumerate(zip(X_batch, out_batch.flat)):
        pred = 1 if prob >= 0.5 else 0
        print(f"  Sample {i}: {row}  →  prob={prob:.4f}  class={pred}")
 
    # ── Regularization test ───────────────────
    print("\n\n── Regularization test ──")

    # total_loss = prediction_loss + lambda * sum(weights)
    # lambda is something used to tune and make sure weights don't explode/vanish in regularization
 
    # if there is no reg, is the loss 0?
    model_no_reg = Parameter_init(config, l1=0.0, l2=0.0)
    assert model_no_reg.regularization_loss() == 0.0, "FAIL: no-reg model should return 0"
    print("  ✅  No reg   → loss = 0.0 (correct)")
 
    # l2 only — L2 reg exists, so there must be SOME penalty
    model_l2 = Parameter_init(config, l1=0.0, l2=0.01)
    l2_loss = model_l2.regularization_loss()
    assert l2_loss > 0.0, "FAIL: L2 loss should be > 0"
    print(f"  ✅  L2 only  → loss = {l2_loss:.6f} (correct, > 0)")
 
    # l1 only — L1 reg exists, so there must be SOME penalty
    model_l1 = Parameter_init(config, l1=0.01, l2=0.0)
    l1_loss = model_l1.regularization_loss()
    assert l1_loss > 0.0, "FAIL: L1 loss should be > 0"
    print(f"  ✅  L1 only  → loss = {l1_loss:.6f} (correct, > 0)")
 
    # both — combined loss must be greater than either alone
    model_both = Parameter_init(config, l1=0.01, l2=0.01)
    both_loss = model_both.regularization_loss()
    assert both_loss > l1_loss and both_loss > l2_loss, "FAIL: combined loss should exceed either individually"
    print(f"  ✅  L1 + L2  → loss = {both_loss:.6f} (correct, > L1 and L2 alone)")
 
    # higher lambda must produce higher penalty
    model_low  = Parameter_init(config, l2=0.01)
    model_high = Parameter_init(config, l2=0.1)
    assert model_high.regularization_loss() > model_low.regularization_loss(), \
        "FAIL: higher lambda should produce higher loss"
    print(f"  ✅  lambda=0.1 loss > lambda=0.01 loss (correct, lambda scales the penalty)")
 
    print("\n  All regularization checks passed!")