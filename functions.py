import matplotlib.pyplot as plt
import numpy as np
import math

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
 
"""
Activation functions
"""
ACTIVATION_FNS = {
    "relu":    relu,
    "sigmoid": sigmoid,
}

"""
Creates weights and biases for network
"""
class Parameter_init:
    def __init__(self, config):
        self.config = config
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
 
    def __repr__(self):
        lines = ["Parameter_init("]
        for i, layer in enumerate(self.layers):
            if layer["type"] == "linear":
                in_d, out_d = layer["W"].shape
                lines.append(f"  [{i}] Linear({in_d} -> {out_d})")
            else:
                lines.append(f"  [{i}] {layer['type'].capitalize()}()")
        lines.append(")")
        return "\n".join(lines)