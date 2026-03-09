import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)

class Parameter_init:
    def __init__(self, config):
        self.config = config
        self.layers = self._build_layers(config)

    # creates all layers
    def _build_layers(self,config):
        layers = []#all layers
        for layer_config in config:
            layer = self._create_layer(layer_config)
            self._init_weights(layer)
            layers.append(layer)
        return layers
    
    # decides what type of layer to make based on config
    # returns Dict
    # assigns basic weights and bias
    def _create_layer(self, layer_config):
        if layer_config["type"] == "linear":
            return {
                "type": "linear",
                "W": np.zeros((layer_config["in"], layer_config["out"])),
                "b": np.zeros((layer_config["out"]))
            }
        elif layer_config["type"] == "relu":
            return {
                "type": "relu",
                "fn": relu
            }
        elif layer_config["type"] == "sigmoid":
            return {
                "type": "sigmoid",
                "fn": sigmoid
            }
    # overwrites the basic weights and bias assigned with properly scaled values
    def _init_weights(self, layer):
        if layer["type"] == "linear":
            fan_in = layer["W"].shape[0]
            layer["W"] = np.random.randn(*layer["W"].shape) * np.sqrt(2.0 / fan_in)

    # data is flown through here
    def forward(self, x):
        for layer in self.layers:
            if layer["type"] == "linear":
                x = x @ layer["W"] + layer["b"]
            #for relu and sigmoid activation
            elif "fn" in layer:
                x = layer["fn"](x)
        return x
