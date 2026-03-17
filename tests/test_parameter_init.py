from functions import sigmoid, relu, ACTIVATION_FNS, Parameter_init
import numpy as np
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
 
    print(f"\nInput x:\n    {x}\n    shape: {x.shape}")
 
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
    print(f"Final output:  {np.round(x, 6)}")
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