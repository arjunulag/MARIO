def linear_activation_forward(x, W, b, activation_fn):
    """
    x : (batch, in_features)
    W : (in_features, out_features)
    b : (out_features,)
    """

    # linear step
    z = x @ W + b

    # activation step
    a = activation_fn(z)

    # cache values needed for backprop
    cache = {
        "x": x,
        "z": z,
        "W": W
    }

    return a, cache