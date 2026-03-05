import matplotlib.pyplot as plt
import numpy as np
import math

def sigmoid(z):
    return 1/(1 + np.exp(-z))

def relu(z):
    return np.maximum(0,z)