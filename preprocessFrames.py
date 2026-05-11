import cv2
import numpy as np
from collections import deque

def preprocess_frame(frame):
    gframe = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    sframe = cv2.resize(gframe, (84,84), interpolation=cv2.INTER_AREA)
    return sframe / 255.0

def build_initial_stack(env):
    stack = deque(maxlen=4)
    for _ in range(4):
        stack.append(preprocess_frame(env.reset()))
    return stack, np.stack(list(stack), axis=0)