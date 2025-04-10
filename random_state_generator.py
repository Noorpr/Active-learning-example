import numpy as np

def generate(seed : int) -> np.random:
    np.random.seed(seed)
    return np.random.randint(1, 100)
