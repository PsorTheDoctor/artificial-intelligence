import numpy as np


def max_dict(d):
    # Returns the argmax (key) and max (value) from a dict
    max_val = max(d.values())
    max_keys = [key for key, val in d.items() if val == max_val]
    return np.random.choice(max_keys), max_val
