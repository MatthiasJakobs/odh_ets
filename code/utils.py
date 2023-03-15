import numpy as np

def smape(a, b):
    assert len(a.shape) <= 1
    assert len(b.shape) <= 1

    num = np.abs(a - b)
    denom = np.abs(a) + np.abs(b)

    return (num / denom).mean()
