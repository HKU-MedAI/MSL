import numpy as np

def norm(x, a,b):
    # input datatype np.uint8
    x = np.array(x, dtype='float')
    x = x/(b-a) - 255*a/(b-a)
    x[x>255.0] = 255.0
    x[x<0.0] = 0.0
    x = x.astype(np.uint8)
    return x

def trunc(x):
    # input datatype float
    x[x>255.0] = 255.0
    x[x<0.0] = 0.0
    return x
