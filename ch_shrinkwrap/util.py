import numpy as np

def fast_3x3_cross(a,b):
    x = a[1]*b[2] - a[2]*b[1]
    y = a[2]*b[0] - a[0]*b[2]
    z = a[0]*b[1] - a[1]*b[0]

    vec = np.array([x,y,z])
    return vec

def fast_sum(vec):
    return vec[0]+vec[1]+vec[2]

def dot(v, w):
    return (v*w).sum()

def dot2(v):
    return (v*v).sum()

def clamp(v, lo, hi):
    if v < lo:
        return lo
    if hi < v:
        return hi
    return v

def sign(x):
    if x > 0:
        return 1
    return -1
    