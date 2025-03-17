import numpy as np
import sys

def in_Polygon(x:np.ndarray, X:np.ndarray)->bool:
    assert np.all(X[0] == X[-1])
    v1 = np.stack([x1 - x for x1 in X[:-1]])
    v2 = np.stack([x2 - x for x2 in X[1:]])

    det = v1[:,:,0]*v2[:,:,1] - v1[:,:,1]*v2[:,:,0]
    dot = v1[:,:,0]*v2[:,:,0] + v1[:,:,1]*v2[:,:,1]

    angle = np.arctan2(det, dot)
    winding_number = np.abs(np.sum(angle, axis = 0))

    return winding_number > np.pi


def get_BoundingBox(pointset:np.ndarray, pad:float)->tuple[np.ndarray]:
    coordmin = np.min(pointset, axis = 0) - pad
    coordmax = np.max(pointset, axis = 0) + pad

    return coordmin, coordmax