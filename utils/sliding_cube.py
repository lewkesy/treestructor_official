from utils.utils import loadply
import numpy as np

def aabb(points):
    x_min = np.min(points[:, 0])
    y_min = np.min(points[:, 1])
    z_min = np.min(points[:, 2])
    x_max = np.max(points[:, 0])
    y_max = np.max(points[:, 1])
    z_max = np.max(points[:, 2])

    return np.array([[x_min, y_min, z_min], [x_max, y_max, z_max]])


