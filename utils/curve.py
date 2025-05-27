import numpy as np
from IPython import embed
from numpy.linalg import norm

def normalize(v):
    return v/(norm(v) + 1e-6)

# def pts_check(pos, dir, pts=None, key=None):

#     if pts is not None:
#         dir_to_pts = pts - pos
#         curr_dir = dir
#         # cosine = a.b / ||a||.||b||
#         cosine = np.dot(dir_to_pts, curr_dir) / (norm(dir_to_pts, axis=1) * norm(curr_dir) + 1e-6)
#         if not (cosine.min() < -0.5 and cosine.max() > 0.5):
#             return False
    
#     return True

def pts_check(pos, dir, pts=None, key=None):
     if pts is not None:
        if ((pos - pts[0])>=0).sum() == 3 and ((pts[1] - pos) >= 0).sum() == 3:
            return True
        else:
            return False
        

def get_curve(ps, pe, ds, de, rs, re, scalar=200, pts=None, axis=-1, key=None, bound_check=True):
    
    length = np.sqrt(np.sum((ps - pe) ** 2))
    ps_control = ps + ds * 0.3 * length
    pe_control = pe - de * 0.6 * length

    samples = [i/scalar for i in range(scalar+1)]

    points = []
    dirs = []
    radius = []
    
    accept_points = []
    accept_dirs = []
    accept_radius = []
    
    for t in samples:
        p = (1-t)**3 * ps + 3 * (1-t)**2*t * ps_control + 3 * (1-t)*t**2 * pe_control + t**3 * pe
        d = ds * (1-t) + de * t
        r = rs * (1-t) + re * t
        points.append(p)
        dirs.append(d)
        radius.append(r)
        
        if bound_check and pts_check(p, d, pts=pts, key=key):
            accept_points.append(p)
            accept_dirs.append(d)
            accept_radius.append(r)

    if bound_check:
        points = np.array(accept_points)
        dirs = np.array(accept_dirs)
        radius = np.array(accept_radius)
    else:
        points = np.array(points)
        dirs = np.array(dirs)
        radius = np.array(radius)
        
    return points, dirs, radius


def sample_circle(point, dir, r, sample_num=10):

    '''
    input:
    points: 3
    dirs  : 3
    r     : 1

    '''

    point += np.random.rand(3) / 10000
    x_bar = np.cross(point, dir)
    x_bar /= np.sqrt(np.sum(x_bar**2) + 1e-6)

    y_bar = np.cross(dir, x_bar)
    y_bar /= np.sqrt(np.sum(y_bar**2)+ 1e-6)

    theta = np.array([i/sample_num for i in range(sample_num)]) * 2 * np.pi

    sampled_points = point[None, :] + r * (np.cos(theta)[:, None] * x_bar[None, :] + np.sin(theta)[:, None] * y_bar[None, :])
    sampled_points = sampled_points.reshape(-1, 3)

    return sampled_points


def rotation_matrix_from_vectors(v1, v2):
    # Ensure that the vectors are unit vectors
    v1 = v1 / (np.linalg.norm(v1)+1e-6)
    v2 = v2 / (np.linalg.norm(v2)+1e-6)

    # Calculate the rotation axis and angle
    axis = np.cross(v1, v2)
    angle = np.arccos(np.dot(v1, v2))

    # If the vectors are already parallel, return the identity matrix
    if np.allclose(axis, 0):
        return np.eye(3)

    # Normalize the axis
    axis /= np.linalg.norm(axis)

    # Rodrigues' rotation formula
    k = axis
    K = np.array([
        [0, -k[2], k[1]],
        [k[2], 0, -k[0]],
        [-k[1], k[0], 0]
    ])
    rotation_matrix = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * np.dot(K, K)

    return rotation_matrix


def random_sample_circle(point, dir, r, sample_num=20, axis=-1):

    height = np.linalg.norm(dir)
    dir /= height
    
    theta = np.linspace(0, -2*np.pi, sample_num)
    sampled_points = np.array([r/2 * np.cos(theta), np.zeros(theta.shape[0]), r/2 * np.sin(theta)])
    rotation_matrix = rotation_matrix_from_vectors(np.array([0, 1, 0]), dir)
    translation_matrix = np.array(point)[None, :]
        
    return np.dot(rotation_matrix, sampled_points).T + translation_matrix
