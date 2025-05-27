import numpy as np
import os
from plyfile import PlyData
from IPython import embed

root = "~/Yshape_embedding"
# filename = os.path.join(root, 'data', 'Walnut_dense.ply')
filename = '/home/zhou1178/Yshape_embedding/data/Walnut_dense.ply'
sample_num = 5000
def save_ply(xyz, fn):

    with open(fn, 'w') as f:
        pn = xyz.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (pn))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(pn):
            f.write('%.6f %.6f %.6f\n' % (xyz[i][0], xyz[i][1], xyz[i][2]))
            
            
def data_process(filename):
    data = PlyData.read(filename)
    x = data['vertex']['x']
    y = data['vertex']['y']
    z = data['vertex']['z']
    pc = np.stack([x, y, z]).T
    pc = pc[np.random.choice(pc.shape[0], sample_num, replace=pc.shape[0]<sample_num)]

    pc -= (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2
    pc /= abs(pc).max()
    height_threshold = 0
    index = np.argwhere(pc[:, 1] > height_threshold).reshape(-1)
    pc[index] += np.random.normal(size=(index.shape[0], 3)) / 75
    
    save_ply(pc, "../data/subsampled.ply")

if __name__ == "__main__":
    data_process(filename)