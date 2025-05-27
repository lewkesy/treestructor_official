from utils.voxelizor import voxelizor
from utils.utils import centralize_data
import numpy as np
from tqdm import tqdm
from IPython import embed
from utils.visualize import save_ply_with_color, save_ply
from plyfile import PlyData, PlyElement

import sys
sys.setrecursionlimit(10000)

class ConnectedComponentcluster():
    def __init__(self) -> None:
        self.v = voxelizor()

    def read_ply(self, filepath):
        if filepath.split('.')[-1] == 'ply':
            data = PlyData.read(filepath)
            x = data["vertex"]["x"]
            y = data["vertex"]["y"]
            z = data["vertex"]["z"]
            pc = np.stack([x, y, z]).T
            
            if 'junctionIndex' in data:
                internode = data["internodeIndex"]['ii']
                junctionnode = data["junctionIndex"]['ji']
                info = np.stack([internode, junctionnode]).T
                pc = np.concatenate([pc, info], axis=1)
        else:
            pc = np.loadtxt(filepath)
        
        pc[:, :3] -= (np.max(pc[:, :3], axis=0) + np.min(pc[:, :3], axis=0)) / 2
        pc[:, :3] /= abs(pc[:, :3]).max()
        
        return pc

    def CCL_cluster(self, voxel_grid):
        cluster = []
        dx, dy, dz = voxel_grid.shape
        
        # dfs
        current_label = 1
        for x in range(dx):
            for y in range(dy):
                for z in range(dz):
                    if voxel_grid[x, y, z] == 0:
                        continue
                    
                    curr_cluster = self.cluster_func(x, y, z, voxel_grid)
                    if len(curr_cluster) != 0:
                        cluster.append(curr_cluster)
                        current_label += 1
        
        return cluster, current_label


    def cluster_func(self, x, y, z, voxel_grid):
        
        dx, dy, dz = voxel_grid.shape
                
        # if this is not a connected voxel
        if voxel_grid[x%dx, y%dy, z%dz] < self.threshold:
            return []
        
        # mark current voxel visited
        voxel_grid[x%dx, y%dy, z%dz] = 0
        # save points in the current voxel
        cluster = self.points_in_voxel[(dy*dz) * x + dz * y + z]
        
        # dfs flush
        for i in range(-1, 2):
            for j in range(-1, 2):
                for k in range(-1, 2):
                    if i == 0 and j == 0 and k == 0:
                        continue
                    curr_cluster = self.cluster_func(x+i, y+j, z+k, voxel_grid)
                    cluster += curr_cluster
                    
        return cluster
        

    def __call__(self, input_data, mag_coeff, threshold=5):

        # voxelization
        if type(input_data) == 'str':
            points = self.read_ply(filename)
        else:
            points = input_data.copy()
            
        self.threshold = threshold

        # print("Start voxelization")
        self.voxel_grid, self.points_in_voxel = self.v(points, mag_coeff=mag_coeff)
        self.ave_points_per_voxel = points.shape[0] / np.sum(self.voxel_grid > 0)
        
        # clustering
        # print("Start clustering")
        cluster_list, label_total = self.CCL_cluster(self.voxel_grid.copy())
        # print("Total cls: %d"%label_total)

        # visualize
        # print("Start visualization")
        visual = []

        dx, dy, dz = self.voxel_grid.shape

        for cluster in cluster_list:                    
            color = np.random.randint(64, size=3) + 192
            for point in cluster:
                visual.append([point[0], point[1], point[2], color[0], color[1], color[2]])

        return cluster_list, visual


if __name__ == "__main__":
    
    filename = './data/normalized_clean_single1.ply'
    tree_cluster = cluster()
    mag_coeff = 50
    threshold = 10
    cluster_list, visual = tree_cluster(filename, mag_coeff=mag_coeff, threshold=threshold)
    save_ply_with_color("clsutered.ply", np.array(visual))