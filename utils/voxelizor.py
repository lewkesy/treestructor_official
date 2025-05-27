import os
from random import gauss
import numpy as np
from IPython import embed
from plyfile import PlyData, PlyElement
from time import time
from tqdm import tqdm

class voxelizor():
    def __init__(self) -> None:
        pass

    def __call__(self, points, mag_coeff=5):
        '''
            output:
                voxel_grid: voxel with number of points in the voxel
                points_in_voxel: points saved in each of the voxel
        '''
        
        self.mag_coeff = mag_coeff
        self.point_to_voxel_mapping = dict()
        self.points = points

        # if self.points.max() <= 1:
        #     self.xyz_range = [-1, -1, -1, 1, 1, 1]
        # else:
        #     self.xyz_range=[round(itm) for itm in [
        #         self.points[:, 0].min(),
        #         self.points[:, 1].min(),
        #         self.points[:, 2].min(),
        #         self.points[:, 0].max(),
        #         self.points[:, 1].max(),
        #         self.points[:, 2].max(),
        #     ]]
        self.xyz_range = [-1, -1, -1, 1, 1, 1]

        voxel_grid, points_in_voxel = self.__get_3D_matrix()

        return voxel_grid, points_in_voxel

    
    def __get_3D_matrix(self):
        lx,ly,lz,hx,hy,hz = self.xyz_range

        scalar = int(self.mag_coeff / (hx - lx))
        # print(scalar)
        dx, dy, dz = map(int, [(hx-lx)*scalar+1, (hy-ly)*scalar+1, (hz-lz)*scalar+1])

        voxel_grid = np.zeros(shape=[dx, dy, dz], dtype=np.int)
        
        points_in_voxel = [[] for _ in range(dx*dy*dz)]
        for point in self.points:

            x, y, z = point[:3]
            voxel_x = int((x - lx) * scalar)
            voxel_y = int((y - ly) * scalar)
            voxel_z = int((z - lz) * scalar)

            # if voxel_grid[voxel_x, voxel_y, voxel_z] == -1:
            #     voxel_grid[voxel_x, voxel_y, voxel_z] = label
            
            voxel_grid[voxel_x, voxel_y, voxel_z] += 1

            point_to_voxel_idx = (dy*dz) * voxel_x + dz * voxel_y + voxel_z
            points_in_voxel[point_to_voxel_idx].append(point)
        
        return voxel_grid, points_in_voxel


if __name__ == '__main__':
    v = voxelizor()

    filename = None
    voxel_grid, voxel_pos = v(filename, mag_coeff=200)

    print(voxel_grid.shape)
