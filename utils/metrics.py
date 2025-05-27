from typing import Any
import numpy as np
import open3d as o3d
from IPython import embed
from utils.visualize import save_ply
from utils.utils import chamfer_distance_numpy, cd_loss_L2, EMD_loss
import torch
import yaml

class Mesh2PointMetrics():
    def __call__(self, mesh_pc, pc, threshold_ratio=0.1):
        self.mesh_pc = mesh_pc
        self.pc = pc
        self.threshold = (np.max(self.pc, axis=0) - np.min(self.pc, axis=0)).min() * threshold_ratio
        
        precision = self.get_precision()
        recall = self.get_recall()

        return precision, recall


    def get_precision(self):
        cd, min_y_to_x = chamfer_distance_numpy(self.mesh_pc, self.pc, direction='y_to_x')
        
        total_num = min_y_to_x.shape[0]
        accepted_pc_num = np.sum(min_y_to_x < self.threshold)
        
        return accepted_pc_num / total_num
    
    
    def get_recall(self):
        cd, min_x_to_y = chamfer_distance_numpy(self.mesh_pc, self.pc, direction='x_to_y')
        
        total_num = min_x_to_y.shape[0]
        accepted_pc_num = np.sum(min_x_to_y < self.threshold)
        
        return accepted_pc_num / total_num
    
    
class Point2PointMetrics():
    def __call__(self, mesh_pc, pc):
        mesh_pc_pytorch = torch.from_numpy(mesh_pc).to('cuda')[None, :, :].float()
        pc_pytorch = torch.from_numpy(pc).to('cuda')[None, :, :].float()
        
        cd = cd_loss_L2(mesh_pc_pytorch, pc_pytorch).item()
        # emd = EMD_loss(mesh_pc_pytorch, pc_pytorch).item()
        emd = np.zeros_like(cd)
        
        return cd, emd