from plyfile import PlyData, PlyElement
import numpy as np
import os
from utils.visualize import save_ply, save_ply_with_color
import argparse
from IPython import embed
from tqdm import tqdm

def centerailze_real_data_axis(pc):

    pc_mean = (np.max(pc[:, :3], axis=0) + np.min(pc[:, :3], axis=0)) / 2
    pc[:, :3] -= pc_mean
    pc[:, :3] -= np.array([0, pc[:, 1].min(), 0])
    scale_ratio = pc[:, 1].max()
    
    print("ratio: ", scale_ratio)
    
    pc[:, :3] = pc[:, :3] / scale_ratio
    # pc[:, 3] = -pc[:, 3]
    
    print(pc[:, 1].min())
    if pc.shape[1] == 3:
        save_ply('./data/normalized_'+filename, pc)
    else:
        save_ply_with_color('./data/normalized_'+filename, pc)


def classification_real_data(pts):
    pc_dict = dict()
    for pc in pts:
        color_key = pc[-3] * 10000000 + pc[-2] * 1000 + pc[-1]
        if color_key not in pc_dict:
            pc_dict[color_key] = []
        pc_dict[color_key].append(pc)
    
    max_height = -1
    min_height = 1e8
    max_num = -1
    for color_key in pc_dict:
        pc_dict[color_key] = np.array(pc_dict[color_key])
        max_height = max_height if max_height > pc_dict[color_key][:, 1].max() else pc_dict[color_key][:, 1].max()
        min_height = min_height if min_height < pc_dict[color_key][:, 1].min() else pc_dict[color_key][:, 1].min()
        max_num = max_num if max_num > pc_dict[color_key].shape[0] else pc_dict[color_key].shape[0]
    
    saved_pc = []
    for color_key in pc_dict:
        if pc_dict[color_key][:, 1].max() < max_height / 2:
            continue
        if pc_dict[color_key].shape[0] == max_num:
            continue
        if pc_dict[color_key][:, 1].min() > min_height:
           pc_dict[color_key][:, 1] -= pc_dict[color_key][:, 1].min() - min_height
        
        saved_pc.append(pc_dict[color_key])
        
    saved_pc = np.concatenate(saved_pc)
    save_ply_with_color("data/TreeLean.ply", saved_pc)
    
    return saved_pc


parser = argparse.ArgumentParser()
parser.add_argument('--filename', default='sample', type=str)
args = parser.parse_args()

filename = "%s.ply"%(args.filename)
filepath = './data/' + filename

# filepath = '/data/zhou1178/' + filename

if filename.split('.')[-1] == 'ply':
    data = PlyData.read(filepath)
    x = data["vertex"]["x"]
    y = data["vertex"]["y"]
    z = data["vertex"]["z"]
    if 'red' in data['vertex']:
        r = data['vertex']['red']
        g = data['vertex']['green']
        b = data['vertex']['blue']
        # pc = np.stack([x, y, z, r, g, b]).T
        pc = np.stack([x, z, y, r, g, b]).T
    else:
        # pc = np.stack([x, y, z]).T
        pc = np.stack([x, z, y]).T
else:
    data = np.loadtxt(filepath)
    pc = data[:, :3]
    
# pc = classification_real_data(pc)
centerailze_real_data_axis(pc)
# if pc.shape[1] == 3:
#     save_ply('./data/normalized_'+filename, pc)
# else:
#     save_ply_with_color('./data/normalized_'+filename, pc)
