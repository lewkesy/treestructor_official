import os
import torch
import numpy as np
import torch.utils.data as data
import pickle
import torch.nn.functional as F
from plyfile import PlyData, PlyElement
from IPython import embed
import random
from tqdm import tqdm
from utils.utils import project_ply_data
from PIL import Image
import random

from sklearn.decomposition import PCA
def get_rotated_pc(pts):
    pca = PCA(n_components=3)
    pca.fit(pts)
    matrix = pca.components_
    inv_matrix = np.linalg.inv(matrix)
    
    rotate_pts = (inv_matrix @ pts.T).T
    
    return rotate_pts, inv_matrix

class SyntheticTreeDataset(data.Dataset):
    
    def __init__(self, task, data_src, data_folder, 
                 sample_num=128, 
                 train=True, 
                 padding='constant', 
                 inference=False, 
                 show_image=False,
                 add_noise=True,
                 dir_to_top=False,
                 add_position=False):
        super().__init__()

        print("loading started")
        print("sample num: ", sample_num)
        print("Padding: ", padding)
        
        self.data_src = data_src
        self.sample_num = sample_num
        self.padding = padding
        self.inference = inference
        self.show_image = show_image
        self.train = train
        self.add_noise = add_noise
        self.dir_to_top=dir_to_top
        self.add_position=add_position
        
        pickle_data = "TreePartData.pkl"
        # pickle_data = "TestTreePartData.pkl"
        with open(os.path.join(self.data_src, data_folder, pickle_data), 'rb') as f:
            dataset = pickle.load(f)
            
        print("Dataset path: ", os.path.join(self.data_src, data_folder, pickle_data))
        print("show image: ", self.show_image)
        print("add noise: ", self.add_noise)
        
        self.data_preprocess(dataset)
        print("Initialized")

    
    def data_preprocess(self, dataset):
        print("Start processing data")
        branch_data = dataset["Branch"]
        junction_data = dataset["Junction"]
        
        if self.train:
            branch_data = branch_data[:int(len(branch_data)*0.9)]
            junction_data = junction_data[:int(len(junction_data)*0.9)]
        else:
            branch_data = branch_data[int(len(branch_data)*0.9):int(len(branch_data)*1)]
            junction_data = junction_data[int(len(junction_data)*0.9):int(len(junction_data)*1)]
        
        print("branch data size: ", len(branch_data))
        print("junction data size: ", len(junction_data))
        
        self.data = branch_data + junction_data
        self.treepart_cls_gt = [0 for _ in range(len(branch_data))] + [1 for _ in range(len(junction_data))]
        
        self.data_length = len(self.data)
        
        print("Current data size: ", self.data_length)
        
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, dataloader_idx):
        
        data = self.data[dataloader_idx]["data"]
        main_dir = self.data[dataloader_idx]["main_dir"]
        treepart_cls = self.treepart_cls_gt[dataloader_idx]
        filepath = self.data[dataloader_idx]["filepath"]
        filename = self.data[dataloader_idx]["filename"]
        normalized_offset = self.data[dataloader_idx]["normalized_offset"]
        radius = self.data[dataloader_idx]["radius"]

        pc = data[:, :3]
        
        if self.add_noise:
            radius_noise_ratio = data[:, 3]
            radius_weighted_noise = (np.random.rand(*(pc.shape))*2-1) / ((1-radius_noise_ratio) * 75 + 25)[:, None]
            pc += radius_weighted_noise
        
        center = np.zeros((3,))
        
        length = pc.shape[0]
        data_position = np.mean(pc, axis=0)
        
        curr_pc = np.stack([center for _ in range(self.sample_num)])
        if self.padding == 'constant':
            np.random.shuffle(pc)
            if pc.shape[0] < self.sample_num:
                curr_pc[:pc.shape[0]] += pc
                padding_num = self.sample_num - pc.shape[0]
            # if points num larger than place holder, insert and sort
            else:
                curr_pc += pc[:self.sample_num]
                padding_num = 0
        elif self.padding == 'sample':
            if pc.shape[0] < self.sample_num:
                sample_idx = np.random.choice(pc.shape[0], self.sample_num-pc.shape[0])
                curr_pc = np.concatenate([pc, pc[sample_idx]], axis=0)
            else:
                sample_idx = np.random.choice(pc.shape[0], self.sample_num, replace=False)
                curr_pc = pc[sample_idx]
            padding_num = 0
            np.random.shuffle(pc)
        
        pc = torch.from_numpy(curr_pc).float()
        normalized_offset = torch.Tensor(normalized_offset).float()
        main_dir = torch.from_numpy(main_dir).float()
        radius = torch.Tensor([radius]).float()
        foliage = 1 if -1 in data[:, -2] else 0

        data_dict = dict(pc=pc,
                         key=-1, 
                         filepath=filepath, 
                         treepart_cls=treepart_cls, 
                         length=length, 
                         position=data_position, 
                         padding_num=padding_num,
                         main_dir=main_dir,
                         normalized_offset=normalized_offset,
                         radius=radius,
                         foliage=foliage)

        return data_dict
    

class SyntheticTwigsTreeDataset(SyntheticTreeDataset):
    def __init__(self, task, data_src, sample_num=128, train=True, padding='wrap', inference=False, show_image=False):
        super().__init__(task, data_src, sample_num, train, padding, inference, show_image)
    
    def data_preprocess(self, dataset):
        branch_data = self.dataset["Branch"]
        junction_data = self.dataset["Junction"]
        
        self.data = branch_data + junction_data
        if self.train:
            self.data = self.data[:int(len(branch_data)*0.8)]
        else:
            self.data = self.data[int(len(branch_data)*0.8):]
            
        self.data_length = len(self.data)
        self.treepart_cls_gt = [0 for _ in range(len(self.data))]
        
        print("Current data size: ", self.data_length)
        
        
class SyntheticInferenceTreeDataset(data.Dataset):
    
    def __init__(self, task, data_src, data_folder, sample_num=500, padding='constant', add_noise=True, loading_data_path="TreePartData.pkl"):
        super().__init__()

        print("sample num: ", sample_num)
        print("Padding: ", padding)
        
        self.data_src = data_src
        self.sample_num = sample_num
        self.padding = padding
        self.add_noise = add_noise
        
        print("Loading data")
        with open(os.path.join(self.data_src, data_folder, loading_data_path), 'rb') as f:
            dataset = pickle.load(f)
        print("Data loaded")
        
        branch_data = []
        junction_data = []
        
        for curr_data in dataset["Branch"]:
            branch_data.append(curr_data)

        for curr_data in dataset["Junction"]:
            junction_data.append(curr_data)
                
        self.data_list = []
        
        self.data_list += junction_data[:int(len(junction_data))]
        self.data_list += branch_data[:int(len(branch_data))]

        self.split_index = len(junction_data[:int(len(junction_data))])
        
        self.data_length = len(self.data_list)
        print("Current data size: ", self.data_length)
        
    def __len__(self):
        return self.data_length

    def __getitem__(self, index):
        
        data = self.data_list[index]["data"]
        filename = self.data_list[index]["filename"]
        filepath = self.data_list[index]["filepath"]
        normalized_offset = self.data_list[index]['normalized_offset']
        treepart_idx = self.data_list[index]['treepart_idx']
        treepart_item_idx = self.data_list[index]['treepart_item_idx']
        
        # pc = (inv_matrix @ data[:, :3].T).T
        pc = data[:, :3]
        if_foliage = np.argwhere(data[:, -1]== -1).shape[0] > data.shape[0] / 3
        
        if self.add_noise:
            radius_noise_ratio = data[:, 3]
            radius_weighted_noise = (np.random.rand(*(pc.shape))*2-1) / ((1-radius_noise_ratio) * 25 + 25)[:, None]
            pc += radius_weighted_noise
        
        length = pc.shape[0]
        data_position = np.mean(pc, axis=0)
        
        # image = self.transforms(image.convert('RGB'))
        
        ######### Determine if the center should be 0 or the mean of the pc
        # curr_pc = np.zeros((self.sample_num, pc.shape[1]))
        center = np.zeros((3,))
        curr_pc = np.stack([center for _ in range(self.sample_num)])
        if self.padding == 'constant':
            np.random.shuffle(pc)
            if pc.shape[0] < self.sample_num:
                curr_pc[:pc.shape[0]] += pc
            # if points num larger than place holder, insert and sort
            else:
                curr_pc += pc[:self.sample_num]
        elif self.padding == 'sample':
            if pc.shape[0] < self.sample_num:
                sample_idx = np.random.choice(pc.shape[0], self.sample_num-pc.shape[0])
                curr_pc = np.concatenate([pc, pc[sample_idx]], axis=0)
            else:
                sample_idx = np.random.choice(pc.shape[0], self.sample_num, replace=False)
                curr_pc = pc[sample_idx]
            np.random.shuffle(pc)
            
        pc = torch.from_numpy(curr_pc).float()
        normalized_offset = torch.Tensor(normalized_offset).float()
        
        return dict(pc=pc, 
                    key=-1, 
                    filename=filename, 
                    length=length, 
                    position=data_position, 
                    spliter_index=self.split_index, 
                    filepath=filepath, 
                    normalized_offset=normalized_offset,
                    treepart_idx=treepart_idx,
                    treepart_item_idx=treepart_item_idx,
                    if_foliage=if_foliage)
    

class SyntheticSegmentationDataset(data.Dataset):
    def __init__(self, data_src, data_selection=20000, train=True, inference=False):
        super().__init__()

        self.data_src = data_src
        self.data_selection = data_selection
        self.inference = inference  
        self.noise_sample_num = data_selection // 10
        self.data_list = []
        self.cls_list = []

        data_list = os.listdir(os.path.join(data_src, "PointCloudSimple"))
        if train:
            self.data_list = data_list[: int(len(data_list) * 0.8)]
        else:
            self.data_list = data_list[int(len(data_list) * 0.8):int(len(data_list) *1)]
        
        self.data = []
        for filename in tqdm(self.data_list):
            self.data.append(np.loadtxt(os.path.join(self.data_src, "PointCloudSimple", filename)))
            
        print("Data preprocess done")
        
    def get_rotation_matrix(self, theta):
        return np.array([[np.cos(theta), 0, -np.sin(theta)],
                         [0, 1, 0],
                         [np.sin(theta), 0, np.cos(theta)]])
    
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        pc_file = self.data_list[index]

        data = self.data[index]
        
        pc = data[:, :3]
        isforks = data[:, -2]
        radius_noise_ratio = data[:, 3]

        offset = (np.min(pc, axis=0) + np.max(pc, axis=0)) / 2
        pc -= offset
        pc /= abs(pc[:, 1]).max()
        # radius_weighted_noise = (np.random.rand(*(pc.shape))*2-1) / ((1-radius_noise_ratio) * 25 + 90)[:, None]
        # pc += radius_weighted_noise
        if not self.inference:
            
            pc = np.dot(self.get_rotation_matrix(np.random.rand()*np.pi*2), pc.T).T
            
            # random noise
            # noise_num = int(pc.shape[0]//5)
            # noise_candidate_idx = np.random.choice(pc.shape[0], noise_num, replace=False)
            # noise = pc[noise_candidate_idx]
            # noise += (np.random.rand(*(noise.shape)) * 2 - 1) * 0.02
            
            # pc = np.concatenate([pc, noise], axis=0)
            # isforks = np.concatenate([isforks, np.zeros(noise_num)])

            shuffle_idx = np.arange(isforks.shape[0])
            np.random.shuffle(shuffle_idx)
            
            pc = np.clip(pc[shuffle_idx], -1, 1)
            isforks = isforks[shuffle_idx]
            
        gt_dict = {"pc": torch.Tensor(pc).float(),
                   "is_fork": torch.from_numpy(isforks).long(), 
                   'filename': pc_file}
        
        return gt_dict