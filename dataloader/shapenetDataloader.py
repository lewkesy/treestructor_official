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
from utils.utils import loadply
from PIL import Image
from torchvision import transforms


class ShapeNetDataset(data.Dataset):
    
    def __init__(self, task, data_src, sample_num=128, train=True, padding='sample'):
        super().__init__()

        print("sample num: ", sample_num)
        print("Padding: ", padding)
        
        self.transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(40),
        ])
        
        self.data_src = data_src
        self.sample_num = sample_num
        self.padding = padding
        
        self.data_list = []
        self.key_list = []
        self.image_list = []
        cls_name = []
        
        key_name = os.listdir(self.data_src)
        for key in key_name:
            if os.path.isdir(os.path.join(self.data_src, key)):
                if len(os.listdir(os.path.join(self.data_src, key, 'points'))) > 2000:
                    cls_name.append(key)
        
        print(cls_name)        
        for i, key in enumerate(cls_name):
            
            curr_point_path = os.path.join(self.data_src, key, 'points')
            curr_img_path = os.path.join(self.data_src, key, 'images')
            curr_files = os.listdir(curr_point_path)
            if train:
                curr_files = curr_files[:int(len(curr_files)*0.8)]
            else:
                curr_files = curr_files[-400:]
            
            # TODO: Debug
            # if train:
            #     curr_files = curr_files[:int(len(curr_files)*0.1)]
            # else:
            #     curr_files = curr_files[-10:]

            for file in curr_files:
                self.data_list.append(os.path.join(curr_point_path, file))
                self.image_list.append(os.path.join(curr_img_path, file.split('.')[0]+'.png'))
                self.key_list.append(i)
               
        print("Current dataset length: ", len(self.data_list))
        
        
    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, index):
        
        filename = self.data_list[index]
        imagename = self.image_list[index]
        key = self.key_list[index]
        
        with open(filename, 'rb') as f:
            pc = np.loadtxt(f)
            pc -= (np.min(pc, axis=0) + np.max(pc, axis=0)) / 2
            pc /= abs(pc).max()
        
        image = self.transforms(Image.open(imagename).convert('RGB'))
        
        
        if self.padding == "sample":
            select_idx = np.random.choice(pc.shape[0], self.sample_num)
            pc = pc[select_idx]
        else:
            if pc.shape[0] < self.sample_num:
                pc = np.pad(pc, ((0,self.sample_num-pc.shape[0]), (0,0)), self.padding)
            else:
                pc = pc[:self.sample_num]
                
        pc = np.clip(pc, -1, 1)
        
        pc = torch.from_numpy(pc).float()
        return dict(pc=pc, image=image, key=key)
    