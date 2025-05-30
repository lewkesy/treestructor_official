## *TreeStructor*: Forest Reconstruction With Neural Ranking
Created by Xiaochen Zhou, Bosheng Li, Bedrich Benes, Ayman Habib, Songlin Fei, Jinyuan Shao, Sören Pirk

|[Webpage](https://lewkesy.github.io/treestructor/)|[Paper](https://ieeexplore.ieee.org/document/10950450)|[Data](https://drive.google.com/file/d/1IkBIoF1MitsuKz3H3gBV6ATwY9YqTRfG/view?usp=sharing)|

![prediction example](https://github.com/lewkesy/treestructor/blob/main/static/images/teaser.png)

### Introduction
This is the official implementation for our [report](https://ieeexplore.ieee.org/document/10950450) published on IEEE transaction of Geoscience and Remote Sensing. We introduce TreeStructor, a novel approach for isolating and reconstructing forest trees. The key novelty is a deep neural model that uses neural ranking to assign pre-generated connectable 3D geometries to a point cloud. TreeStructor is trained on a large set of synthetically generated point clouds. The input to our method is a forest point cloud that we first decompose into point clouds that approximately represent trees and then into point clouds that represent their parts. We use a point cloud encoder-decoder to compute embedding vectors that retrieve the best-fitting surface mesh for each tree part point cloud from a large set of predefined branch parts. Finally, the retrieved meshes are connected and oriented to obtain individual surface meshes of all trees represented by the forest point cloud.

In this repository, we release code and data for training TreeStructor and inference code for [peak density clustering](https://github.com/lewkesy/PeakDensityCluster) and neural ranking. Please find *EvoEngine*(https://github.com/edisonlee0212/EvoEngine) for tree geometry connection and visualization (coming soon).

### Citation
If you find our work useful in your research, please consider citing:
```
@ARTICLE{10950450,
        author={Zhou, Xiaochen and Li, Bosheng and Benes, Bedrich and Habib, Ayman and Fei, Songlin and Shao, Jinyuan and Pirk, Sören},
        journal={IEEE Transactions on Geoscience and Remote Sensing}, 
        title={TreeStructor: Forest Reconstruction with Neural Ranking}, 
        year={2025},
        volume={},
        number={},
        pages={1-1},
        keywords={Vegetation;Point cloud compression;Forestry;Solid modeling;Three-dimensional displays;Vegetation mapping;Skeleton;Image reconstruction;Vectors;Laser radar;Neural Networks;Forest Modeling;3D Reconstruction;and Remote Sensing},
        doi={10.1109/TGRS.2025.3558312}}
```

### Environment install
We encourage our users to use Conda environment for the best experience. The following instructions should be conducted in your local environment:

```
conda create -n treestructor python=3.8
conda activate treestructor
```

Our codebase is tested under Pytorch1.13 + CUDA 11.7 environment. Please install the related environment by: 
```
conda install -c "nvidia/label/cuda-11.7.0" cuda-toolkit
pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117
```

Please install packages and compile kernal from [RSCNN](https://github.com/Yochengliu/Relation-Shape-CNN) by:
```
pip install -r requirement.txt
```

### Usage
Please train the model with proper dataset by:
```
python train.py
```
You can customize the traning strategy in the codebase. The current training is using 4 GPUs. The initial parameters are designed for the best performance for training purpose.

For inference, please save your data in ./data folder first. Please keep in mind that the direction of tree growth is **Y-axis**. If the point scanning for your data is pointing to Z-axis, please run the normalization process:
```
python real_data_normalization_multi.py --filename $YOUR_FILENAME
```

Please run command for peak clustering and neural ranking:
```
python reconstruction_inference.py --filename $YOUR_FILENAME --rotate_augmentation --scale 80 --sample_num 40000 --candidate_num 5
```

Here are the breakdown for the parameters:
- filename: filename for the point cloud
- scale: scale size for re-normalization.
- sample_num: sampled number for each single tree in the forest point cloud.
- candidate_num: selected candidates for each neural ranking results. E.g, 10 means selecting the top 10 tree part from the dataset with the closest neural feature and find the tree part with the lowest chamfer distance as the candidate.


### Output of Inference
When you finished the inference code, you will find your output in ./results folder. Here are some example of key files you may interested in:

1. MST_seg_normalization_xxx.ply: This is the .ply file visualize the result of Peak Density Clustering from your input point cloud. These individual parts in different colors will be used in the next step for neural ranking.
2. rebuild_MST_normalized_xxx.ply: This is the .ply file visualize the result of neural ranking. All tree part point clouds come from dataset during the neural ranking process.
3. xxx.yml: Yaml file is a tree graph represent the connectivity of tree parts. This is a key file used for tree part connection and mesh generation. You need to pass this file to the tree mesh generation algorithm which has not been released yet. We will keep working on this section and update soon.

![process](./process.png)

### Tree Mesh Generation Pipeline
If you are interested in generating tree meshes from the tree graph, or if you are looking for further collaboration, please contact me by *detailzxc2010@gmail.com* and my PhD advisor *bbenes@purdue.edu*. We would further share a codebase for the final generation process, or we would love to help generating reconstruction results for your data. Please share the point cloud file and yaml file if you would like to get generated results from us.

### License
Our code is released under MIT License (see LICENSE file for details).

## TODO
- [] Code release for tree connection lib and visualization
- [] Webpage for online reconstruction
