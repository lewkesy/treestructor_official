import torch
from utils.visualize import save_ply_with_color, save_ply
from utils.utils import visualize_MST_list, adaptive_sampling, chamfer_distance_numpy, visualize_kmeans_dict, add_noise_by_height, reconstruction_project_ply_data, slice_input_data
from utils.visualize_Yshape import generate_cylinder
from utils.peak_cluster import DensityPeakCluster
import numpy as np
# from models.dir_point_transformer import BranchReconstruction
from models.dir_model import BranchReconstruction
from tqdm import tqdm
from IPython import embed
import os
import pickle
from plyfile import PlyData
from PIL import Image
from sklearn.metrics import pairwise_distances
from sklearn.cluster import DBSCAN
import math
from utils.dijkstra import find_closest_tree_center
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R
import argparse
from IPython import embed
from numba import njit, prange
    
os.environ["OMP_NUM_THREADS"] = "64"

def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm

def find_rotation_matrix(v, u):
    v = np.array(v)
    u = np.array(u)
    
    # Compute cross product and normalize
    k = normalize(np.cross(v, u))

    # Compute angle between vectors
    theta = np.arccos(np.dot(normalize(v), normalize(u)))

    # Compute skew-symmetric cross-product matrix
    K = np.array([[0, -k[2], k[1]],
                  [k[2], 0, -k[0]],
                  [-k[1], k[0], 0]])

    # Compute rotation matrix using Rodrigues' rotation formula
    rotation = np.eye(3) + np.sin(theta) * K + (1 - np.cos(theta)) * np.dot(K, K)

    return rotation

def rotate_augmentation_process(pc, treepart_info, real_pc, pred_dir, relative_pos):
    lookup_dir = normalize(treepart_info['main_dir'])
    
    #TODO: find the relative dir
    # pred_dir = relative_pos - np.array([0, -1, 0])
    pred_dir = normalize(pred_dir)
    
    # rotation_matrix = find_rotation_matrix(lookup_dir, pred_dir)
    rotation_matrix = np.eye(3)
    initial_guess = np.array(R.from_matrix(rotation_matrix).as_euler('xyz', degrees=True).tolist() + [0, 0, 0, 1])
    
    # Define the objective function
    def objective_function(params):
        # Reshape the parameters into a 3x3 rotation matrix
        rotation_matrix = R.from_euler('xyz', params[:3]).as_matrix()
        transform_vector = params[3:6]
        scale = params[-1]
        
        # Apply the rotation
        augmented_pc = np.dot(rotation_matrix, pc.T).T * scale + transform_vector
        rotate_dir = np.dot(rotation_matrix, lookup_dir)
        
        # Calculate the difference
        difference_cd = chamfer_distance_numpy(augmented_pc, real_pc)
        difference_dir = (1 - np.dot(normalize(rotate_dir), normalize(pred_dir))) * 5
        
        # difference_bbox = np.sum(abs(np.max(augmented_pc, axis=0) - np.max(real_pc, axis=0)) + abs(np.min(augmented_pc, axis=0) - np.min(real_pc, axis=0)))
        
        return difference_cd + difference_dir # + difference_bbox
        # return difference_cd # + difference_bbox

    result = minimize(objective_function, initial_guess, method='BFGS', options={'maxiter':10})
    rotation_matrix = R.from_euler('xyz', result.x[:3]).as_matrix()
    transform_vector = result.x[3:6]
    scale = result.x[-1]
    
    # rotate point clouds and rescale point clouds
    augmented_pc = np.dot(rotation_matrix, pc.T).T * scale + transform_vector
    treepart_info['main_dir'] = normalize(np.dot(rotation_matrix, lookup_dir))
        
    for key in treepart_info['feature']:
        treepart_info['feature'][key]['Start Position'] = (np.dot(rotation_matrix, treepart_info['feature'][key]['Start Position'])) * scale + transform_vector
        treepart_info['feature'][key]['End Position'] = (np.dot(rotation_matrix, treepart_info['feature'][key]['End Position'])) * scale + transform_vector
        treepart_info['feature'][key]['Start Direction'] = normalize(np.dot(rotation_matrix, treepart_info['feature'][key]['Start Direction']))
        treepart_info['feature'][key]['End Direction'] = normalize(np.dot(rotation_matrix, treepart_info['feature'][key]['End Direction']))
        treepart_info['feature'][key]['Start Thickness'] *= scale
        treepart_info['feature'][key]['End Thickness'] *= scale
        
    
    return augmented_pc, treepart_info.copy(), chamfer_distance_numpy(real_pc, augmented_pc)
    

def visualize_foliage(filepath, data_list, tree_foliage):
    color_pc = []
    for i, foliage in enumerate(tree_foliage):
        curr_pc = np.array(data_list[i])
        if foliage:
            color = np.array([[0, 255, 0] for _ in range(curr_pc.shape[0])])
        else:
            color = np.array([[255, 0, 0] for _ in range(curr_pc.shape[0])])

        color_pc.append(np.concatenate([curr_pc, color], axis=1))
        
    save_ply_with_color(filepath, np.concatenate(color_pc, axis=0))
    
    
def load_data(filepath, adding_noise, scale):
    if filepath.split('.')[-1] == 'ply':
        data = PlyData.read(filepath)
        x = data["vertex"]["x"]
        y = data["vertex"]["y"]
        z = data["vertex"]["z"]
        pc = np.stack([x, y, z]).T
    else:
        data = np.loadtxt(filepath)
        pc = data[:, :3]
        radius_noise_ratio = data[:, 3]
        
        global_offset = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2
        pc -= global_offset
        global_ratio = np.max(abs(pc[:, 1]))
        pc /= global_ratio
        
        radius_weighted_noise = np.random.rand(*(pc.shape)) / ((1-radius_noise_ratio) * 25 + 50)[:, None]
        noise_pc = pc + radius_weighted_noise
        pc = np.concatenate([pc, noise_pc], axis=0)
        
        pc = pc * global_ratio + global_offset  
    
    # global_offset = (np.max(pc, axis=0) + np.min(pc, axis=0)) / 2
    # pc -= global_offset
    global_ratio = np.max(abs(pc[:, 1]))
    pc /= global_ratio
    pc *= scale
    
    save_ply("input.ply", pc)
    
    return pc


def slice_input_data_process(zip):
    pc, x_min, x_max, z_min, z_max, grid_x, grid_z, x_interval, z_interval = zip
    data_slice = [[] for _ in range(grid_x * grid_z)]
    
    for pts in pc:
        x_idx = int((pts[0]-x_min) / x_interval)
        z_idx = int((pts[2]-z_min) / z_interval)
        data_slice[z_idx * grid_x + x_idx].append(pts)
    
    return data_slice

    
def treepart_denoise(pc, eps=None, min_samples=5):
    
    # return pc, [], eps, None

    save_ply("denoise_before_pc.ply", pc)
    eps_dist = -1
    eps_set = 0.02
    cal_eps = None
    if eps is None:
        eps_dist = pairwise_distances(pc, pc)
        eps_dist += np.eye(eps_dist.shape[0]) * 1e8
        cal_eps = eps_dist.min() * 4

        eps = np.max([cal_eps, eps_set])
    
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pc[:, :3])
    cleaned_pts = []
    noise_pts = []
    for i, key in enumerate(clustering.labels_):
        if key == -1:
            noise_pts.append(pc[i])
            continue
        cleaned_pts.append(pc[i])
    cleaned_pts = np.array(cleaned_pts)
    save_ply("denoised_pc.ply", cleaned_pts)

    return cleaned_pts, noise_pts, eps, cal_eps


def find_tree_center_points(raw_pc, ratio=0.05, h_ratio=0.005, single_root=False):
    
    potential_center_points = []
    
    pc = raw_pc.copy()
    y_len = pc[:, 1].max() - pc[:, 1].min()
    potential_root_points_idx = np.where(pc[:, 1] < (pc[:, 1].min() + y_len * h_ratio))
    potential_root = pc[potential_root_points_idx]

    clustering = DBSCAN(eps=y_len* ratio, min_samples=5).fit(potential_root)
    cluster_dict = {}
    max_num = -1
    max_label = -1
    
    for i, key in enumerate(clustering.labels_):
        if key == -1:
            continue
        if key not in cluster_dict:
            cluster_dict[key] = []
        cluster_dict[key].append(potential_root[i])
        
        if max_num < potential_root[i].shape[0]:
            max_num = potential_root[i].shape[0]
            max_label = key
            
    if single_root:
        key_list = list(cluster_dict.keys())
        for key in key_list:
            if key != max_label:
                del cluster_dict[key]
        
    color_dict = visualize_kmeans_dict(cluster_dict, "candidate_roots.ply")
    
    tree_root_radius_dict = dict()

    tree_root_dict = dict()
    for key in cluster_dict:
        pts = np.array(cluster_dict[key])
        center_pts = np.mean(pts, axis=0)
        tree_root_dict[key] = center_pts.copy()
        
        # calculate potential tree center
        center_pts[1] = (pc[:, 1].max() + pc[:, 1].min()) / 2
        potential_center_points.append(center_pts)
    
        # calculate root radius for tree part clustering
        x_radius = (pts[:, 0].max() - pts[:, 0].min()) / 2
        z_radius = (pts[:, 2].max() - pts[:, 2].min()) / 2
        tree_root_radius_dict[key] = min(max(x_radius, z_radius, 0.5), 1)
        # tree_root_radius_dict[key] = max(x_radius, z_radius)

    return np.array(potential_center_points), tree_root_dict, tree_root_radius_dict, color_dict


def add_noise_for_sync_data(origin_data, pc):
    radius_noise_ratio = origin_data[:, 3]
    radius_weighted_noise = np.random.rand(*(pc.shape)) / ((1-radius_noise_ratio) * 25 + 50)[:, None]
    noisy_pc = pc + radius_weighted_noise
    
    return noisy_pc


def segmentation_pointcloud(pc, seg_sample_num, root_radius, compression=1, adding_noise=False, denoise=False):
    # Compress ratio: resize ratio for multiple trees. Tree will alwayas have y axis as the higest dim
    
    global_offset = (np.min(pc, axis=0) + np.max(pc, axis=0)) / 2
    pc -= global_offset
    global_ratio = abs(pc).max() * compression
    pc /= global_ratio  
            
    ####################   generate sampling from input point clouds ###########################

    # remove scatter points    
        
    if denoise:
        print("Desnoising, eps is: ", min(0.03, 0.5*(root_radius/global_ratio)))
        denoised_pc, _, _, _ = treepart_denoise(pc, eps=min(0.03, root_radius/global_ratio), min_samples=5)

        print("Denoised pc num: ", denoised_pc.shape[0])
        pc = denoised_pc
        
    if adding_noise:
        print("Adding Noise")
        save_ply("add_noise_before.ply", pc)
        noisy_pc = add_noise_by_height(pc, ratio=0.025, noise_division=20000/pc.shape[0])
        save_ply("add_noise.ply", noisy_pc)

        pc = noisy_pc
        
    if pc.shape[0] == 0:
        print("No points Issue Detected")
        return None, None, None

    # sampling
    print("Start Sampling Process")
    if pc.shape[0] > seg_sample_num:
        pc = adaptive_sampling(pc, target_num=seg_sample_num, shuffle=True, random=False)
        # sample_idx = np.random.choice(pc.shape[0], seg_sample_num)
        # pc = pc[sample_idx]
        save_ply("sampled_pc.ply", pc)
        
    # adding noise for sync data
    np.random.shuffle(pc)
    
    print("Denoised PC size: ", pc.shape[0])
    print("PC size: ", pc.shape[0])
    save_ply("./sampled_result.ply", np.array(pc))
    
    pc = pc * global_ratio  + global_offset

    return pc


def density_cluster(pc, root_radius, threshold, dist_cutoff_ratio, density_cutoff_ratio, if_junction):
    
    res = []
    peak_cluster = DensityPeakCluster()
    if pc.shape[0] != 0:
        cluster_dict = peak_cluster(pc,
                                    # threshold=max(root_radius, 0.03), 
                                    threshold=threshold, #0.8 small tree # 1 sync #1.35 real forest
                                    height=np.max(pc[:, 1])-np.min(pc[:, 1]),
                                    dist_cutoff_ratio=dist_cutoff_ratio, # 0.05 sync # real 0.045, 0.0275 campus full
                                    density_cutoff_ratio=density_cutoff_ratio, 
                                    noise_cutoff_ratio=0, 
                                    max_dist_cutoff_ratio=0,
                                    junction=if_junction)
        
        for i, cls in enumerate(cluster_dict):
            pts = np.array(cluster_dict[cls])
            res.append(pts)  
    
    del peak_cluster
    
    return res

            
def point_cloud_decomposition(pc, root_radius):

    noise_pc = []
    data_list = []
    
    
    cleaned_pts, noise_pts, denoise_eps, _ = treepart_denoise(pc[:, :3], eps=min(0.03, root_radius), min_samples=1)
    noise_pc += noise_pts
    print("MST finish")
    
    print("junction points num: ", cleaned_pts.shape[0])
    print("dist threshold: ", denoise_eps)
    print("radius: ", root_radius)
    print("Threshold: ", max(root_radius, denoise_eps))
    
    if cleaned_pts.shape[0] != 0:
        # data_list = density_cluster(cleaned_pts, root_radius, threshold=root_radius * 1.15, dist_cutoff_ratio=0.04, density_cutoff_ratio=0, if_junction=True)
        # data_list = density_cluster(cleaned_pts, root_radius, threshold=root_radius * 2.35, dist_cutoff_ratio=0.04, density_cutoff_ratio=0, if_junction=True)
        data_list = density_cluster(cleaned_pts, root_radius, threshold=root_radius * 1.75, dist_cutoff_ratio=0.035, density_cutoff_ratio=0, if_junction=True)

    print("Junction MST Done, data length: ", len(data_list))

    return data_list, noise_pc
    
    
def main():
    
    os.system("rm -r visualize/treepart/*")
    os.system("rm -r visualize/treepart_pc/*")
    os.system("rm -r visualize/raw_treepart_pc/*")
    os.system("rm -r visualize/treepart_projector/*")
    os.system("rm -r visualize/tmp/*")
    os.system("rm -r *.ply")

    parser = argparse.ArgumentParser()
    parser.add_argument('--filename', default='.', type=str)
    parser.add_argument('--visualize_candidate', action='store_true')
    parser.add_argument('--adding_noise', action='store_true')
    parser.add_argument('--denoise', action='store_true')
    parser.add_argument('--rotate_augmentation', action='store_true')
    parser.add_argument('--scale', type=float, default=10)
    parser.add_argument('--candidate_num', type=int, default=1)
    parser.add_argument('--sample_num', type=int, default=30000)
    parser.add_argument('--num_per_species', type=int, default=10)

    args = parser.parse_args()
    
    device = torch.device("cuda")

    os.makedirs("./visualize", exist_ok=True)
    os.makedirs("./visualize/raw_treepart_pc", exist_ok=True)
    os.makedirs("./visualize/treepart", exist_ok=True)
    os.makedirs("./visualize/treepart_pc", exist_ok=True)
    os.makedirs("./visualize/chamfer_distance", exist_ok=True)
    os.makedirs("./visualize/tmp", exist_ok=True)
    os.makedirs("./visualize/treepart_candidate", exist_ok=True)
    os.makedirs("./visualize/treepart_projector", exist_ok=True)
        
    # os.system("rm *.ply")
    # root = "/data/zhou1178/TreePointCloudDataFoliage/"
    # root = '/data/zhou1178/TreePointCloudDataNoise'
    root = '/data/zhou1178/TreeStructorData'
    
    ply_src = "./data"
    filename = args.filename
    pc_size = 500
    pc_lower_bound = 10
    seg_sample_num = args.sample_num
    batch_size = 100
    tree_num = 12000
    candidate_num = args.candidate_num
    num_per_species = args.num_per_species
    
    # If visualize the generated model
    save_mesh = True
    denoise = args.denoise
    visualize_candidate = args.visualize_candidate
    adding_noise = args.adding_noise
    rotate_augmentation = args.rotate_augmentation


    pickle_name = "RSCNNTreePartFeatureMainDirconstant%dNoNoiseNumPerSpecies%d.pkl"%(tree_num, num_per_species)
    # pickle_name = "RSCNNTreePartFeatureMainDirconstant%dAddNoiseNumPerSpecies%d.pkl"%(tree_num, num_per_species)
    
    # checkpoint_path = "./checkpoints/foliage_noise6400_1000_0.0_1.0_RSCNN_constant/epoch=169-step=234885.ckpt"
    # checkpoint_path = "./checkpoints/foliage6400_1000_0.0_1.0_RSCNN_constant/epoch=144-step=212660.ckpt"
    checkpoint_path = './checkpoints/treestructor12000_1000_0.0_1.0_RSCNN_constant/epoch=39-step=127360.ckpt'
    

    print("Filename: ", filename)
    data_info = 'TreePartInfo%d'%(tree_num)
    padding = "constant" if "constant" in pickle_name else "sample"
    print("Padding strategy: ", padding)

    forest_pc = load_data(os.path.join(ply_src, filename), adding_noise, args.scale)
    print("data loaded")
    
    tree_center_point_candidates, tree_root_dict, tree_root_radius_dict, color_dict = find_tree_center_points(forest_pc, ratio=0.015, single_root=False)
    
    # HACK: visualization
    # if len(tree_root_radius_dict) > 1 or forest_pc.shape[0] < 20000:
    #     import sys
    #     sys.exit()
    
    os.system("rm -r ./results/%s/treepart"%filename.split('.')[0])
    os.system("rm -r ./results/%s/raw_treepart_pc"%filename.split('.')[0])
    os.makedirs("./results/%s"%filename.split('.')[0], exist_ok=True)
    os.makedirs("./results/%s/treepart"%filename.split('.')[0], exist_ok=True)
    os.makedirs("./results/%s/raw_treepart_pc"%filename.split('.')[0], exist_ok=True)
    os.makedirs("./visualize/raw_treepart_pc", exist_ok=True)
    
    print("Find root forest")
    save_ply("./results/%s/vis_pc_ori_%s.ply"%(filename.split('.')[0], filename), forest_pc)
    
    
    # slice forest 
    # slice_pc_dict = slice_input_data(forest_pc, center_points=tree_center_point_candidates, pieces=slice_piece)
    slice_pc_dict = slice_input_data(forest_pc, center_points=tree_center_point_candidates)
    color_dict = visualize_kmeans_dict(slice_pc_dict, './results/%s/sliced_forest_%s.ply'%(filename.split('.')[0], filename), ratio=1, color_dict=color_dict)
    print("Slice forest")
    
    # load look up dataset
    with open(os.path.join(root, pickle_name), 'rb') as f:
        info = pickle.load(f)
    
        treepart_feature = info['feature']
        names = info['name']
        filepaths = info['filepaths']
        check_foliage = info['if_foliage']
        data_treepart_idx = info['treepart_idx']
        data_treepart_item_idx = info['treepart_item_idx']
        print("Load normal dataset")
    
    print("Pickle File Loaded: %s"%(pickle_name))
    
    # load model
    embedding_model = BranchReconstruction.load_from_checkpoint(checkpoint_path).to(device)
    embedding_model.eval()
    print("Model loaded")
    
    raw_pc_offset = []
    raw_pc_ratio = []
    raw_pc_list = []
    pc_center_offset = []
    treepart_identity_point = []
    pc_points = []
    normalized_pc_points = []
    filtered_points = []
    valid_datalist = []
    total_datalist = []
    junction_list = []
    branch_list = []
    species_color = dict()
    
    global_offset = (np.min(forest_pc, axis=0) + np.max(forest_pc, axis=0)) / 2
    normalized_forest_pc = forest_pc.copy() - global_offset
    global_ratio = abs(normalized_forest_pc[:, 1]).max()
    
    for k in tqdm(slice_pc_dict):
        
        ##################  Segmentation section  ##################################
        pc = slice_pc_dict[k].copy()
        print(pc.mean(0), pc.min(0), 0)
        root_radius = tree_root_radius_dict[k]
        pc = segmentation_pointcloud(pc, seg_sample_num, root_radius, compression=1, adding_noise=False, denoise=False)
        
        # ###############  remove points close to junction  ####################
        print("Re-allocate point clouds")
        
        curr_root_pts = tree_root_dict[k].copy()
        
        global_offset = curr_root_pts
        global_offset[1] = 0
        
        pc -= global_offset
        pc /= global_ratio  
        root_radius /= global_ratio
        
        print(pc.mean(0), pc.min(0), 2)
        
        print("Global offset: ", global_offset)
        print("Global ratio: ", global_ratio)

        ####################   pc decomposition ###########################  
        data_list, noise_pc = point_cloud_decomposition(pc, root_radius)
        
        print("Start Embedding Process")
            
        # add noise pts
        if len(noise_pc) > 0:
            filtered_points += (np.array(noise_pc) * global_ratio + global_offset).tolist()
        
        for raw_data_idx, data in enumerate(data_list):
            
            raw_points = np.array(data)
            raw_points = raw_points * global_ratio + global_offset
            total_datalist.append(raw_points)
            
            # check if in scatter points
            if raw_points.shape[0] < pc_lower_bound:
                filtered_points += raw_points.tolist()
                continue
        
            # save center offset in normalized whole tree position
            pc_center_offset.append(np.mean(raw_points, axis=0))
            treepart_identity_point.append(raw_points[np.argsort(raw_points[:, 1])[raw_points.shape[0]//2]])
            valid_datalist.append(raw_points)
            
            # offset and ratio here normalize the tree part into uniform space for further shifting
            offset = (np.max(raw_points, axis=0) + np.min(raw_points, axis=0)) / 2
            normalized_points = raw_points - offset
            ratio = global_ratio
            normalized_points /= ratio
            
            if padding == 'constant':
                np.random.shuffle(normalized_points)
                center = np.zeros((3,))
                if pc_size >= normalized_points.shape[0]:
                    points = np.stack([center for _ in range(pc_size)])
                    points[:normalized_points.shape[0]] = normalized_points
                else:
                    points = normalized_points[:pc_size]
            elif padding == 'sample':
                if pc_size >= normalized_points.shape[0]:
                    sample_idx = np.random.choice(normalized_points.shape[0], pc_size-normalized_points.shape[0])
                    points = np.concatenate([normalized_points, normalized_points[sample_idx]], axis=0)
                else:
                    sample_idx = np.random.choice(normalized_points.shape[0], pc_size, replace=False)
                    points = normalized_points[sample_idx]
                np.random.shuffle(points)
            
            raw_pc_offset.append(offset)
            raw_pc_ratio.append(ratio)
            raw_pc_list.append(raw_points)
            normalized_pc_points.append(normalized_points)
            pc_points.append(points)


    colored_pc = []
    for treepart in junction_list:
        for i in range(treepart.shape[0]):
            colored_pc.append([treepart[i][0], treepart[i][1], treepart[i][2], 255, 0, 0])
    for treepart in branch_list:
        for i in range(treepart.shape[0]):
            colored_pc.append([treepart[i][0], treepart[i][1], treepart[i][2], 0, 0, 255])
    save_ply_with_color("./results/%s/recon_reassign_segmentation.ply"%filename.split('.')[0], np.array(colored_pc))
    
    print("The number of samples is %d"%len(pc_points))
    save_ply("scatter.ply", np.array(filtered_points))
    
    pc_points = np.array(pc_points).astype(float)
    
    color_list = visualize_MST_list(valid_datalist, fn="./results/%s/MST_seg_%s.ply"%(filename.split('.')[0], filename), threshold=pc_lower_bound)   
    visualize_MST_list(total_datalist, fn="./results/%s/MST_seg_ori_%s.ply"%(filename.split('.')[0], filename), threshold=0)  
    
    
     # find the corresponding tree center
    treepart_identity_point = np.array(treepart_identity_point)
    # print(tree_center_point_candidates)
    
    
    ###################### Djistra  ###################################
    print("Test djikstra")
    
    forest_lookup_data = np.concatenate(total_datalist, axis=0)
    save_ply("./results/%s/%s_pc.ply"%(filename.split('.')[0], filename), forest_lookup_data/10)
    
    save_ply("dji_identity", treepart_identity_point)
    tree_roots = np.array([tree_root_dict[key] for key in tree_root_dict])
        
    tree_root_belonging = find_closest_tree_center(forest_lookup_data,
                                                     tree_roots,
                                                     treepart_identity_point,
                                                     sampling_number=forest_lookup_data.shape[0]//8,
                                                     simple=True) #8

    # cluster by djikstra algorithm
    treeparts_by_tree_root = dict()
    for i, d in enumerate(tree_root_belonging):
        center, key = d
        if key not in treeparts_by_tree_root:
            treeparts_by_tree_root[key] = []
        treeparts_by_tree_root[key].append(valid_datalist[i])

    # find the tree centers based on the clustering results
    tree_centers = dict()
    for i, key in enumerate(treeparts_by_tree_root):
        treeparts_by_tree_root[key] = np.concatenate(treeparts_by_tree_root[key], axis=0)
        # tree_centers[key] = (np.max(treeparts_by_tree_root[key], axis=0) + np.min(treeparts_by_tree_root[key], axis=0)) / 2
        tree_centers[key] = tree_roots[key]
        tree_centers[key][1] = (np.max(treeparts_by_tree_root[key][:, 1]) + np.min(treeparts_by_tree_root[key][:, 1])) / 2
    
    visualize_kmeans_dict(treeparts_by_tree_root, "./results/%s/tree_candidate_%s.ply"%(filename.split('.')[0], filename), color_dict=color_dict)
    
    # calculate tree center
    tree_center_for_treepart = []
    for i, d in enumerate(tree_root_belonging):
        center, key = d
        tree_center_for_treepart.append(tree_centers[key])

    tree_center_for_treepart = np.array(tree_center_for_treepart)
    treepart_center_offset = np.array(pc_center_offset) - tree_center_for_treepart
    treepart_center_offset /= global_ratio

    # for key in treeparts_by_tree_root:
    #     save_ply("tree_candidate_%s_%d.ply"%(filename, key), treeparts_by_tree_root[key])

    pc_points = torch.Tensor(pc_points).to(device)
    treepart_center_offset = torch.Tensor(treepart_center_offset).to(device)
    
    # embedding, recon_pc, pred_cls, pred_dirs, pred_radius = embedding_model(pc_points)
    print("Input pc size: ", pc_points.shape[0])
    embedding = []
    pred_dirs = []
    
    batch_num = pc_points.shape[0] // batch_size + 1 if pc_points.shape[0] % batch_size != 0 else pc_points.shape[0] // batch_size
    for batch_idx in tqdm(range(batch_num)):
        curr_embedding, recon_pc, _, curr_pred_dirs, _ = embedding_model(pc_points[batch_size*batch_idx:batch_size*(1+batch_idx)], 
                                                                                treepart_center_offset[batch_size*batch_idx:batch_size*(1+batch_idx)])
        embedding.append(curr_embedding.detach().cpu().numpy())
        pred_dirs.append(curr_pred_dirs.detach().cpu().numpy())
    
    embedding = np.concatenate(embedding, axis=0)
    pred_dirs = np.concatenate(pred_dirs, axis=0)
    pc_points = pc_points.detach().cpu().numpy()
    recon_pc = recon_pc.detach().cpu().numpy()
    treepart_center_offset = treepart_center_offset.detach().cpu().numpy()
    
    # visualize the gt pts with dir color
    color_by_dir = ((1 + pred_dirs / np.sqrt(np.sum(pred_dirs**2, axis=1, keepdims=True))) / 2 * 255.).astype(int)
    
    ####################   calculate feature matching ###########################
    print("Calculate the mapping")

    # treepart_feature = np.array(treepart_feature)
    dist = np.zeros((embedding.shape[0], len(treepart_feature)))
    mapping_batch_size = 50
    iter_nums = math.ceil(len(treepart_feature) / mapping_batch_size)

    for i in tqdm(range(iter_nums)):
        treepart_feature_batch = np.array(treepart_feature[i*mapping_batch_size: (i+1)*mapping_batch_size])
        # dist.append(pairwise_distances(embedding, treepart_feature_batch, metric='l2'))
        dist_per_batch = pairwise_distances(embedding, treepart_feature_batch, metric='l2')
        dist[:, i*mapping_batch_size: (i+1)*mapping_batch_size] += dist_per_batch.astype(np.float16)
    
    
    # dist[branch_idx, :gallery_junction_branch_spliter_index] += 1e8
    # dist[:junction_branch_split_index, gallery_junction_branch_spliter_index:] += 1e8
    print("Latent dist prepared")
    
    @njit(parallel=True)
    def fast_argsort(a):
        sorted_indices = np.empty_like(a, dtype=np.int64)
        for i in prange(a.shape[0]):  # Loop over rows in parallel
            sorted_indices[i] = np.argsort(a[i])
        return sorted_indices
    
    # part_sorted_indices = np.argpartition(dist, candidate_num, axis=1)[:, :candidate_num]  # Fast
    # sorted_idx = part_sorted_indices[np.arange(dist.shape[0])[:, None], np.argsort(a[np.arange(dist.shape[0])[:, None], part_sorted_indices])]
    
    # sorted_idx = []
    # for curr_dist in tqdm(dist):
    #     sorted_idx.append(np.argsort(curr_dist)[:candidate_num])
    
    sorted_idx = fast_argsort(dist)[:, :candidate_num]
    del dist
    # matching_idx = np.argmin(dist, axis=1).tolist()
    
    skeleton = []
    new_tree_list = []
    treepart_spline = []
    treepart_spline_dir = []
    treepart_spline_radius = []
    treepart_spline_color = []
    treepart_chamfer_distance = []
    tree_part_dir = []
    tree_foliage = []
    
    image_iter = 0
    image_gallery = []
    
    name_idx_record = np.zeros((len(treepart_feature),))
    print("Generation")
    for i, matching_idx in enumerate(tqdm(sorted_idx)):
        min_chamfer_dist = 1e8
        selected_idx = 0
        for curr_idx, name_idx in enumerate(matching_idx):
            
            filepath = filepaths[name_idx]
            treepart_idx = data_treepart_idx[name_idx]
            treepart_item_idx = data_treepart_item_idx[name_idx]
            
            with open(filepath, 'rb') as f:
                curr_data = pickle.load(f)[treepart_idx][treepart_item_idx]
                
            species = names[name_idx].split('_')[0] + str(int(names[name_idx].split('.')[0].split('_')[1])//900)

            # tree_foliage.append(check_foliage[name_idx])
            
            with open(os.path.join(root, data_info, names[name_idx].split('.')[0]+'.pkl'), 'rb') as f:
                curr_treepart_info = pickle.load(f)[treepart_idx][treepart_item_idx]

            curr_pc = curr_data[:, :3]
            
            if rotate_augmentation:
                curr_pc, curr_treepart_info, curr_chamfer_dist = rotate_augmentation_process(curr_pc, curr_treepart_info, normalized_pc_points[i], pred_dirs[i], treepart_center_offset[i])
                if curr_chamfer_dist < min_chamfer_dist:
                    min_chamfer_dist = curr_chamfer_dist
                    pc = curr_pc
                    treepart_info = curr_treepart_info
                    selected_idx = curr_idx
                    data = curr_data
                    selected_name_idx = name_idx
            else:
                min_chamfer_dist = chamfer_distance_numpy(curr_pc, normalized_pc_points[i])
                pc = curr_pc
                treepart_info = curr_treepart_info
                selected_idx = curr_idx
                data = curr_data
                selected_name_idx = name_idx
                
            if visualize_candidate:
                curr_image = reconstruction_project_ply_data(curr_pc)[:, :, :3]
                im = Image.fromarray(curr_image.astype(np.uint8))
                im.save('./visualize/tmp/%d_%d.png'%(i, curr_idx))
                
                curr_image = reconstruction_project_ply_data(normalized_pc_points[i])[:, :, :3]
                im = Image.fromarray(curr_image.astype(np.uint8))
                im.save('./visualize/tmp/ori_%d.png'%(i))
            
            pc_by_index = np.array([curr_pc.min(0), curr_pc.max(0)]) * 1.25
            
            # save_ply("./visualize/tmp/top_%d_%d_treepart_%s_pc.ply"%(i, curr_idx, names[name_idx].split('.')[0]), curr_pc)
            # generate_cylinder_vis(curr_treepart_info, pc_by_index, np.zeros((3,)), 1, "top_%d_%d_%s_%d"%(i, curr_idx, filename.split('.')[0], key), 
            #                       bound_check=True, if_save_mesh=False,
            #                       foliage=check_foliage[name_idx],
            #                       foliage_radius=root_radius*global_ratio)
                
               
        internode = data[:, -1].astype(int)
        tree_part_dir.append(treepart_info['main_dir'])
        treepart_chamfer_distance.append(min_chamfer_dist)
        filepath_idx_list = sorted_idx[i]
        name_idx_record[selected_name_idx] += 1
        tree_foliage.append(check_foliage[selected_name_idx])
        
        # denormalize This is used for visualize the rebuilt-kemans tree
        
        '''
                 visualize pc
        '''
        
        # visualize the selected tree parts
        vis_pc = pc
        if species not in species_color:
            species_color[species] = np.random.randint(128, size=3) + 127
        curr_color = species_color[species]
        vis_color = np.array([curr_color for _ in range(vis_pc.shape[0])])
        save_ply_with_color("./visualize/treepart_pc/top_%d_%s.ply"%(i, names[name_idx].split('.')[0]), 
                            np.concatenate([vis_pc, vis_color], axis=1))
        
        vis_pc = vis_pc * raw_pc_ratio[i] + raw_pc_offset[i]
        save_ply_with_color("./visualize/treepart_pc/%d_%s_ori.ply"%(i, names[name_idx].split('.')[0]), 
                            np.concatenate([vis_pc, vis_color], axis=1))
        
        new_tree_list.append(vis_pc)
        
        # visualize the original points
        ori_treepart = normalized_pc_points[i]
        vis_color = np.array([color_list[i] for _ in range(ori_treepart.shape[0])])
        save_ply_with_color("./visualize/raw_treepart_pc/%d_ori.ply"%(i), 
                            np.concatenate([ori_treepart* raw_pc_ratio[i] + raw_pc_offset[i], vis_color], axis=1))
        
        vis_color = np.array([color_list[i] for _ in range(ori_treepart.shape[0])])
        save_ply_with_color("./visualize/raw_treepart_pc/%d.ply"%(i), 
                            np.concatenate([ori_treepart, vis_color], axis=1))
        
        if species not in species_color:
            species_color[species] = np.random.randint(128, size=3) + 127
        curr_color = species_color[species]
        vis_color = np.array([curr_color for _ in range(ori_treepart.shape[0])])
        save_ply_with_color("./visualize/raw_treepart_pc/species_%d_%s.ply"%(i, names[name_idx].split('.')[0]), 
                            np.concatenate([ori_treepart* raw_pc_ratio[i] + raw_pc_offset[i], vis_color], axis=1))
        
        # pc_by_index = dict()
        # for index, internode_index in enumerate(internode):
        #     if internode_index not in pc_by_index:
        #         pc_by_index[internode_index] = []
        #     pc_by_index[internode_index] += [[pc[index, 0], pc[index, 1], pc[index, 2]]]
        
        raw_pc_normalize_offset = raw_pc_offset[i]
        raw_pc_normalize_ratio = raw_pc_ratio[i]
        # global offset, raw_input_normalize_offset, match_normalize_offset
        bounding_treepart_pc = (valid_datalist[i] - raw_pc_normalize_offset) / raw_pc_normalize_ratio
        
        pc_by_index = np.array([bounding_treepart_pc.min(0), bounding_treepart_pc.max(0)])
        
        color = color_list[i]
        
        curr_treepart_skeleton, curr_treepart_dir, curr_treepart_radius, curr_treepart_color = generate_cylinder(treepart_info, 
                                                                                            pc_by_index, 
                                                                                            raw_pc_normalize_offset, 
                                                                                            raw_pc_normalize_ratio,
                                                                                            "100_full_%s"%names[name_idx].split('.')[0],
                                                                                            filename.split('.')[0],
                                                                                            color,
                                                                                            i, 
                                                                                            save_mesh=False,
                                                                                            foliage=check_foliage[name_idx],
                                                                                            foliage_radius=root_radius*global_ratio
                                                                                            )
        
        
        # generate 70% bbox meshes
        # pc_by_index_offset = pc_by_index.mean(0)
        # pc_by_index = (pc_by_index - pc_by_index_offset) * 0.7 + pc_by_index_offset
        # generate_cylinder(treepart_info, 
        #                         pc_by_index, 
        #                         raw_pc_normalize_offset, 
        #                         # np.zeros((3,)),
        #                         raw_pc_normalize_ratio,
        #                         "%s"%names[name_idx].split('.')[0],
        #                         filename.split('.')[0],
        #                         color,
        #                         i, 
        #                         save_mesh=True,
        #                         foliage=check_foliage[name_idx],
        #                         foliage_radius=root_radius*global_ratio,
        #                         foliage_per_node=3)
        
        # filter tree parts with no info

        treepart_spline.append(curr_treepart_skeleton)
        treepart_spline_dir.append(curr_treepart_dir)
        treepart_spline_radius.append(curr_treepart_radius)
        # treepart_spline_color.append([int(np.max([int((1-treepart_info['Level']/5)*255.), 0])),
        #                               int(np.max([int((1-treepart_info['Level']/5)*255.), 0])), 
        #                               255])
        treepart_spline_color.append(curr_treepart_color)
    
    # visualization
    if visualize_candidate:
        # image = np.asarray(Image.open(image_path))
        
        image_gallery = []
        for img_x in range(len(sorted_idx)):
            curr_image_list = []
            curr_image = np.asarray(Image.open("./visualize/tmp/ori_%d.png"%(img_x)))[:, :, :3]
            curr_image_list.append(curr_image)
            for img_y in range(candidate_num):
                
                curr_image = np.asarray(Image.open("./visualize/tmp/%d_%d.png"%(img_x, img_y)))[:, :, :3]
                curr_image_list.append(curr_image)
            
            curr_image_list = np.concatenate(curr_image_list, axis=1)
            image_gallery.append(curr_image_list)
            
            if img_x % 10 == 9:
                image_gallery = np.concatenate(image_gallery, axis=0)
                im = Image.fromarray(image_gallery.astype(np.uint8))
                im.save('./visualize/treepart_projector/%d.png'%(img_x//10))
                image_gallery = []
        
        if len(image_gallery) != 0:
            image_gallery = np.concatenate(image_gallery, axis=0)
            im = Image.fromarray(image_gallery.astype(np.uint8))
            im.save('./visualize/treepart_projector/%d.png'%(img_x//10))
        
    tree_part_dir = np.array(tree_part_dir)
    color_by_dir = ((1 + tree_part_dir / np.sqrt(np.sum(tree_part_dir**2, axis=1, keepdims=True))) / 2 * 255.).astype(int)
    treepart_chamfer_distance = np.array(treepart_chamfer_distance)
    cd_max = np.max([treepart_chamfer_distance.max(), 0.05])
    print("cd max: ", cd_max)

    
    visualize_MST_list(new_tree_list, "./results/%s/rebuild_MST_%s.ply"%(filename.split('.')[0], filename), threshold=pc_lower_bound, color_list=color_list)
    visualize_MST_list(new_tree_list, "./results/%s/rebuild_MST_dir_%s.ply"%(filename.split('.')[0], filename), threshold=pc_lower_bound, color_list=color_by_dir)

    yaml_dict = {'Tree': {'Scatter Points': [[0,0,0]],
                        'Tree Parts': []}}
    
    # register scatter points
    for pts in filtered_points:
        if sum(pts) != 0:
            yaml_dict['Tree']['Scatter Points'].append([float(pts[0]), float(pts[1]), float(pts[2])])
    

    # register Treeparts
    print("raw pc num: ", len(raw_pc_list))
    print("Treepart num: ", len(treepart_spline))
    print("foliage: ", len(tree_foliage))
    
    for treepart_idx in range(len(raw_pc_list)):
        if len(treepart_spline_color[treepart_idx]) == 0:
            continue
        treepart_dict = {'treepart_id': int(treepart_idx),
                         'Allocated Points': [],
                        'Branches': [],
                        'foliage': 1 if tree_foliage[treepart_idx] else 0,
                        'Color': [int(treepart_spline_color[treepart_idx][0]), 
                                  int(treepart_spline_color[treepart_idx][1]), 
                                  int(treepart_spline_color[treepart_idx][2])]}
        

        # insert allocated points
        for pts in raw_pc_list[treepart_idx]:
            treepart_dict['Allocated Points'].append([float(pts[0]), float(pts[1]), float(pts[2])])

        for junction_idx in range(len(treepart_spline[treepart_idx])):
            start_pos, end_pos = treepart_spline[treepart_idx][junction_idx][0], treepart_spline[treepart_idx][junction_idx][-1]
            start_dir, end_dir = treepart_spline_dir[treepart_idx][junction_idx][0], treepart_spline_dir[treepart_idx][junction_idx][-1]
            start_radius, end_radius = treepart_spline_radius[treepart_idx][junction_idx][0], treepart_spline_radius[treepart_idx][junction_idx][-1]
            
            curr_node_dict = {
                'Start Pos': [float(start_pos[0]), float(start_pos[1]), float(start_pos[2])],
                'End Pos': [float(end_pos[0]), float(end_pos[1]), float(end_pos[2])],
                'Start Dir': [float(start_dir[0]), float(start_dir[1]), float(start_dir[2])],
                'End Dir': [float(end_dir[0]), float(end_dir[1]), float(end_dir[2])],
                'Start Radius': float(start_radius),
                'End Radius': float(end_radius),
            }
            
            treepart_dict['Branches'].append(curr_node_dict)
        
            
        # skip invalid tree parts
        if len(treepart_dict['Branches']) == 0:
            continue
        yaml_dict['Tree']['Tree Parts'].append(treepart_dict)
        
    
    import yaml
    with open("./results/%s/%s_%d.yml"%(filename.split('.')[0], filename.split('.')[0], num_per_species), 'w') as f:
        yaml.dump(yaml_dict, f, default_flow_style=None)
    print("Yaml saved")
    
    visualize_foliage("./results/%s/foliage_vis.ply"%(filename.split('.')[0]), data_list, tree_foliage)
    
    cd = chamfer_distance_numpy(forest_pc, np.concatenate(new_tree_list).reshape(-1, 3))
    print("Chamfer distance: ", cd)
        
    os.system("mv visualize/raw_treepart_pc ./results/%s"%filename.split('.')[0])
    
    with open("hist_%s_%d.pkl"%(filename.split('.')[0], num_per_species), 'wb') as f:
        pickle.dump(name_idx_record, f)
    
        
if __name__ == "__main__":
    main()