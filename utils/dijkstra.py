import numpy as np
from sklearn.metrics import pairwise_distances
from utils.utils import adaptive_sampling, slice_input_data, visualize_kmeans_dict
from IPython import embed
from utils.visualize import save_ply_with_color, save_ply
from tqdm import tqdm
import multiprocessing
from sklearn.cluster import DBSCAN
import sys


def treepart_denoise(pc, eps=None, min_samples=5):

    # save objs for comparison
    save_ply("denoise_before_pc.ply", pc)
    eps_dist = -1
    eps_set = 0.02
    cal_eps = None
    if eps is None:
        eps_dist = pairwise_distances(pc, pc)
        eps_dist += np.eye(eps_dist.shape[0]) * 1e8
        cal_eps = eps_dist.min() * 4

        eps = np.max([cal_eps, eps_set])
    
    # filter scattered points by DBSCAN
    clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(pc[:, :3])
    cleaned_pts = []
    noise_pts = []
    for i, key in enumerate(clustering.labels_):
        if key == -1:
            noise_pts.append(pc[i])
            continue
        cleaned_pts.append(pc[i])
    cleaned_pts = np.array(cleaned_pts)
    
    # save for comparison after denoising
    save_ply("denoised_pc.ply", cleaned_pts)

    return cleaned_pts, noise_pts, eps, cal_eps


def visualize_pts(sampled_pc, treepart_centers):
    color = np.zeros((sampled_pc.shape))
    colored_sampled_pc = np.concatenate([sampled_pc, color], axis=1)
    
    color = np.ones((treepart_centers.shape)) * 255
    color[:, 1:] = color[:, 1:] * 0
    colored_treepart_centers = np.concatenate([treepart_centers, color], axis=1)
    
    save_ply_with_color("colored_sampled_center_dji.ply", np.concatenate([colored_sampled_pc, colored_treepart_centers], axis=0))


def simple_closest_tree_center(tree_roots, treepart_centers):
    
    treepart_number = treepart_centers.shape[0]
    treepart_center_belonging = [0 for _ in range(treepart_number)]

    closest_root_idx = np.argmin(pairwise_distances(treepart_centers, tree_roots), axis=1)
    
    for i in range(treepart_number):
        candidate_idx = closest_root_idx[i]
        treepart_center_belonging[i] = [tree_roots[candidate_idx], candidate_idx]

    return treepart_center_belonging

    
def closest_tree_center(pc, tree_roots, treepart_centers, sampling_number=12000):
    
    # mark tree roots
    tree_number = tree_roots.shape[0]
    treepart_number = treepart_centers.shape[0]
    
    radius = (pc[:, 1].max() - pc[:, 1].min()) / 2
    #### process input pc #####
    
    if pc.shape[0] > sampling_number:
        pc = adaptive_sampling(pc, target_num=sampling_number, shuffle=False)
    data_slice_dict = slice_input_data(pc, tree_roots)
    visualize_kmeans_dict(data_slice_dict, 'dji_sub_sample.ply', ratio=1)
    
    print("Radius: ", radius)
    
    # build tree root dict
    tree_roots_dist = pairwise_distances(tree_roots, tree_roots)
    # find the max min of the root distance as threshold
    # if tree_roots_dist.shape[0] <= 6:
    #     dist_thr = np.inf
    # else:    
    #     dist_thr = np.min(tree_roots_dist, axis=1)[6]
    
    dist_thr = np.inf
    
    tree_root_closest_dict = dict()
    for i in range(tree_roots_dist.shape[0]):
        tree_root_closest_dict[i] = np.argsort(tree_roots_dist[i]).reshape(-1)[:6] # include itself

    # visualize_pts(pc, treepart_centers)
    
    sub_sampling_num=10000
    pc_list = []
    
    treepart_center_to_roots_dist = pairwise_distances(treepart_centers[:, ::2], tree_roots[:, ::2], n_jobs = -1)
    
    for treepart_idx, treepart_center in enumerate(tqdm(treepart_centers)):
        root_idx = np.argsort(treepart_center_to_roots_dist[treepart_idx])[0]
        root_candidates = tree_root_closest_dict[root_idx]
        curr_pc = []
        for key in root_candidates:
            curr_pc.append(data_slice_dict[key])
        curr_pc = np.concatenate(curr_pc, axis=0)
        if curr_pc.shape[0] > sub_sampling_num:
            curr_pc = adaptive_sampling(curr_pc, target_num=sub_sampling_num, shuffle=False)
        save_ply("sampled_center_dji.ply", curr_pc)
        
        # first: tree roots
        # middle: point clouds in the area
        # end: treepart center
        curr_pc = np.concatenate(([tree_roots, curr_pc, treepart_center[None,:]]), axis=0)
        
        # curr_pc_distance = pairwise_distances(curr_pc, curr_pc)
        pc_list.append(curr_pc)

    
    input_list = []
    threads_num = 60
    number_per_thread = int(np.ceil(treepart_centers.shape[0] / threads_num))
    for i in range(threads_num):
        input_list.append(tuple([pc_list[i*number_per_thread:(i+1)*number_per_thread], range(i*number_per_thread, (i+1)*number_per_thread), tree_number, dist_thr]))
    # input_list.append(tuple([pc_list[threads_num*number_per_thread:], range(threads_num*number_per_thread, treepart_number), tree_number, dist_thr]))
    
    print("Thread Number: ", threads_num)
    print("Tree part per thread: ", number_per_thread)
    treepart_center_belonging = [0 for _ in range(treepart_number)]
    
    #TODO: debug here with single thread
    # data_process(input_list[5])
    
    with multiprocessing.Pool(len(input_list)) as pool:
        for zip in pool.imap_unordered(data_process, input_list):
            for data in zip:
                treepart_center_idx, candidate_idx = data
                treepart_center_belonging[treepart_center_idx] = [tree_roots[candidate_idx], candidate_idx]

    return treepart_center_belonging


def data_process(zip):

    pc_list, batch_treepart_center_idx, tree_number, dist_thr = zip
    # the last elememt will be the treepart center
    result = []
    for i in tqdm(range(len(pc_list))):
        
        treepart_center_idx = batch_treepart_center_idx[i]
        curr_pc = pc_list[i]
        adj_matrix = np.sqrt(np.sum((curr_pc[:, None, :] - curr_pc[None, :, :]) ** 2, axis=-1))

        # the threshold for breaking graph is the min distance of the root distance
        adj_matrix[np.where(adj_matrix > dist_thr)] = -1

        candidate_idx = dijkstra(adj_matrix, start_node=-1, end_node_num=tree_number, start_point=curr_pc[-1], root_points=curr_pc[:tree_number])
        result.append([treepart_center_idx, candidate_idx])
        
        del adj_matrix
        del curr_pc
        
    
    return result


def dijkstra(adj_matrix, start_node, end_node_num, start_point, root_points):
    
    # Number of nodes in the graph
    augmented_adj_matrix = np.exp(adj_matrix*2)
    # augmented_adj_matrix = adj_matrix
    num_nodes = adj_matrix.shape[0]
    
    # Create lists to track distance and visited nodes
    augmented_distance = [np.inf] * num_nodes
    distance = [np.inf] * num_nodes
    visited = [False] * num_nodes
    iters = [0] * num_nodes
    max_interval = [0] * num_nodes

    # Set the distance of the start_node to itself as 0
    distance[start_node] = 0
    augmented_distance[start_node] = 0

    scatter_part = False
    for _ in range(num_nodes):
        # Find the node with the minimum distance from the set of unvisited nodes
        min_augmented_distance = np.inf
        min_distance = np.inf
        min_node = None
        for node in range(num_nodes):
            if not visited[node] and augmented_distance[node] < min_augmented_distance:
                min_augmented_distance = augmented_distance[node]
                min_distance = distance[node]
                min_node = node

        # Mark the selected node as visited
        ## if a tree part is a scattered part, assign the closest root
        if min_node is None:
            scatter_part = True
            print("scatter part")
            break
        
        visited[min_node] = True
        
        # Update distances of adjacent nodes
        for node in range(num_nodes):
            if not visited[node] and adj_matrix[min_node][node] > 0:
                augmented_new_distance = augmented_distance[min_node] + augmented_adj_matrix[min_node][node]
                new_distance = distance[min_node] + adj_matrix[min_node][node]
                if augmented_new_distance < augmented_distance[node]:
                    augmented_distance[node] = augmented_new_distance
                    distance[node] = new_distance
                    iters[node] = iters[min_node] + 1
        
    if scatter_part:
        augmented_distance[:end_node_num] = pairwise_distances(start_point[None, :], root_points)[0]
    
    return np.argmin(np.array(augmented_distance[:end_node_num]))


def find_closest_tree_center(pc, tree_roots, treepart_centers, sampling_number=12000, simple=False):
    
    if simple:
        return simple_closest_tree_center(tree_roots, treepart_centers)
    else:
        return closest_tree_center(pc, tree_roots, treepart_centers, sampling_number)


    
if __name__ == "__main__":
    
    import pickle
    with open('rscnn_dji.pkl', 'rb') as f:
        saved_data = pickle.load(f)
        print("pickle loaded")
    
    tree_center_belonging = find_closest_tree_center(saved_data['pc'], saved_data['root'], saved_data['identity'], saved_data['pc'].shape[0]//8)
    
    vis_dict = dict()
    tree_center_for_treepart = []
    for i, d in enumerate(tree_center_belonging):
        center, key = d
        tree_center_for_treepart.append(center)
        if key not in vis_dict:
            vis_dict[key] = []
        vis_dict[key].append(saved_data['valid_datalist'][i])
    
    for i, key in enumerate(vis_dict):
        vis_dict[key] = np.concatenate(vis_dict[key], axis=0)
    
    visualize_kmeans_dict(vis_dict, "tree_candidate_debug.ply")