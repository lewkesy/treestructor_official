import torch.nn as nn
import torch
import numpy as np
from plyfile import PlyData, PlyElement
from extensions.chamfer_distance.chamfer_distance import ChamferDistance
# from extensions.earth_movers_distance.emd import EarthMoverDistance
from utils.visualize import save_ply_with_color, save_ply
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors
import os
from scipy.spatial.distance import cdist, pdist, squareform
from IPython import embed
from sklearn.metrics import pairwise_distances
import open3d as o3d
from torchvision import transforms
from tqdm import tqdm
from sklearn.cluster import DBSCAN

# from utils.chamfer3D import dist_chamfer_3D

# def calc_cd(output, gt, calc_f1=False, return_raw=False, normalize=False, separate=False):
#     # cham_loss = dist_chamfer_3D.chamfer_3DDist()
#     cham_loss = dist_chamfer_3D.chamfer_3DDist()
#     dist1, dist2, idx1, idx2 = cham_loss(gt, output)
#     cd_p = (torch.sqrt(dist1).mean(1) + torch.sqrt(dist2).mean(1)) / 2
#     cd_t = (dist1.mean(1) + dist2.mean(1))

#     if separate:
#         res = [torch.cat([torch.sqrt(dist1).mean(1).unsqueeze(0), torch.sqrt(dist2).mean(1).unsqueeze(0)]),
#                torch.cat([dist1.mean(1).unsqueeze(0),dist2.mean(1).unsqueeze(0)])]
#     else:
#         res = [cd_p, cd_t]
#     if return_raw:
#         res.extend([dist1, dist2, idx1, idx2])
#     return res

# def calc_dcd(x, gt, alpha=200, n_lambda=0.5, return_raw=False, non_reg=False):
#     x = x.float()
#     gt = gt.float()
#     batch_size, n_x, _ = x.shape
#     batch_size, n_gt, _ = gt.shape
#     assert x.shape[0] == gt.shape[0]

#     if non_reg:
#         frac_12 = max(1, n_x / n_gt)
#         frac_21 = max(1, n_gt / n_x)
#     else:
#         frac_12 = n_x / n_gt
#         frac_21 = n_gt / n_x

#     cd_p, cd_t, dist1, dist2, idx1, idx2 = calc_cd(x, gt, return_raw=True)
#     # dist1 (batch_size, n_gt): a gt point finds its nearest neighbour x' in x;
#     # idx1  (batch_size, n_gt): the idx of x' \in [0, n_x-1]
#     # dist2 and idx2: vice versa
#     exp_dist1, exp_dist2 = torch.exp(-dist1 * alpha), torch.exp(-dist2 * alpha)

#     loss1 = []
#     loss2 = []
#     for b in range(batch_size):
#         count1 = torch.bincount(idx1[b])
#         weight1 = count1[idx1[b].long()].float().detach() ** n_lambda
#         weight1 = (weight1 + 1e-6) ** (-1) * frac_21
#         loss1.append((- exp_dist1[b] * weight1 + 1.).mean())

#         count2 = torch.bincount(idx2[b])
#         weight2 = count2[idx2[b].long()].float().detach() ** n_lambda
#         weight2 = (weight2 + 1e-6) ** (-1) * frac_12
#         loss2.append((- exp_dist2[b] * weight2 + 1.).mean())

#     loss1 = torch.stack(loss1)
#     loss2 = torch.stack(loss2)
#     loss = (loss1 + loss2) / 2

#     res = [loss, cd_p, cd_t]
#     if return_raw:
#         res.extend([dist1, dist2, idx1, idx2])

#     return res

def treepart_denoise(pc, eps=None, min_samples=5, vis=True):

    if vis:
        save_ply("denoise_before_pc.ply", pc[:, :3])
    eps_dist = -1
    eps_set = 0.02
    cal_eps = None
    if eps is None:
        eps_dist = pairwise_distances(pc[:, :3], pc[:, :3])
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
    
    if vis:
        save_ply("denoised_pc.ply", cleaned_pts[:, :3])

    return cleaned_pts, noise_pts, eps, cal_eps


def find_tree_center_points(raw_pc, ratio=0.035):
    
    potential_center_points = []
    
    pc = raw_pc.copy()
    y_len = pc[:, 1].max() - pc[:, 1].min()
    potential_root_points_idx = np.where(pc[:, 1] < (pc[:, 1].min() + y_len* ratio))
    potential_root = pc[potential_root_points_idx]

    clustering = DBSCAN(eps=y_len/75, min_samples=5).fit(potential_root[:, :3])
    cluster_dict = {}
    for i, key in enumerate(clustering.labels_):
        if key == -1:
            continue
        if key not in cluster_dict:
            cluster_dict[key] = []
        cluster_dict[key].append(potential_root[i])
    
    tree_root_radius_dict = dict()
    for key in cluster_dict:
        pts = np.array(cluster_dict[key])
        center_pts = np.mean(pts, axis=0)
        
        # calculate potential tree center
        center_pts[1] = (pc[:, 1].max() + pc[:, 1].min()) / 2
        potential_center_points.append(center_pts)
    
        # calculate root radius for tree part clustering
        x_radius = (pts[:, 0].max() - pts[:, 0].min()) / 2
        z_radius = (pts[:, 2].max() - pts[:, 2].min()) / 2
        tree_root_radius_dict[key] = max(x_radius, z_radius)

    return np.array(potential_center_points), tree_root_radius_dict


def slice_input_data(pc, center_points, vis=True):

    # project 3D xyz to 2D
    project_pc = pc[:, :3][:, ::2]
    project_root = center_points[:, :3][:, ::2]
    
    dist = pairwise_distances(project_pc, project_root)
    belonging = np.argmin(dist, axis=1)

    data_dict = dict()
    if vis:
        for idx, belonging_idx in enumerate(tqdm(belonging)):
            if belonging_idx not in data_dict:
                data_dict[belonging_idx] = []
            data_dict[belonging_idx].append(pc[idx])
    else:
        for idx, belonging_idx in enumerate(belonging):
            if belonging_idx not in data_dict:
                data_dict[belonging_idx] = []
            data_dict[belonging_idx].append(pc[idx])
    
    for key in data_dict:
        data_dict[key] = np.array(data_dict[key])
        
    return data_dict


def vector_normalization(vector):
    if len(vector.shape) > 2:
        vector = vector.reshape(-1, 3)
    
    vector = vector/ torch.sqrt(torch.sum(vector ** 2, dim=1))[:, None]

    return vector


# loss zoo
def cosine_similarity(x1, x2):
    x1 = x1.reshape(-1, 3)
    x2 = x2.reshape(-1, 3)
    cos = nn.CosineSimilarity()
    return (1 - cos(x1, x2)).mean()

CD = ChamferDistance()
# EMD = EarthMoverDistance()


def cd_loss_L1(pcs1, pcs2):
    """
    L1 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    dist1 = torch.sqrt(dist1)
    dist2 = torch.sqrt(dist2)
    return (torch.mean(dist1) + torch.mean(dist2)) / 2.0


def cd_loss_L2(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return torch.mean(dist1) + torch.mean(dist2)


def chamfer_distance(pcs1, pcs2):
    """
    L2 Chamfer Distance.

    Args:
        pcs1 (torch.tensor): (B, N, 3)
        pcs2 (torch.tensor): (B, M, 3)
    """
    dist1, dist2 = CD(pcs1, pcs2)
    return dist1, dist2


def chamfer_distance_numpy(x, y, metric='l2', direction='bi'):
    """Chamfer distance between two point clouds
    Parameters
    ----------
    x: numpy array [n_points_x, n_dims]
        first point cloud
    y: numpy array [n_points_y, n_dims]
        second point cloud
    metric: string or callable, default ‘l2’
        metric to use for distance computation. Any metric from scikit-learn or scipy.spatial.distance can be used.
    direction: str
        direction of Chamfer distance.
            'y_to_x':  computes average minimal distance from every point in y to x
            'x_to_y':  computes average minimal distance from every point in x to y
            'bi': compute both
    Returns
        chamfer_distance. min dist for each point if direction is not bi
    -------
    chamfer_dist: float
        computed bidirectional Chamfer distance:
            sum_{x_i \in x}{\min_{y_j \in y}{||x_i-y_j||**2}} + sum_{y_j \in y}{\min_{x_i \in x}{||x_i-y_j||**2}}
    """
    
    if direction=='y_to_x':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        chamfer_dist = np.mean(min_y_to_x)
        return chamfer_dist, min_y_to_x
    elif direction=='x_to_y':
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_x_to_y)
        return chamfer_dist, min_x_to_y
    elif direction=='bi':
        x_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(x)
        min_y_to_x = x_nn.kneighbors(y)[0]
        y_nn = NearestNeighbors(n_neighbors=1, leaf_size=1, algorithm='kd_tree', metric=metric).fit(y)
        min_x_to_y = y_nn.kneighbors(x)[0]
        chamfer_dist = np.mean(min_y_to_x) + np.mean(min_x_to_y)
    else:
        raise ValueError("Invalid direction type. Supported types: \'y_x\', \'x_y\', \'bi\'")
        
    return chamfer_dist


# def EMD_loss(pcs1, pcs2):
#     """
#     EMD Loss.

#     Args:
#         xyz1 (torch.Tensor): (b, N, 3)
#         xyz2 (torch.Tensor): (b, N, 3)
#     """
#     dists = EMD(pcs1, pcs2)
#     return torch.mean(dists)


# load ply data

def read_ply(filepath, sync=False):
    with open(filepath, 'rb') as f:
        data = PlyData.read(filepath)
    
    x = data['vertex']['x'][:, None]
    y = data['vertex']['y'][:, None]
    z = data['vertex']['z'][:, None]

    if sync:
        isfork = np.array(data['junction']['value']) == 1
    else:
        isfork = data['vertex']['red'] == 255

    point = np.concatenate((x, y, z), axis=-1)
    label = isfork

    return point, label


def loadply(fn):
    
    with open(fn,"r") as freader:
        
        header=True
        vertices_count=0
        primitives_count=0
        while header:
            line = freader.readline()
            str=line.strip().split(' ')
            if str[0]=='element':
                if str[1]=='vertex':
                    vertices_count=int(str[2])
                elif str[1]=='primitive':
                    primitives_count=int(str[2])
            elif str[0]=='end_header':
                header=False
            #otherwise continue
        pointset=[]


        for i in range(vertices_count):
            line = freader.readline()
            numbers=line.strip().split(' ')
            pt=[]
            for n in numbers:
                pt.append(n)
            pointset.append(pt)

        '''primitives=[]
        for i in range(primitives_count):
            line = freader.readline()
            numbers = line.strip().split(' ')
            pr=[]
            for j in range(len(numbers)):
                pr.append(float(numbers[j]))
            primitives.append(pr)'''

    return np.array(pointset)


# data augmentation

def centralize_data(data):
    if len(data.shape) == 3:
        data = data[0]

    offset = (np.max(data[:, :3], axis=0) + np.min(data[:, :3], axis=0)) / 2
    data[:, :3] -= offset[None, :]
    ratio = abs(data[:, :3]).max()
    data[:, :3] /= ratio
    
    # data[:, [1, 2]] = data[:, [2, 1]]
    # data[:, 1] = data[:, 1] - data[:, 1].min() - 1

    return data, offset, ratio

def sphere_visualize(sphere_center, sphere_radius, pc):
    # visualize
    # generate sphere
    u = np.random.rand(200,1) * 2 - 1
    v = np.random.rand(200,1) * 2 - 1
    w = np.random.rand(200,1) * 2 - 1

    norm = (u*u + v*v + w*w)**(0.5)

    xi,yi,zi = u/norm,v/norm,w/norm
    sphere = np.concatenate((xi, yi, zi), axis=1) * sphere_radius + sphere_center[None, :]
    sphere_color = np.ones_like(sphere) * 255
    sphere_color[:, 1:] = 0
    sphere = np.concatenate((sphere, sphere_color), axis=1)

    pc = np.concatenate((pc, sphere), axis=0)

    return pc, sphere

def kmeans(raw_points, sample_center, max_iter=1, show=False):
    
    centers = sample_center.copy()
    iter = 0
    
    sampled_points = raw_points.copy()
    while iter < max_iter:
        if show:
            print("Kmeans iter: %d"%iter)
        dis = cdist(sampled_points[:, :3], centers[:, :3])
        index = np.argmin(dis, axis=-1)
        
        kmeans_dict = {}
        for i in range(index.shape[0]):
            if index[i] not in kmeans_dict:
                kmeans_dict[index[i]] = []
            kmeans_dict[index[i]].append(sampled_points[i])
        
        new_centers = []
        for i in kmeans_dict:
            kmeans_dict[i] = np.array(kmeans_dict[i])
            mean = np.mean(kmeans_dict[i], axis=0)
            new_centers.append(mean)
            
        centers = np.array(new_centers)
        iter += 1
    
    return kmeans_dict, centers


def get_kmeans_clusters(raw_points, sample_num, max_iter=1):
    sample_center = raw_points[np.random.choice(raw_points.shape[0], sample_num, replace=False)]
    kmeans_dict, centers = kmeans(raw_points, sample_center, max_iter)
    return kmeans_dict

    
def visualize_kmeans_dict(kmeans_dict, fn="kmeans.ply", ratio=1, color_dict=None):
    
    if color_dict is None:
        existing_color = dict()
    else:
        existing_color = color_dict
    visual = []
    
    for i, key in enumerate(kmeans_dict):
        if key not in existing_color:
            existing_color[key] = np.random.randint(192, size=3) + 64
        color = existing_color[key]
        
        for point in kmeans_dict[key]:
            visual.append([point[0], point[1], point[2], color[0], color[1], color[2]])
            
    visual = np.array(visual)
    visual[:, :3] *= ratio
    
    save_ply_with_color(fn, visual)
    
    return existing_color
    
    
def visualize_MST_list(data_list, fn="kmeans.ply", threshold=100, color_list=None):
    existing_color = dict()
    visual = []
    curr_color_list = []
    
    for i, data in enumerate(data_list):
        if np.array(data).shape[0] < threshold:
            continue
        if color_list is None:
            color = np.random.randint(128, size=3) + 128
            curr_color_list.append(color)
        else:
            color = color_list[i]
            
        for point in data:
            visual.append([point[0], point[1], point[2], color[0], color[1], color[2]])
    
    save_ply_with_color(fn, np.array(visual))

    return curr_color_list


def reconstruction_project_ply_data(pts, type="Branch", color=None, image_path=None):
    if image_path is None:
        image_path = 'tmp_for_reconstruction.png'
    pc = pts / 0.075
    x = pc[:, 0]
    y = pc[:, 1]
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    if color is None:
        color = tuple(np.array([0, 0, 255])/255.)
    plt.plot(x, y, marker='o', linestyle="", markerfacecolor=color)
    plt.savefig(image_path, dpi='figure')
    plt.close()
    image = np.asarray(Image.open(image_path))
    if type == "Junction":
        image = image - (image==255) * 64
    
    return image


def reconstruction_load_treepart_ply_as_image(filepath, type):
    data = np.loadtxt(filepath)
    pc = data[:, :3]
    radius_noise_ratio = data[:, 3]
    radius_weighted_noise = np.random.rand(*(pc.shape)) / ((1-radius_noise_ratio) * 25 + 50)[:, None]
    pc += radius_weighted_noise
    return reconstruction_project_ply_data(pc, type)



def load_treepart_ply_as_image(filepath, add_noise=True):
    data = np.loadtxt(filepath)
    pc = data[:, :3]
    if add_noise:
        radius_noise_ratio = data[:, 3]
        radius_weighted_noise = np.random.rand(*(pc.shape)) / ((1-radius_noise_ratio) * 25 + 50)[:, None]
        pc += radius_weighted_noise
    return project_ply_data(pc)


def project_ply_data(pts):
    fig = plt.figure(figsize=(3.2, 2.4))
    pc = pts / 0.05
    x = pc[:, 0]
    y = pc[:, 1]
    plt.xlim(-1, 1)
    plt.ylim(-1, 1)
    plt.axis('off')
    color = tuple(np.array([79, 110, 196])/255.)
    plt.plot(x, y, marker='o', linestyle="", markerfacecolor=color)
    random_num = np.random.rand()
    plt.savefig('tmp_for_reconstruction_%.2f.png'%(random_num), dpi=20)
    plt.close()
    return np.asarray(Image.open('tmp_for_reconstruction_%.2f.png'%(random_num)))

    
def draw_tsne(embeddings, keys, eval_filepath, data_projector_folder, current_epoch, task, length, position, padding_num, cls):

    os.makedirs('./tsne_plot/%s'%(task), exist_ok=True)
    canvas_size = 3200
    canvas = np.zeros((canvas_size, canvas_size, 3))
    embedding_list = []
    padding_color_list = []
    cls_color_list = []
    pos_color_list = []
    tsne_list = []
    image_list = []
    visualize_image = []
    category = []
    
    embedding = embeddings
    
    for filepath in eval_filepath:
        image_list.append(load_treepart_ply_as_image(filepath)[:, :, :3])
    key = keys
    
    data_length = embedding.shape[0]
    # color = np.random.randint(128, size=3) + 127
    padding_color = []
    pos_color = []
    cls_color = []
    
    for i in range(embedding.shape[0]):
        #by default
        # color.append(np.random.randint(128, size=3) + 127)
        # by length
        # color.append(np.array([255 * length[i]/500, 0, 0]).astype(int))
        # by position
        pos_color.append(np.array(255*(position[i]+1)/2).astype(int))
        # by padding_num
        padding_color.append(np.array([255*(padding_num[i]/500), 0, 255]).astype(int))
        cls_color.append(np.array([(1-cls[i]) * 255, cls[i] * 255, 0]).astype(int))
        
    padding_color = np.array(padding_color)
    pos_color = np.array(pos_color)
    cls_color = np.array(cls_color)
    # if key != -1:
    #     image = np.clip((image < 1).astype(float) + color[:, :, None, None]/255, 0, 1)

    embedding_list.append(embedding)
    padding_color_list.append(padding_color)
    pos_color_list.append(pos_color)
    cls_color_list.append(cls_color)
    
    tsne = TSNE(n_components=2, init='random').fit_transform(np.concatenate(embedding_list, axis=0))
    # tsne_3d = TSNE(n_components=3, init='random').fit_transform(np.concatenate(embedding_list, axis=0))
    
    # visualize tsne 2d images
    plt.scatter(tsne[:, 0].tolist(), tsne[:, 1].tolist(), c=(np.concatenate(padding_color_list, axis=0)/255).tolist())    
    plt.savefig('./tsne_plot/%s/%s_tsne_%d_padding.jpg'%(task, task, current_epoch))
    plt.close('all')
    visualize_image.append(np.asarray(Image.open('./tsne_plot/%s/%s_tsne_%d_padding.jpg'%(task, task, current_epoch))))
    category.append("padding")
    
    plt.scatter(tsne[:, 0].tolist(), tsne[:, 1].tolist(), c=(np.concatenate(pos_color_list, axis=0)/255).tolist())    
    plt.savefig('./tsne_plot/%s/%s_tsne_%d_pos.jpg'%(task, task, current_epoch))
    plt.close('all')
    visualize_image.append(np.asarray(Image.open('./tsne_plot/%s/%s_tsne_%d_pos.jpg'%(task, task, current_epoch))))
    category.append("position")
    
    plt.scatter(tsne[:, 0].tolist(), tsne[:, 1].tolist(), c=(np.concatenate(cls_color_list, axis=0)/255).tolist())    
    plt.savefig('./tsne_plot/%s/%s_tsne_%d_cls.jpg'%(task, task, current_epoch))
    plt.close('all')
    visualize_image.append(np.asarray(Image.open('./tsne_plot/%s/%s_tsne_%d_cls.jpg'%(task, task, current_epoch))))
    category.append("class")
    
    ###### TESTING VISUALIZATION ######
    embedding_shape_num = tsne.shape[0]
    plt.scatter(tsne[:embedding_shape_num//2, 0].tolist(), tsne[:embedding_shape_num//2, 1].tolist(), c=(np.concatenate(cls_color_list, axis=0)/255)[:embedding_shape_num//2].tolist())    
    plt.savefig('./tsne_plot/%s/%s_tsne_%d_cls_half.jpg'%(task, task, current_epoch))
    plt.close('all')
    
    ratio = 0.8
    tsne_list = tsne
    tsne_list = (tsne_list / abs(tsne_list).max() * canvas_size // 2 * ratio + np.ones((embeddings.shape[0], 2)) * canvas_size // 2).astype(int)
    image_list = np.stack(image_list)
    _, h, w, _ = image_list.shape
    
    for i, pos in enumerate(tsne_list):
        center_w, center_h = pos
        canvas[center_h:center_h+h, center_w:center_w+w] = image_list[i, :, :, :3]
    
    im = Image.fromarray((canvas).astype(np.uint8))
    im.save("./tsne_plot/%s/%s_tsne_image_%d.jpg"%(task, task, current_epoch))
    print("TSNE Drawn")
    visualize_image.append((canvas).astype(np.uint8))
    category.append("tsne")
    
    return visualize_image, category
    
    
def draw_closest_figs(embeddings, embedding_gallery, filepath, filepath_gallery, data_projector_folder, current_epoch, task='Tree', add_noise=True):
    os.makedirs('./tsne_plot/%s'%(task), exist_ok=True)
    
    total_image_set = []
    for i in range(5):
        curr_embedding = embeddings[i]
        curr_image = load_treepart_ply_as_image(filepath[i], add_noise)[:, :, :3]
        # np.asarray(Image.open(os.path.join(data_projector_folder, filepath[i])))[:, :, :3]
        
        indices = np.argsort(np.sum((embedding_gallery - curr_embedding[None, :])**2, axis=1))[:10]
        closest_filepath = filepath_gallery[indices]
        closest_images = [load_treepart_ply_as_image(filepath, add_noise)[:, :, :3] for filepath in closest_filepath]
        image_set = np.concatenate([curr_image] + closest_images, axis=1)
        total_image_set.append(image_set)
        
    for i in range(-5, 0):
        curr_embedding = embeddings[i]
        curr_image = load_treepart_ply_as_image(filepath[i], add_noise)[:, :, :3]
        # np.asarray(Image.open(os.path.join(data_projector_folder, filepath[i])))[:, :, :3]
        
        indices = np.argsort(np.sum((embedding_gallery - curr_embedding[None, :])**2, axis=1))[:10]
        closest_filepath = filepath_gallery[indices]
        closest_images = [load_treepart_ply_as_image(filepath, add_noise)[:, :, :3] for filepath in closest_filepath]
        image_set = np.concatenate([curr_image] + closest_images, axis=1)
        total_image_set.append(image_set)
    
    total_image_set = np.concatenate(total_image_set, axis=0)
    im = Image.fromarray((total_image_set).astype(np.uint8))
    im.save('./tsne_plot/%s/cloeset_%d.jpg'%(task, current_epoch))
            
    print("Closest Fig Drawn")
    
    return (total_image_set).astype(np.uint8)


def denoise(points, threshold):
    distances = pdist(points[:, :3])
    
    dist_matrix = squareform(distances)
    dist_matrix += np.eye(dist_matrix.shape[0], dist_matrix.shape[1]) * 1e8
    min_dist_matrix = np.sort(dist_matrix, axis=1)

    new_points = points[np.where(min_dist_matrix[:, 5]< threshold)]
    
    return new_points  


def remove_close_points_to_junction(points, segmentation, threshold, multi_process=True):
    
    if multi_process:
        dist_matrix = pairwise_distances(points[:, :3])
    else:
        dist_matrix = np.zeros((points.shape[0], points.shape[0]))
        for i in range(points.shape[0]):
            for j in range(points.shape[0]):
                dist_matrix[i][j] = np.sqrt(np.sum((points[i] - points[i]) ** 2))
    junction_index = np.argwhere(segmentation==1).reshape(-1)
    
    junction_index = np.argwhere(np.sum(dist_matrix[junction_index] < threshold, axis=0) > 0).reshape(-1)
    filtered_branch_index = []
    filtered_junction_index = []
    
    for i in range(segmentation.shape[0]):
        # skil foliage
        # if points[i, -2] == -1:
        #     continue
        if i not in junction_index:
            filtered_branch_index.append(i)
        else:
            filtered_junction_index.append(i)
            
    filtered_branch_index = np.array(filtered_branch_index)
    filtered_junction_index = np.array(filtered_junction_index)
    
    return dist_matrix, filtered_junction_index, filtered_branch_index 


def adaptive_sampling(points, target_num=15000, shuffle=False, random=True):
    
    sampled_pcd = points
    if points.shape[0] > target_num:
        if random:
            if points.shape[0] > target_num:
                sample_idx = np.random.choice(points.shape[0], target_num, replace=False)
                sampled_pc = points[sample_idx]
                sampled_pcd = sampled_pc
            else:
                sample_idx = np.random.choice(points.shape[0], target_num-points.shape[0], replace=True)
                sampled_pc = points[sample_idx]
                sampled_pcd = np.concatenate([points, sampled_pc], axis=0)
        else:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            
            sampled_pcd = pcd.farthest_point_down_sample(target_num)
            sampled_pcd = np.asarray(sampled_pcd.points)
        
    if shuffle:
        # sampled_pcd = sampled_pcd[np.random.choice(sampled_pcd.shape[0], target_num, replace=False)]
        np.random.shuffle(sampled_pcd)
        
    return sampled_pcd


def normalize(v):
    norm = np.sqrt(np.sum(v ** 2, axis=1))[:, None] + 1e-4
    return v / norm

def add_noise_by_height(pc, ratio=0.02, noise_division=5):
    num = pc.shape[0]
    if num > 0:
        noise_candidate_idx = np.random.choice(num, int(num/noise_division))
        noise_base = pc[noise_candidate_idx].copy()
        
        # embed()
        root = np.mean(pc[np.argwhere(pc[:, 1] < (pc[:, 1].min() + 0.05 * (pc[:, 1].max() - pc[:, 1].min())))], axis=0).reshape(-1)[:3]
        root_points_dist = pairwise_distances(noise_base[:, :3], root[None, :])
        root_points_ratio = 2 / (root_points_dist / root_points_dist.max() + 2)
        sample_dir = root[None, :] - noise_base[:, :3]
        sample_dir[:, 1] *= 0
        sample_dir = normalize(sample_dir)
        # embed()
        # ratio = (1 - noise_base[:, 1] / noise_base[:, 1].max()) * ratio # * root_points_ratio.reshape(-1)
        ratio = root_points_ratio.reshape(-1) * ratio
        noise_base[:, :3] += ((np.random.rand(noise_base[:, :3].shape[0])) * ratio)[:, None] * sample_dir
        
        noisy_pc = np.concatenate([pc, noise_base], axis=0)

    return noisy_pc

# def adaptive_sampling(points, layer=10, target_num=15000):
#     eps = 1e-8
#     num = points.shape[0]
#     points_hierarchy = [[] for _ in range(layer)]
    
#     h_max, h_min = points[:, 1].max(), points[:, 1].min()
#     height = h_max - h_min
#     v = height / layer
#     sample_per_layer = target_num // layer

#     for point in points:
#         curr_h = point[1] - h_min - eps
#         points_hierarchy[int(curr_h / v)].append(point)
    
#     sampled_points = []
#     points_hierarchy_to_sample = []
#     remaining_points = 0
#     for points in points_hierarchy:
#         if len(points) < sample_per_layer:
#             sampled_points += points
#         else:
#             points_hierarchy_to_sample.append(points)
#             remaining_points += len(points)
            
#     sample_per_layer = target_num // layer
    
#     sampled_points_per_layer_index = np.random.choice(len(points), sample_per_layer, replace=sample_per_layer>len(points))
#     sampled_points.append(np.array(points)[sampled_points_per_layer_index])

#         print("Replace: ", sample_per_layer, " ", len(points))
        
#     return np.array(sampled_points)


    