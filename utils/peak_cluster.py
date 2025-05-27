import numpy as np
from sklearn.metrics import pairwise_distances
import matplotlib.pyplot as plt
from IPython import embed
# from utils.utils import visualize_kmeans_dict, save_ply_with_color, sphere_visualize
from time import time
import copy
from tqdm import tqdm
from numba import njit, prange
import sys
    

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


def save_ply_with_color(fn, xyz):

    with open(fn, 'w') as f:
        pn = xyz.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (pn))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('end_header\n')
        for i in range(pn):
            f.write('%.6f %.6f %.6f %d %d %d\n' % (xyz[i][0], xyz[i][1], xyz[i][2], xyz[i][3], xyz[i][4], xyz[i][5]))
       
       
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
     
            
class DensityPeakClusterSrc():
    
    # @staticmethod    
    # def find_distance(a, b):
    #     dist = np.zeros((a.shape[0], b.shape[0]))
    #     threshold = 200
    #     iter_num = int(np.ceil(a.shape[0] / threshold))
    #     for i in tqdm(range(iter_num)):
    #         dist[i*threshold:(i+1)*threshold] = pairwise_distances(a[i*threshold:(i+1)*threshold], b, metric='l1', n_jobs=1)
        
    #     return dist
    
    @staticmethod    
    @njit(parallel=True)
    def find_distance(a, b):
        # Initialize the result distance matrix
        dist = np.zeros((a.shape[0], b.shape[0]))
        
        for i in prange(a.shape[0]):
            for j in range(b.shape[0]):
                dist[i, j] = np.sqrt(np.sum((a[i] - b[j]) ** 2))
        
        return dist
    
    # @staticmethod
    # def find_validation(weighted_density):
    #     validation = np.zeros((weighted_density.shape[0], weighted_density.shape[0]))
    #     # (weighted_density[:, None] < weighted_density[None, :]).astype(int)
        
    #     for i in tqdm(range(weighted_density.shape[0])):
    #         validation[i] = (weighted_density < weighted_density[i]).astype(int)
        
    #     return validation
    
    @staticmethod  
    @njit(parallel=True)
    def find_validation(weighted_density):
        result = np.zeros((weighted_density.shape[0], weighted_density.shape[0]), dtype=np.int8)
        
        for i in prange(weighted_density.shape[0]):
            for j in range(weighted_density.shape[0]):
                result[i, j] = weighted_density[i] < weighted_density[j]
                
        return result

    @staticmethod  
    @njit(parallel=True)
    def regularize_weight(dist, root_points_ratio, threshold):
        # weight = 1- (dist / (self.threshold * self.root_points_ratio))
        # 1- (dist / (self.threshold * self.root_points_ratio))
        res = np.zeros((dist.shape[0], dist.shape[1]))
        for i in prange(res.shape[0]):
            curr_weight = 1 - dist[i] / (root_points_ratio[i] * threshold)
            # res[i] = (curr_weight > 0) * curr_weight
            res[i] = np.where(curr_weight > 0, curr_weight, 0)
        return res
    
    @staticmethod  
    @njit(parallel=True)
    def find_min_dist_matrix(dist, valid_connection):
        res = np.zeros((dist.shape[0], dist.shape[0]))
        # min_dist_matrix = dist * valid_connection + (1-valid_connection) * 1e5
        for i in prange(dist.shape[0]):
            res[i] = dist[i] * valid_connection[i] + (1-valid_connection[i])*1e5
        
        return res
    
    
    def __call__(self, points, 
                 threshold, 
                 height,
                 dist_cutoff_ratio=0.035, 
                 density_cutoff_ratio=0.035, 
                 noise_cutoff_ratio=0.1, 
                 max_dist_cutoff_ratio=0.03, 
                 junction=True
                 ):
        
        self.points = np.array(points).astype(np.float32)
        self.threshold = threshold
        self.dist_cutoff_ratio = dist_cutoff_ratio
        self.density_cutoff_ratio = density_cutoff_ratio
        self.points_num = self.points.shape[0]
        self.noise_cutoff_ratio = noise_cutoff_ratio
        self.max_dist_cutoff_ratio = max_dist_cutoff_ratio
        self.height = height
        self.junction=junction

        self.root_points_ratio = np.ones((self.points.shape[0]))
        sys.setrecursionlimit(50000)
        
        # calculate ratio from pts to root point
        root = np.array([0, self.points[:, 1].min(), 0])
        
        # dist from point to root
        root_points_dist = pairwise_distances(self.points[:, :3], root[None, :], n_jobs=1).reshape(-1)
        # ratio [0.5, 1], from far to close to root
        
        self.root_points_ratio = (2/(2 + np.sqrt(np.sqrt(root_points_dist/self.height)))).reshape(-1)

        
        # define two params
        self.density, self.min_dist, self.closest_neighbour_idx, self.max_density_index, self.dist = self.data_process()
        # print("peak clustering data process finish")

        self.min_dist_cutoff = self.threshold

        # find cluster center
        cluster_center_idx = self.find_cluster_center()
        # print("Center Found: %d centers"%(cluster_center_idx.shape[0]))
        
        cluster_idx_dict = self.clustering(cluster_center_idx)
        res = dict()
        
        for key in cluster_idx_dict.keys():
            res[key] = self.points[np.array(cluster_idx_dict[key])]
        
        # print("cluster num:", len(list(res.keys())))
        # self.visualize_density_dist()
        # self.visualize_density_pointcloud()
        # self.visualize_sphere_pointcloud(self.points[0, :3], self.threshold, res)
        # self.visualize_dist_hist()
        # visualize_kmeans_dict(res, cluster_file)
        
        return res

    
    def dist_augment(self, dist):
        return 1/(1 + np.exp(-((dist / self.ori_dist_max) - 0.5) * 2)) 
    
    
    def data_process(self):
        
        dist = self.find_distance(self.points[:, :3], self.points[:, :3])
        print("Completed Graph Created")
        
        # weights = np.maximum(np.zeros_like(dist), 1- (dist / (self.threshold * self.root_points_ratio))) ** 2
        weights = self.regularize_weight(dist, self.root_points_ratio, self.threshold)
        print("Point Density Generated")
        # weights = np.maximum(np.zeros_like(dist), 1- (dist / (self.threshold * self.root_points_ratio)))

        # calculate density
        # pts_inside_sphere_range = dist < self.threshold
        # weighted_density = np.sum(pts_inside_sphere_range * weights, axis=1) # / np.sum(pts_inside_sphere_range, axis=1)
        
        weighted_density = np.sum(weights, axis=1, dtype=np.float32) # / np.sum(pts_inside_sphere_range, axis=1)
        max_density_indexes = np.argmax(weighted_density).reshape(-1)
        
        # find the valid higher density point by density
        # valid_connection = (weighted_density[:, None] < weighted_density[None, :]).astype(int)
        valid_connection = self.find_validation(weighted_density)
        print("Valid Connection Generated")
        
        # embed()
        # for max_density_index in max_density_indexes:
        #     valid_connection[max_density_index][max_density_index] = 1
        
        # calculate min dist with only higher density points
        # augment dist
        self.ori_dist_max = dist.max()
        # dist =  self.dist_augment(dist)
        min_dist_matrix = self.find_min_dist_matrix(dist, valid_connection)
        # min_dist_matrix = dist * valid_connection + (1-valid_connection) * 1e5
        min_dist = np.min(min_dist_matrix, axis=1)

        closest_neighbour_idx = np.argmin(min_dist_matrix, axis=1).reshape(-1)
        if min_dist.max() > 1e7:
            print("Huge number error occur!")
            embed()
        for max_density_index in max_density_indexes:
            min_dist[max_density_index] = min_dist.max() * 1.2
        
        del dist
        
        return weighted_density, min_dist, closest_neighbour_idx, max_density_index
        
    
    def find_cluster_center(self):
        # min_distance_cutoff = self.min_dist.min() + (self.min_dist.max()-self.min_dist.min()) * self.dist_cutoff_ratio
        
        min_distance_cutoff = np.ones(self.min_dist.shape[0]) * self.min_dist_cutoff * self.root_points_ratio
        distance_idx_candidate = np.argwhere(self.min_dist > min_distance_cutoff).reshape(-1)
        
        
        return distance_idx_candidate
        # density_cutoff = self.density.min() + (self.density.max()-self.density.min()) * self.density_cutoff_ratio
        # density_idx_candidate = np.argwhere(self.density > density_cutoff).reshape(-1)
        # intersect = np.intersect1d(distance_idx_candidate, density_idx_candidate)

        # if intersect.shape[0] == 0:
        #     return distance_idx_candidate
        # return intersect

    
    def find_cluster_center_by_hist(self):
        a, b = np.histogram(self.min_dist[np.where(self.density > max(self.density)*0.5)], 20)
    

    def clustering(self, cluster_center_idx):
        lookup = [-1 for _ in range(self.closest_neighbour_idx.shape[0])]
        cluster_dict = {}
        
        # embed()
        # register center points
        for i, idx in enumerate(cluster_center_idx):
            
            # the first node is always the center point
            cluster_dict[idx] = [idx]
            lookup[idx] = idx
        
        for i in range(self.closest_neighbour_idx.shape[0]):
            if lookup[i] != -1:
                continue
            self.find_cluster(lookup, cluster_dict, i)
            
        return cluster_dict
    
    
    def find_cluster(self, lookup, cluster_dict, idx):
        if lookup[idx] != -1:
            return lookup[idx]
        else:
            cluster_num = self.find_cluster(lookup, cluster_dict, self.closest_neighbour_idx[idx])
            lookup[idx] = cluster_num
            cluster_dict[cluster_num].append(idx)
            return cluster_num
    
    
    def find_merging_points(self, merge_table, idx):
        res = []
        if sum(merge_table[idx]) == 0:
            return res
        
        if merge_table[idx][idx] != 0:
            res.append(idx)
            
        for i in range(idx+1, merge_table.shape[0]):
            if merge_table[idx][i] != 0:
                res += self.find_merging_points(merge_table, i)
        
        merge_table[idx] = merge_table[idx] * 0
        return res 

    @staticmethod
    def prim_mst(adj_matrix):
        
        num_nodes = adj_matrix.shape[0]
        mst = np.zeros((num_nodes, num_nodes))  # List to store the edges of the MST
        selected_nodes = set()
        selected_nodes.add(0)  # Start with the first node

        count = 0
        while len(selected_nodes) < num_nodes:
            min_edge_weight = 1e8
            min_edge = None
            count += 1
            for node in selected_nodes:
                for neighbor in range(num_nodes):
                    if neighbor not in selected_nodes and adj_matrix[node][neighbor] > 0:
                        if adj_matrix[node][neighbor] < min_edge_weight:
                            min_edge_weight = adj_matrix[node][neighbor]
                            min_edge = (node, neighbor)

            if min_edge is not None:
                mst[min_edge[0], min_edge[1]] = min_edge_weight
                mst[min_edge[1], min_edge[0]] = min_edge_weight
                selected_nodes.add(min_edge[1])

        return mst

    @staticmethod
    def dijkstra(adj_matrix, start_node, end_node):
        # Number of nodes in the graph
        augmented_adj_matrix = (adj_matrix*100) ** 2
        num_nodes = adj_matrix.shape[0]

        # Create lists to track distance and visited nodes
        augmented_distance = [sys.maxsize] * num_nodes
        distance = [sys.maxsize] * num_nodes
        visited = [False] * num_nodes

        # Set the distance of the start_node to itself as 0
        distance[start_node] = 0
        augmented_distance[start_node] = 0

        for _ in range(num_nodes):
            # Find the node with the minimum distance from the set of unvisited nodes
            min_distance = sys.maxsize
            min_node = None
            for node in range(num_nodes):
                if not visited[node] and augmented_distance[node] < min_distance:
                    min_distance = augmented_distance[node]
                    min_node = node

            # Mark the selected node as visited
            visited[min_node] = True

            # Update distances of adjacent nodes
            for node in range(num_nodes):
                if not visited[node] and augmented_adj_matrix[min_node][node] > 0:
                    augmented_new_distance = augmented_distance[min_node] + augmented_adj_matrix[min_node][node]
                    new_distance = distance[min_node] + adj_matrix[min_node][node]
                    if augmented_new_distance < augmented_distance[node]:
                        augmented_distance[node] = augmented_new_distance
                        distance[node] = new_distance

        # Return the shortest distance between the start_node and end_node
        return distance[end_node]


    def visualize_dist_hist(self):
        plt.hist(self.min_dist, bins=100)
        plt.savefig("dist_hist.png")
        plt.close()
           
    def visualize_density_dist(self):
        
        def dist_curve(x):
            normalized_x = x - x.min()
            normalized_x -= normalized_x.max() * self.density_cutoff_ratio
            normalized_x /= normalized_x.max()

            return np.maximum((self.max_dist_cutoff_ratio * self.min_dist.max()) * normalized_x** 3 + self.min_dist_cutoff , self.min_dist_cutoff) 
        
        min_distance_cutoff = self.min_dist.min() + (self.min_dist.max()-self.min_dist.min()) * self.dist_cutoff_ratio
        density_cutoff = self.density.min() + (self.density.max()-self.density.min()) * self.density_cutoff_ratio
        
        plt.scatter(self.density.tolist(), self.min_dist.tolist())
        plt.plot([density_cutoff, density_cutoff], [self.min_dist.min(), self.min_dist.max()])
        # plt.plot([0, density_cutoff.max()], [min_distance_cutoff, min_distance_cutoff])
        
        plt.plot(np.sort(self.density), dist_curve(np.sort(self.density)))
        plt.savefig("vis.jpg")
        plt.close()
        
        
    def visualize_density_pointcloud(self):
        color_prior = self.density / self.density.max()
        vis_res = []
        for i in range(self.points.shape[0]):
            vis_res.append([self.points[i][0], self.points[i][1], self.points[i][2], int(255*color_prior[i]), 0, 0])
        
        save_ply_with_color("weight_density.ply", np.array(vis_res))
    
    
    def visualize_sphere_pointcloud(self, center, radius, ori_data_dict):
        u = np.random.rand(2000,1) * 2 - 1
        v = np.random.rand(2000,1) * 2 - 1
        w = np.random.rand(2000,1) * 2 - 1

        norm = (u*u + v*v + w*w)**(0.5)

        xi,yi,zi = u/norm,v/norm,w/norm
        sphere = np.concatenate((xi, yi, zi), axis=1) * radius + center[None, :]
        data_dict = copy.deepcopy(ori_data_dict)
        visualize_kmeans_dict(data_dict, "peak_cluster.ply")
        data_dict[-1] = sphere
        visualize_kmeans_dict(data_dict, "peak_cluster_sphere.ply")
        
        
class DensityPeakCluster(DensityPeakClusterSrc):
    def __call__(self, points, 
                    threshold, 
                    height,
                    dist_cutoff_ratio=0.035, 
                    density_cutoff_ratio=0.035, 
                    noise_cutoff_ratio=0.1, 
                    max_dist_cutoff_ratio=0.03, 
                    junction=True
                    ):
            
            self.points = np.array(points).astype(np.float32)
            self.threshold = threshold
            self.dist_cutoff_ratio = dist_cutoff_ratio
            self.density_cutoff_ratio = density_cutoff_ratio
            self.points_num = self.points.shape[0]
            self.noise_cutoff_ratio = noise_cutoff_ratio
            self.max_dist_cutoff_ratio = max_dist_cutoff_ratio
            self.height = self.points[:, 1].max() - self.points[:, 1].min()
            self.junction=junction

            self.root_points_ratio = np.ones((self.points.shape[0]))
            sys.setrecursionlimit(50000)
            
            # calculate ratio from pts to root point
            root = np.array([0, self.points[:, 1].min(), 0])
            
            # dist from point to root
            root_points_dist = pairwise_distances(self.points[:, :3], root[None, :], n_jobs=1).reshape(-1)
            # ratio [0.5, 1], from far to close to root
            # embed()
            
            # self.root_points_ratio = (3/(3 + 2*(root_points_dist/self.height)**2)).reshape(-1)
            self.root_points_ratio = (2/(2 + (root_points_dist/self.height)**2)).reshape(-1)
            
            del root_points_dist
            
            # define two params
            self.density, self.min_dist, self.closest_neighbour_idx, self.max_density_index = self.data_process()
            # print("peak clustering data process finish")

            #old min dist cutoff
            # self.min_dist_cutoff = min(self.min_dist) + (max(self.min_dist) - min(self.min_dist)) * self.dist_cutoff_ratio # * self.root_points_ratio
            
            self.min_dist_cutoff = self.threshold
            # print("Data processed")
            
            # find cluster center
            cluster_center_idx = self.find_cluster_center()
            # print("Center Found: %d centers"%(cluster_center_idx.shape[0]))
            
            # print("Start clusteirng")
            cluster_idx_dict = self.clustering(cluster_center_idx)
            res = dict()
            
            for key in cluster_idx_dict.keys():
                res[key] = self.points[np.array(cluster_idx_dict[key])]
            
            # print("cluster num:", len(list(res.keys())))
            self.visualize_density_dist()
            self.visualize_density_pointcloud()
            self.visualize_sphere_pointcloud(self.points[0, :3], self.threshold, res)
            # self.visualize_dist_hist()
            # visualize_kmeans_dict(res, cluster_file)
            
            return res
        
        
if __name__ == "__main__":
    from plyfile import PlyData, PlyElement
    def find_root_radius(raw_pc, ratio=0.02):
        
        potential_center_points = []
        
        pc = raw_pc.copy()
        y_len = pc[:, 1].max() - pc[:, 1].min()
        potential_root_points_idx = np.where(pc[:, 1] < (pc[:, 1].min() + y_len* ratio))
        root_points = pc[potential_root_points_idx]
        potential_root = np.mean(root_points, axis=0)
        potential_root[1] = pc[:, 1].min()

        x_radius = (root_points[:, 0].max() - root_points[:, 0].min()) / 2
        z_radius = (root_points[:, 2].max() - root_points[:, 2].min()) / 2
        
        return potential_root, max(x_radius, z_radius)
    
    
    data = PlyData.read('./data/Birch_2.ply')
    x = data["vertex"]["x"]
    y = data["vertex"]["y"]
    z = data["vertex"]["z"]
    
    pc = np.array([x, y, z]).T
    print(pc.shape)
    peak_cluster = DensityPeakCluster()
    
    _, root_radius = find_root_radius(pc)
    cluster_dict = peak_cluster(pc,
                                threshold=max(root_radius, 0.03), 
                                height=np.max(pc[:, 1])-np.min(pc[:, 1]),
                                dist_cutoff_ratio=0.035, # 0.05 sync # real 0.045, 0.0275 campus full
                                density_cutoff_ratio=0, 
                                noise_cutoff_ratio=0, 
                                max_dist_cutoff_ratio=0,
                                junction=True)
    
    