import os
import numpy as np
from scipy.spatial.distance import pdist
from scipy.spatial.distance import squareform
from IPython import embed
from plyfile import PlyData, PlyElement 
from utils.visualize import save_ply_with_color
from sklearn.metrics import pairwise_distances

class MST():
    def __init__(self, weights):
        self.edge_weight = []
        self.weights = weights
        self.parent = np.zeros(weights.shape[0], dtype=int) - 1
        self.mst = self.prim_mst(weights)
    
    def prim_mst(self, weights):
        n = weights.shape[0]
        key = np.full(n, np.inf)
        mst = []
        visited = np.zeros(n, dtype=bool)

        # Start from vertex 0
        key[0] = 0

        # Repeat until all vertices have been visited
        while not np.all(visited):
            # Find the vertex with the minimum key value among the unvisited vertices
            u = np.argmin(key)
            visited[u] = True

            # Add the corresponding edge to the MST
            if self.parent[u] >= 0:
                mst.append((self.parent[u], u, key[u]))
                self.edge_weight.append(key[u])

            # Update the key values of the neighbors of u
            for v in range(n):
                if not visited[v] and weights[u, v] < key[v]:
                    key[v] = weights[u, v]
                    self.parent[v] = u
            
            key[u] = np.inf
            
        return mst

    def partition(self, threshold):
        
        root_index = [0]
        for u in self.mst:
            parent, curr, dist = u
            if dist > threshold:
                root_index.append(curr)
                
        lookup_visited = np.zeros(self.weights.shape[0], dtype=bool)
        root_cluster = np.zeros(self.weights.shape[0], dtype=int)-1
        for i in range(self.weights.shape[0]):
            if lookup_visited[i]:
                continue
            else:
                self.seek_cluster(i, lookup_visited, root_cluster, root_index)
        
        return root_cluster
    
    
    def seek_cluster(self, i, lookup_visited, root_cluster, root_index):
        
        path_index = []
        curr_index = i
        
        lookup_visited[curr_index]= True
        while curr_index not in root_index:
            path_index.append(curr_index)
            lookup_visited[curr_index] = True
            curr_index = self.parent[curr_index]
        
        if len(path_index) != 0:
            root_cluster[np.array(path_index)] = curr_index
        else:
            root_cluster[curr_index] = curr_index
        
        
def split_pointclouds(points, threshold):
    
    clusters = []
    curr_dist_matrix = pairwise_distances(points[:, :3])

    curr_dist_matrix += np.eye(curr_dist_matrix.shape[0], curr_dist_matrix.shape[1]) * 1e8
    # print("Weight matrix calculated")
    
    mst = MST(curr_dist_matrix)
    partition = mst.partition(threshold)
    for key in np.unique(partition):
        pos = np.where(partition==key)
        clusters.append(points[pos])
    # print("Cluster num: %d"%len(clusters))
    return clusters
    
    
if __name__ == '__main__':
    
    # dist_matrix = np.array([[1e8, 5, 6, 9], [5, 1e8, 10, 8], [6, 10, 1e8, 7], [9, 8, 7, 1e8]])
    # mst = MST(dist_matrix)
    # p = mst.partition(5.5)
    data = PlyData.read("../data/branch.ply")

    x = data["vertex"]["x"]
    y = data["vertex"]["y"]
    z = data["vertex"]["z"]
    pc = np.stack([x, y, z]).T
    
    cluster = split_pointclouds(pc, 0.01)
    print("Cluster num: %d"%(len(cluster)))
    pts = []
    for cls in cluster:
        if len(cls) < 50:
            continue
        color = np.random.randint(128, size=3) + 127
        for p in cls:
            pts.append([p[0], p[1], p[2], color[0], color[1], color[2]])
        
    save_ply_with_color("../mst_branch.ply", np.array(pts))
