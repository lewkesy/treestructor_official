import os
import sys

from utils.curve import get_curve, random_sample_circle, rotation_matrix_from_vectors
import numpy as np
from IPython import embed
from utils.visualize import save_ply, save_mesh, save_mesh_with_color, save_mesh_wit_uv
from tqdm import tqdm


def vector_normalize(v, axis=1):
    return v / np.sqrt(np.sum(v ** 2, axis=axis, keepdims=True))


def rodrigues_rotation(axis_vec, dir, theta):
    return dir*np.cos(theta) + np.cross(axis_vec, dir, axis=1)*np.sin(theta) + axis_vec*np.sum(axis_vec*dir, axis=1)*(1-np.cos(theta))


def gen_dir(d_root, root, dst):
    d = dst - root
    axis_vec = vector_normalize(np.cross(d, d_root), axis=1)
    theta = (np.random.rand(1)[0]*60-30) / 180 * np.pi
    
    direction = rodrigues_rotation(axis_vec, d_root, theta)

    return direction


def vector_transform(vec, rotation, transformation):
    if len(vec.shape) == 2:
        vec = vec[:, :, None]
    transformed_vec =  np.matmul(rotation, vec)[:, :, 0] + transformation[:, 0, :]

    return transformed_vec


def visualize_yshape_point_cloud(pos, dir, radius, name):

    root, left, right = pos
    d_root, d_left, d_right = vector_normalize(dir)
    r_root, r_left, r_right = radius

    left_points, left_dirs, left_radius = get_curve(root, left, d_root, d_left, r_root, r_left)
    right_points, right_dirs, right_radius = get_curve(root, right, d_root, d_right, r_root, r_right)

    left_sampled_points = random_sample_circle(left_points[1:], left_dirs[1:], left_radius[1:])
    right_sampled_points = random_sample_circle(right_points[1:], right_dirs[1:], right_radius[1:])

    points = np.concatenate((left_sampled_points, right_sampled_points), axis=0)

    save_ply('visualize/yshape/%s.ply' % name, points)


def gen_mesh(points, dirs, rs, start_idx, sample_num=20, foliage=False, foliage_start_idx=0, foliage_per_node=5, foliage_radius=0):
    
    total_points = []
    total_faces = []
    foliage_total_points = []
    foliage_total_faces = []
    
    if points.shape[0] == 0:
        return None, None
    
    # sample all nodes for each circle
    for point, dir, r in zip(points, dirs, rs):
        
        sampled_points = random_sample_circle(point, dir, r, sample_num)
        total_points.append(sampled_points)
    
    # find two layers for mesh generation
    for layer in range(len(total_points)-1):

        node = layer * sample_num + start_idx
        for i in range(sample_num-1):

            # find the start node for the mesh
            total_faces.append([3, node, node + sample_num + 1, node + sample_num])
            total_faces.append([3, node, node + 1, node + sample_num + 1])
            node += 1
        
        # find the last mesh for this layer
        total_faces.append([3, node, node + sample_num + 1 - sample_num, node + sample_num])
        total_faces.append([3, node, node + 1 - sample_num, node + sample_num + 1 - sample_num])
    
    total_points = np.concatenate(total_points, axis=0)
    
    if foliage:
        dist = np.sqrt(((points[1] - points[0])**2).sum())
        node = foliage_start_idx
        for point, dir, r in zip(points[1:], dirs, rs):
            for _ in range(foliage_per_node):
                quad = np.array([[-0.5,0,-0.5], 
                        [-0.5,0,0.5], 
                        [0.5,0,0.5], 
                        [0.5,0,-0.5]])
                
                quad_size = r / 2 if foliage_radius == 0 else foliage_radius/2.5
                ratio = foliage_radius / r
                
                position = (np.random.rand(3,)*2-1)*r*ratio + point
                lookat_position = point+dir*dist*(0.75+0*np.random.rand())
                plane_rotation_matrix = rotation_matrix_from_vectors(np.array([1,0,0]), position-point)
                
                rotation_matrix = rotation_matrix_from_vectors(np.array([0,1,0]), lookat_position-position)
                quad = (rotation_matrix@plane_rotation_matrix@quad.T).T * quad_size + position
                
                foliage_total_points.append(quad)
                foliage_total_faces.append([3, node, node+1, node+2])
                foliage_total_faces.append([3, node+2, node+3, node])
                node += 4
                
        foliage_total_points = np.concatenate(foliage_total_points, axis=0)
        
    return np.array(total_points), np.array(total_faces), np.array(foliage_total_points), np.array(foliage_total_faces)


def visualize_yshape_mesh(name, 
                        pos, 
                        dir, 
                        radius, 
                        raw_pc_normalize_offset, 
                        raw_pc_normalize_ratio,
                        pts, 
                        key, 
                        if_save_mesh=True,
                        color=None,
                        bound_check=True,
                        sample_num=10,
                        foliage=False,
                        foliage_radius=0,
                        foliage_per_node=3):
    
    point_list = []
    dir_list = []
    radius_list = []

    for i in range(1, pos.shape[0]):
        
        curr_points, curr_dirs, curr_radius = get_curve(pos[0], pos[i], dir[0], dir[i], radius[0], radius[i], scalar=20, pts=pts, key=key, bound_check=bound_check)

        if curr_points.shape[0] >= 2:
            curr_points = curr_points * raw_pc_normalize_ratio + raw_pc_normalize_offset
            point_list.append(curr_points) 
            dir_list.append(curr_dirs)
            radius_list.append(curr_radius * raw_pc_normalize_ratio)

        else:
            print(curr_points.shape)
            print(pos[0], pos[1])
            print(pts[0], pts[1])
            print("skip")
    
    total_mesh_points = []
    total_mesh_faces = []
    total_foliage_points = []
    total_foliage_faces = []
    start_idx = 0
    foliage_start_idx = 0
    
    for curve_idx in range(len(point_list)):
        mesh_points, mesh_faces, foliage_points, foliage_meshes = gen_mesh(point_list[curve_idx], 
                                                                           dir_list[curve_idx], 
                                                                           radius_list[curve_idx], 
                                                                           start_idx,
                                                                           sample_num, 
                                                                           foliage, 
                                                                           foliage_start_idx,
                                                                           foliage_per_node=foliage_per_node, 
                                                                           foliage_radius=foliage_radius)
        if mesh_points is not None:
            total_mesh_points.append(mesh_points)
            total_mesh_faces.append(mesh_faces)
            total_foliage_points.append(foliage_points)
            total_foliage_faces.append(foliage_meshes)
            
            start_idx += mesh_points.shape[0]
            foliage_start_idx += foliage_points.shape[0]


    total_mesh_points = np.array(total_mesh_points).reshape(-1, 3)
    total_mesh_faces = np.array(total_mesh_faces).reshape(-1, 4)
    if foliage:
        total_foliage_points = np.array(total_foliage_points).reshape(-1, 3)
        total_foliage_faces = np.array(total_foliage_faces).reshape(-1, 4)
        
        
    if if_save_mesh is True:
        if total_mesh_faces.shape[0] != 0:
            filepath_list = name.split('/')
            filepath_list[-1] = "foliage_"+filepath_list[-1]
            foliage_name = "/".join(filepath_list)
            
            if color is None:
                save_mesh(name, total_mesh_points, total_mesh_faces)
                if foliage:
                    save_mesh_wit_uv(foliage_name, total_foliage_points, total_foliage_faces, None)
            else:
                save_mesh_with_color(name, total_mesh_points, total_mesh_faces, [color for _ in range(total_mesh_points.shape[0])])
                if foliage:
                    # embed()
                    save_mesh_wit_uv(foliage_name, total_foliage_points, total_foliage_faces, [color for _ in range(total_foliage_points.shape[0])])
                
        else:
            print("No mesh generated")
            point_list = []
            radius_list = []

    return point_list, dir_list, radius_list, total_mesh_points, total_mesh_faces


def generate_cylinder_vis(treepart_info, pc_by_index, offset, ratio, filename, bound_check=True, if_save_mesh=False, sample_num=10, color=None,
                      foliage=False, 
                      foliage_radius=0,
                      foliage_per_node=3):
    
    features = treepart_info['feature']
    main_dir= treepart_info['main_dir']
    
    splines = []
    splines_radius = []
    splines_dirs = []
    spline_color = [0, 0, 0]
    
    total_mesh_points = []
    total_mesh_faces = []
    
    bound_pts = pc_by_index.copy()
    for i, key in enumerate(features):

        start_pos = features[key]['Start Position']
        start_dir = features[key]['Start Direction']
        start_radius = features[key]['Start Thickness']
            
        end_pos = features[key]['End Position']
        end_dir = features[key]['End Direction']
        end_radius= features[key]['End Thickness']
        
        point_list, dir_list, radius_list, curr_mesh_points, curr_mesh_faces = visualize_yshape_mesh(
                    "./visualize/"+filename+"_"+str(key)+"_"+str(i)+".ply",
                    np.stack([start_pos, end_pos]),
                    np.stack([start_dir, end_dir]),
                    np.array([start_radius, end_radius]),
                    offset,
                    ratio,
                    bound_pts,
                    key,
                    bound_check=bound_check,
                    if_save_mesh=if_save_mesh,
                    color=color,
                    foliage=foliage,
                    foliage_radius=foliage_radius,
                    foliage_per_node=foliage_per_node
                    )
                
        curr_mesh_faces[:, 1:] += len(total_mesh_points)
        total_mesh_faces += curr_mesh_faces.tolist()
        total_mesh_points += curr_mesh_points.tolist()
        
        for i in range(len(point_list)):
            if point_list[i].shape[0] > 2:
                splines.append([point_list[i][0], point_list[i][-1]])
                splines_dirs.append([dir_list[i][0], dir_list[i][-1]])
                splines_radius.append([radius_list[i][0], radius_list[i][-1]])
        
    # return splines, splines_dirs, splines_radius, spline_color
    # if len(total_mesh_points) == 0:
    #     print("No meshes")
    #     embed()

    if np.array(total_mesh_points).shape[0] > 0:
        save_mesh("./visualize/tmp/"+filename+".ply", np.array(total_mesh_points), np.array(total_mesh_faces))
     
    return splines, splines_dirs, splines_radius, spline_color


def generate_cylinder(treepart_info, 
                      pc_by_index, 
                      offset, 
                      ratio, 
                      filename, 
                      dir_name, 
                      spline_color_np, 
                      global_iter, 
                      save_mesh, 
                      foliage=False, 
                      foliage_radius=0,
                      foliage_per_node=3):
    
    features = treepart_info['feature']
    main_dir= treepart_info['main_dir']
    
    splines = []
    splines_radius = []
    splines_dirs = []
    spline_color = [0, 0, 0]
    
    # max_bound_pts = 0
    # total_bound_pts = 0
    # for internode_index in pc_by_index:
    #     curr_length = len(pc_by_index[internode_index])
    #     if max_bound_pts < curr_length:
    #         max_bound_pts = curr_length
    #     total_bound_pts += curr_length

    bound_pts = pc_by_index.copy()
    for i, key in enumerate(features):
        
        start_pos = features[key]['Start Position']
        start_dir = features[key]['Start Direction']
        start_radius = features[key]['Start Thickness']
            
        end_pos = features[key]['End Position']
        end_dir = features[key]['End Direction']
        end_radius= features[key]['End Thickness']
        
        point_list, dir_list, radius_list, _, _ = visualize_yshape_mesh(
                    'results/%s/treepart/'%dir_name+filename+"_"+str(key)+"_"+str(i)+"_"+str(global_iter)+".ply",
                    np.stack([start_pos, end_pos]),
                    np.stack([start_dir, end_dir]),
                    np.array([start_radius, end_radius]),
                    offset,
                    ratio,
                    bound_pts,
                    key,
                    save_mesh,
                    color=spline_color_np,
                    foliage=foliage,
                    foliage_radius=foliage_radius,
                    foliage_per_node=foliage_per_node)
                
            
        for i in range(len(point_list)):
            if point_list[i].shape[0] > 2:
                splines.append([point_list[i][0], point_list[i][-1]])
                splines_dirs.append([dir_list[i][0], dir_list[i][-1]])
                splines_radius.append([radius_list[i][0], radius_list[i][-1]])

            
    # spline_color = [int(c) for c in (main_dir+1)/2 * 255.]
    # if spline_color[0] > spline_color[1]:
    #     embed()
    if spline_color_np is None:
        spline_color = None
    else:
        spline_color = [int(spline_color_np[color_idx]) for color_idx in range(spline_color_np.shape[0])]
        
    return splines, splines_dirs, splines_radius, spline_color


if __name__ == "__main__":

    pos = np.array([[0, -1, 0] , [1, 0, 0], [-1, 0, 0]]).astype(np.float)
    dir = np.array([[0, 1, 0], [1, 0, 0], [-1, 0, 0]]).astype(np.float)
    r = np.array([0.2, 0.2, 0.2])
    visualize_yshape_mesh(pos, dir, r, "test.ply")