import numpy as np
from plyfile import PlyData, PlyElement

def sphere_visualize(sphere_center, sphere_radius, pc, filename):
    # visualize
    # generate sphere
    sphere_point_number = 1000
    u = np.random.rand(sphere_point_number,1) * 2 - 1
    v = np.random.rand(sphere_point_number,1) * 2 - 1
    w = np.random.rand(sphere_point_number,1) * 2 - 1

    norm = (u*u + v*v + w*w)**(0.5)

    xi,yi,zi = u/norm,v/norm,w/norm
    sphere = np.concatenate((xi, yi, zi), axis=1) * sphere_radius + sphere_center[None, :]

    sphere_color = np.ones_like(sphere) * 255
    sphere_color[:, 1:] = 0

    sphere = np.concatenate((sphere, sphere_color), axis=1)

    if pc.shape[1] == 3:
        color = np.array([[0, 0, 255] for _ in range(pc.shape[0])])
        pc = np.concatenate([pc, color], axis=1)
        
    pc = np.concatenate((pc, sphere), axis=0)
    save_ply_with_color(filename, pc)

    return 


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

def save_segmentation(fn, xyz, is_forks):

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
            if is_forks[i] == 1:
                f.write('%.6f %.6f %.6f %d %d %d\n' % (xyz[i][0], xyz[i][1], xyz[i][2], 255, 0, 0))
            else:
                f.write('%.6f %.6f %.6f %d %d %d\n' % (xyz[i][0], xyz[i][1], xyz[i][2], 0, 0, 255))


def save_ply(fn, xyz, color=None):

    with open(fn, 'w') as f:
        pn = xyz.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (pn))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('end_header\n')
        for i in range(pn):
            if color is None:
                f.write('%.6f %.6f %.6f\n' % (xyz[i][0], xyz[i][1], xyz[i][2]))

def save_mesh_wit_uv(fn, xyz, face, color):

    vertices_num = xyz.shape[0]
    uv_coord = {
        0: [0, 0],
        1: [0, 1],
        2: [1, 1],
        3: [1, 0]
    }
    
    if color is not None:
        vertex = np.empty(vertices_num, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1'),
            ('s', 'f4'), ('t', 'f4')
        ])

        # uv_coord[i%4][0], uv_coord[i%4][1]
        for i in range(vertices_num):
            vertex[i] = (
                xyz[i][0], xyz[i][1], xyz[i][2],
                color[i][0], color[i][1], color[i][2],
                uv_coord[i%4][0], uv_coord[i%4][1]
            )
    else:
        vertex = np.empty(vertices_num, dtype=[
            ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('s', 'f4'), ('t', 'f4')
        ])
        
        for i in range(vertices_num):
            vertex[i] = (
                xyz[i][0], xyz[i][1], xyz[i][2],
                uv_coord[i%4][0], uv_coord[i%4][1]
            )

    vertex_element = PlyElement.describe(vertex, 'vertex')
    num_faces = face.shape[0]
    face_with_indices = np.empty(num_faces, dtype=[('vertex_indices', 'i4', (3,))])
    for i in range(num_faces):
        face_with_indices[i] = (face[i][1:],)
        
    face_element = PlyElement.describe(face_with_indices, 'face')
    new_ply_data = PlyData([vertex_element, face_element], text=True)
    new_ply_data.write(fn)
    

def save_mesh(fn, xyz, faces):
    with open(fn, 'w') as f:
        point_num = xyz.shape[0]
        face_num = faces.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (point_num))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('element face %d\n' % (face_num))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        for i in range(point_num):
            f.write('%.6f %.6f %.6f\n' % (xyz[i][0], xyz[i][1], xyz[i][2]))
        for i in range(face_num):
            f.write('%d %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2] ,faces[i][3]))


def save_mesh_with_color(fn, xyz, faces, color):
    with open(fn, 'w') as f:
        point_num = xyz.shape[0]
        face_num = faces.shape[0]
        f.write('ply\n')
        f.write('format ascii 1.0\n')
        f.write('element vertex %d\n' % (point_num))
        f.write('property float x\n')
        f.write('property float y\n')
        f.write('property float z\n')
        f.write('property uchar red\n')
        f.write('property uchar green\n')
        f.write('property uchar blue\n')
        f.write('element face %d\n' % (face_num))
        f.write('property list uchar int vertex_index\n')
        f.write('end_header\n')
        for i in range(point_num):
            f.write('%.6f %.6f %.6f %d %d %d\n' % (xyz[i][0], xyz[i][1], xyz[i][2], int(color[i][0]), int(color[i][1]), int(color[i][2])))
        for i in range(face_num):
            f.write('%d %d %d %d\n' % (faces[i][0], faces[i][1], faces[i][2] ,faces[i][3]))