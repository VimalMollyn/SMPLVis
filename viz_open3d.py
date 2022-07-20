"""
create a key for each body to be rendered

"""

import numpy as np 
from smpl_np import SMPLModel
import open3d as o3d
import pygame
from pathlib import Path

clock = pygame.time.Clock()

# load data 
data = {
    "t": np.load("./Y_test_last.npy").reshape(-1, 72),
    "p": np.load("./Y_pred_last.npy").reshape(-1, 72)
}

save = False

if save:
    path_to_save = Path("frames")
    path_to_save.mkdir(exist_ok=True)

    for key in data.keys():
        (path_to_save / key).mkdir(exist_ok=True)

data_len = data[list(data.keys())[0]].shape[0]

## set up colors for the mesh 
mesh_base_c = np.array([192, 192, 192]) / 255 ## Some type of gray
mesh_error_c = np.array([255, 0, 0]) / 255

trans = {
    "t": np.array([-1, 0, 0]),
    "p": np.array([1, 0, 0])
}

# load the smpl model and process the points
models = {key: SMPLModel("./model.pkl") for key in data.keys()}

# create visualizer
viewer = o3d.visualization.Visualizer()
viewer.create_window(width=1080, height=1080, visible=True)

opt = viewer.get_render_option()
opt.background_color = np.array([1,1,1])

meshes = {}

# create the mesh and add the mesh to the scene
for key in models.keys():
    meshes[key] = o3d.geometry.TriangleMesh()
    meshes[key].triangles = o3d.utility.Vector3iVector(models[key].faces)
    meshes[key].vertices = o3d.utility.Vector3dVector(models[key].verts)
    meshes[key].compute_vertex_normals()
    viewer.add_geometry(meshes[key])

# every timestep, update the pose and obtain the new vertices
for i in range(data_len):
    # set the pose params
    for key in models.keys():
        models[key].set_params(pose = data[key][i].reshape(-1, 3))

    # calculate mesh error
    mesh_error = np.linalg.norm(models["t"].verts - models["p"].verts, axis=1)
    t = np.stack([mesh_error, mesh_error, mesh_error], axis=1)

    colors = t * (mesh_error_c - mesh_base_c) + mesh_base_c

    meshes["p"].vertex_colors = o3d.utility.Vector3dVector(colors)

    for key in models.keys():
        meshes[key].triangles = o3d.utility.Vector3iVector(models[key].faces)
        meshes[key].vertices = o3d.utility.Vector3dVector(models[key].verts + trans[key])
        meshes[key].compute_vertex_normals()
        viewer.update_geometry(meshes[key])

    viewer.poll_events()
    viewer.update_renderer()

    if save:
        viewer.capture_screen_image(str(path_to_save / key / f"{i}.png"), do_render=False)

    # time.sleep(25/60) # TODO update this
    clock.tick(100)
