"""
create a key for each body to be rendered

"""

import numpy as np 
from smpl_np import SMPLModel
import open3d as o3d
import pygame
from pathlib import Path
import pickle as pkl

clock = pygame.time.Clock()

# with open("/Users/paan/mnt/fig2/T7_2TB/CHI23/IMUPoser/scripts/1.2 LSTM predict subsets/end2end_results.pkl", "rb") as f:
#     data = pkl.load(f)

# with open("/Users/paan/mnt/fig2/T7_2TB/CHI23/IMUPoser/scripts/1.1 Baselines with AMASS and averaging/end2end_results_global.pkl", "rb") as f:
#     data = pkl.load(f)

# with open("/Users/paan/mnt/fig1/CHI23/IMUPoser/results/LSTMwjointloss_lw_lp_h/results.pkl", "rb") as f:
#     data = pkl.load(f)

with open("/Users/paan/mnt/fig1/CHI23/IMUPoser/ProcessMocap/video/imu_data/TCS_S2_2/results_rw_lp_rp.pkl", "rb") as f:
    data = pkl.load(f)

# with open("/Users/paan/mnt/fig1/CHI23/IMUPoser/ProcessMocap/video/imu_data/TCS_S2_2/results_rw_rp.pkl", "rb") as f:
#     r_data = pkl.load(f)

with open("/Users/paan/mnt/fig1/CHI23/IMUPoser/ProcessMocap/video/imu_data/TCS_S2_2/results_rw.pkl", "rb") as f:
    r_data = pkl.load(f)

data["p"] = r_data["t"]

# data["t"] = data["p"]

# with open("/Users/paan/mnt/edusense/CHI23/IMUPoser/scripts/3. Transformer/end2end_results.pkl", "rb") as f:
#     data = pkl.load(f)

# with open("./results.pkl", "rb") as f:
#     data = pkl.load(f)

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
models = {key: SMPLModel("./model.pkl") for key in ["p", "t"]}

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
    print(i)
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
    clock.tick(60)
