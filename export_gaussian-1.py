import os
import torch
import numpy as np
from scene import GaussianModel
from arguments import ModelParams, PipelineParams
import plyfile

def export_gaussian_model(model_path, iteration=20000):
    # Load trained model
    dataset = ModelParams().parse_args(['-s', 'data/dnerf/bouncingballs', '--model_path', model_path])
    pipeline = PipelineParams().parse_args([])
    gaussian_model = GaussianModel(dataset.sh_degree)
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    gaussian_model.load_ply(ply_path)

    # Load deformation data (time-varying)
    deformation_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/deformation.pth")
    deformation_table = torch.load(deformation_path, map_location='cpu')
    # Assuming deformation_table contains time-varying offsets; adjust based on your model's structure
    deformation = deformation_table.get('deformation', torch.zeros_like(gaussian_model.get_xyz)).cpu().numpy()

    # Extract Gaussian attributes
    xyz = gaussian_model.get_xyz.cpu().numpy()  # 3D positions
    colors = gaussian_model.get_features.cpu().numpy()[:, :3]  # RGB or first 3 SH coefficients
    opacity = gaussian_model.get_opacity.cpu().numpy()  # Alpha values
    scale = gaussian_model.get_scaling.cpu().numpy()  # Scale per axis
    rotation = gaussian_model.get_rotation.cpu().numpy()  # Quaternion rotations

    # Combine with deformation (time-varying offsets)
    # Adjust this based on how 4DGaussians stores time data; here we assume per-point offsets
    vertex_data = np.zeros(len(xyz), dtype=[
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),  # Position
        ('r', 'f4'), ('g', 'f4'), ('b', 'f4'),  # Color
        ('opacity', 'f4'),  # Opacity
        ('scale_x', 'f4'), ('scale_y', 'f4'), ('scale_z', 'f4'),  # Scale
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),  # Quaternion
        ('deform_x', 'f4'), ('deform_y', 'f4'), ('deform_z', 'f4')  # Deformation offsets
    ])
    vertex_data['x'], vertex_data['y'], vertex_data['z'] = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    vertex_data['r'], vertex_data['g'], vertex_data['b'] = colors[:, 0], colors[:, 1], colors[:, 2]
    vertex_data['opacity'] = opacity
    vertex_data['scale_x'], vertex_data['scale_y'], vertex_data['scale_z'] = scale[:, 0], scale[:, 1], scale[:, 2]
    vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3'] = rotation[:, 0], rotation[:, 1], rotation[:, 2], rotation[:, 3]
    vertex_data['deform_x'], vertex_data['deform_y'], vertex_data['deform_z'] = deformation[:, 0], deformation[:, 1], deformation[:, 2]

    # Save to PLY
    output_ply = os.path.join(model_path, 'exported_gaussian.ply')
    ply_element = plyfile.PlyElement.describe(vertex_data, 'vertex')
    plyfile.PlyData([ply_element]).write(output_ply)
    print(f"Exported PLY to {output_ply}")

if __name__ == "__main__":
    model_path = "output/dnerf/bouncingballs"  # Your experiment path
    export_gaussian_model(model_path, iteration=30000)
