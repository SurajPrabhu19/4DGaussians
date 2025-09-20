#command used: python export_gaussian.py --model_path output/dnerf/bouncingballs --source_path data/dnerf/bouncingballs --sh_degree 3 --images images --white_background --resolution 512

import os
import torch
import numpy as np
from scene import GaussianModel
from arguments import ModelParams, PipelineParams, get_combined_args
import plyfile
import argparse

def export_gaussian_model(model_path, iteration=20000):
    # Ensure model_path is absolute
    model_path = os.path.abspath(model_path)
    
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Export Gaussian model to PLY")
    # Initialize parameter groups with the parser
    model_params = ModelParams(parser)
    pipeline_params = PipelineParams(parser)
    
    # Provide default arguments to avoid cfg_args dependency
    args_list = [
        '-s', 'data/dnerf/bouncingballs',
        '--model_path', model_path,
        '--sh_degree', '3',
        '--white_background', 'True',
        '--eval', 'True'
    ]
    
    # Debug cfg_args path
    cfg_path = os.path.join(model_path, "cfg_args")
    print(f"Attempting to load cfg_args from: {cfg_path}")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path) as cfg_file:
                cfg_content = cfg_file.read()
                print(f"cfg_args content: {cfg_content}")
        except Exception as e:
            print(f"Error reading cfg_args: {e}")
    
    # Try to merge with cfg_args if available
    try:
        args = get_combined_args(parser)
    except FileNotFoundError:
        print(f"Warning: cfg_args not found in {model_path}. Using default arguments.")
        args = parser.parse_args(args_list)
    except Exception as e:
        print(f"Error processing cfg_args: {e}. Using default arguments.")
        args = parser.parse_args(args_list)

    # Extract parameters
    dataset = model_params.extract(args)
    pipeline = pipeline_params.extract(args)
    gaussian_model = GaussianModel(dataset.sh_degree, args)  # Pass args to GaussianModel
    ply_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/point_cloud.ply")
    
    # Verify PLY file exists
    if not os.path.exists(ply_path):
        raise FileNotFoundError(f"PLY file not found at {ply_path}")
    
    gaussian_model.load_ply(ply_path)

    # Load deformation data (time-varying)
    deformation_path = os.path.join(model_path, f"point_cloud/iteration_{iteration}/deformation_table.pth")
    if not os.path.exists(deformation_path):
        print(f"Warning: Deformation file not found at {deformation_path}. Using zero deformations.")
        deformation = np.zeros_like(gaussian_model.get_xyz.detach().cpu().numpy())
    else:
        deformation_table = torch.load(deformation_path, map_location='cpu')
        # Adjust based on actual deformation structure; assuming 'deformation' key
        deformation = deformation_table.get('deformation', torch.zeros_like(gaussian_model.get_xyz)).detach().cpu().numpy()

    # Extract Gaussian attributes
    xyz = gaussian_model.get_xyz.detach().cpu().numpy()  # 3D positions
    colors = gaussian_model.get_features.detach().cpu().numpy()[:, :3]  # RGB or first 3 SH coefficients
    opacity = gaussian_model.get_opacity.detach().cpu().numpy()  # Alpha values
    scale = gaussian_model.get_scaling.detach().cpu().numpy()  # Scale per axis
    rotation = gaussian_model.get_rotation.detach().cpu().numpy()  # Quaternion rotations

    # Combine with deformation (time-varying offsets)
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
    export_gaussian_model(model_path, iteration=20000)

