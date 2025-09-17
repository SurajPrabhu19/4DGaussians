#!/usr/bin/env python3
import os
import argparse
import torch
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args

def setup_4d_training(args):
    """
    Set up 4D Gaussian Splatting training from video frames
    """
    # Override ModelHiddenParams for 4D Gaussian Splatting
    args.kplanes_config = {
        'grid_dimensions': 2,
        'input_coordinate_dim': 4,  # x, y, z, time
        'output_coordinate_dim': 32,
        'resolution': [64, 64, 64, 75]  # Adjust based on video length
    }
    args.multires = [1, 2]
    args.net_width = 64
    args.defor_depth = 0
    
    # Set up training config
    args.iterations = 15000
    args.position_lr_init = 0.00016
    args.position_lr_final = 0.0000016
    args.position_lr_delay_mult = 0.01
    args.position_lr_max_steps = 30000
    args.feature_lr = 0.0025
    args.opacity_lr = 0.05
    args.scaling_lr = 0.005
    args.rotation_lr = 0.001
    args.percent_dense = 0.01
    args.densification_interval = 100
    args.opacity_reset_interval = 3000
    args.densify_from_iter = 500
    args.densify_until_iter = 15000
    args.random_background = False
    
    # Force temporal optimization
    args.time_smoothness_weight = 0.01
    args.loss_weight_entropy = 0.0
    
    return args

def train_4d_gaussians(scene_dir, expname, port=6009):
    """
    Train 4D Gaussian Splatting from prepared scene directory
    
    Args:
        scene_dir: Directory containing prepared scene (images, poses, point cloud)
        expname: Experiment name for output
        port: Port for visualization server
    """
    # Create argument parser
    parser = argparse.ArgumentParser(description="Train 4D Gaussian Splatting")
    model = ModelParams(parser)
    pipeline = PipelineParams(parser)
    hyperparam = ModelHiddenParams(parser)
    
    # Add custom arguments
    parser.add_argument("--expname", type=str, default=expname, help="Experiment name")
    parser.add_argument("--port", type=int, default=port, help="Port for visualization server")
    
    # Parse arguments
    args = get_combined_args(parser)
    args.source_path = scene_dir
    
    # Set up 4D training parameters
    args = setup_4d_training(args)
    
    # Start training (use existing train.py from 4DGaussians)
    print(f"Starting 4D Gaussian Splatting training for {expname}")
    print(f"Output will be saved to output/{expname}")
    
    # Import and call train function from 4DGaussians
    from train import train
    train(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 4D Gaussian Splatting from video")
    parser.add_argument("--scene_dir", type=str, required=True, help="Prepared scene directory")
    parser.add_argument("--expname", type=str, required=True, help="Experiment name")
    parser.add_argument("--port", type=int, default=6009, help="Port for visualization server")
    
    args = parser.parse_args()
    train_4d_gaussians(args.scene_dir, args.expname, args.port)