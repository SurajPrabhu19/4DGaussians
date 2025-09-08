#!/usr/bin/env python3
# point_cloud_viewer.py

# Visualizes the 4D Gaussian splat point cloud and camera poses in a COLMAP-like static view
# Keyboard controls:
# 1. Left/Right arrow keys: Previous/Next time step
# 2. ESC/Q: Quit the viewer
# Mouse controls:
# 1. Drag: Rotate the view
# 2. Scroll: Zoom in/out
# 3. Shift + Drag: Pan the view

# python point_cloud_viewer.py --model_path "output/dnerf/bouncingballs/" --configs arguments/dnerf/bouncingballs.py --iteration 14000

import argparse
import os
import numpy as np
import torch
import open3d as o3d
from scene import Scene
from gaussian_renderer import GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from utils.general_utils import safe_state
from utils.graphics_utils import getWorld2View2
from time import time

class PointCloudViewer:
    def __init__(self):
        # Parse arguments
        parser = argparse.ArgumentParser(description="4D Gaussian Splatting Point Cloud Viewer")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        hyperparam = ModelHiddenParams(parser)
        parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load")
        parser.add_argument("--configs", type=str, help="Path to config file")
        args = get_combined_args(parser)
        
        self.model_path = args.model_path
        self.iteration = args.iteration
        self.configs = args.configs if hasattr(args, 'configs') else None
        print(f"Loading model from {self.model_path}, iteration {self.iteration}")
        
        if self.configs:
            try:
                exec(open(self.configs).read(), globals())
            except Exception as e:
                print(f"Warning: Failed to load config file {self.configs}: {e}")
        
        # Override ModelHiddenParams to match checkpoint
        args.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64, 64, 64, 75]
        }
        args.multires = [1, 2]
        args.net_width = 64
        args.defor_depth = 0
        
        with torch.no_grad():
            try:
                self.gaussians = GaussianModel(model.extract(args).sh_degree, hyperparam.extract(args))
                self.scene = Scene(model.extract(args), self.gaussians, load_iteration=self.iteration, shuffle=False)
                self.cameras = self.scene.getVideoCameras()
                if len(self.cameras) == 0:
                    raise ValueError("No cameras found. Check if model was trained correctly.")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        self.time_idx = 0
        self.max_time_idx = len(self.cameras) - 1
        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window(window_name="4D Gaussian Splatting Point Cloud Viewer", width=800, height=800)
        self.geometry_added = False
        
        # Instructions
        self.instructions = [
            "Keys:",
            "  Left/Right: Previous/Next Time Step",
            "  ESC/Q: Quit",
            "Mouse:",
            "  Drag: Rotate",
            "  Scroll: Zoom",
            "  Shift + Drag: Pan"
        ]
        
        # Register key callbacks
        self.vis.register_key_callback(262, lambda vis: self.next_time_step(vis))  # Right arrow
        self.vis.register_key_callback(263, lambda vis: self.prev_time_step(vis))  # Left arrow
        self.vis.register_key_callback(81, lambda vis: self.quit(vis))  # Q
        self.vis.register_key_callback(256, lambda vis: self.quit(vis))  # ESC

    def get_point_cloud(self, time_idx):
        """Extract point cloud at a specific time step"""
        try:
            with torch.no_grad():
                # Get Gaussian positions and colors
                points = self.gaussians._xyz  # Shape: (N, 3)
                colors = self.gaussians._features_dc.squeeze(-1)  # Shape: (N, 3)
                
                # Apply deformation for the given time step
                if hasattr(self.gaussians, 'deformation') and self.gaussians.deformation is not None:
                    time = torch.tensor([time_idx / self.max_time_idx], dtype=torch.float32, device="cuda")
                    points = points + self.gaussians.deformation(points, time)
                
                # Convert to numpy
                points = points.detach().cpu().numpy()
                colors = colors.detach().cpu().numpy()
                colors = np.clip(colors, 0, 1)  # Ensure colors in [0, 1]
                
                # Create Open3D point cloud
                pcd = o3d.geometry.PointCloud()
                pcd.points = o3d.utility.Vector3dVector(points)
                pcd.colors = o3d.utility.Vector3dVector(colors)
                return pcd
        except Exception as e:
            print(f"Error getting point cloud for time_idx {time_idx}: {e}")
            return None

    def get_camera_frustums(self, time_idx):
        """Create camera frustum for the current time step"""
        try:
            view = self.cameras[time_idx]
            c2w = torch.inverse(view.world_view_transform).detach().cpu().numpy()
            fov_x = view.FoVx if hasattr(view, 'FoVx') else np.pi / 4
            fov_y = view.FoVy if hasattr(view, 'FoVy') else np.pi / 4
            frustum = o3d.geometry.LineSet.create_camera_visualization(
                width=800, height=800, focal_length=1.0 / np.tan(fov_x / 2),
                extrinsic=np.linalg.inv(c2w)  # Open3D expects world-to-camera
            )
            return [frustum]
        except Exception as e:
            print(f"Error creating camera frustum for time_idx {time_idx}: {e}")
            return []

    def update_visualization(self):
        """Update the visualization with point cloud and camera at current time_idx"""
        self.vis.clear_geometries()
        pcd = self.get_point_cloud(self.time_idx)
        frustums = self.get_camera_frustums(self.time_idx)
        if pcd:
            self.vis.add_geometry(pcd)
        for frustum in frustums:
            self.vis.add_geometry(frustum)
        
        # Add text (simulated with print for simplicity)
        print(f"Time Step: {self.time_idx}/{self.max_time_idx}")
        for line in self.instructions:
            print(line)
        
        self.geometry_added = True

    def next_time_step(self, vis):
        self.time_idx = min(self.time_idx + 1, self.max_time_idx)
        self.update_visualization()
        vis.update_renderer()

    def prev_time_step(self, vis):
        self.time_idx = max(self.time_idx - 1, 0)
        self.update_visualization()
        vis.update_renderer()

    def quit(self, vis):
        vis.destroy_window()

    def run(self):
        print("Starting point cloud viewer. Press 'Q' or 'ESC' to exit.")
        self.update_visualization()
        
        # Set up view control
        view_ctl = self.vis.get_view_control()
        view_ctl.set_constant_z_near(0.1)
        view_ctl.set_constant_z_far(100.0)
        
        # Run visualization
        self.vis.run()
        self.vis.destroy_window()

if __name__ == "__main__":
    try:
        viewer = PointCloudViewer()
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()