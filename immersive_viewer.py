#!/usr/bin/env python3
# immersive_viewer.py

# This viewer provides an immersive visualization of a 4D Gaussian splat with:
# Keyboard controls:
# 1. Left/Right arrow keys (or A/D): Navigate between frames (back/forward in time)
# 2. W/S: Move camera forward/backward
# 3. A/D: Strafe camera left/right (also used for frame navigation)
# 4. Up/Down: Move camera up/down
# 5. +/-: Zoom in/out (adjust field of view)
# 6. P: Toggle play/pause
# 7. R: Reset view
# 8. ESC/Q: Quit
# Mouse controls:
# 1. Drag (left-click): Rotate camera (yaw and pitch)
# 2. Scroll: Zoom in/out

# python immersive_viewer.py --model_path "output/dnerf/bouncingballs/" --configs arguments/dnerf/bouncingballs.py --iteration 14000

import argparse
import os
import cv2
import numpy as np
import torch
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args
from utils.general_utils import safe_state
from time import time
from utils.graphics_utils import getWorld2View2

class ImmersiveViewer:
    def __init__(self):
        # Parse arguments
        parser = argparse.ArgumentParser(description="Immersive 4D Gaussian Splatting Viewer")
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
                self.cam_type = self.scene.dataset_type
                bg_color = [1,1,1] if model.extract(args).white_background else [0, 0, 0]
                self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                self.cameras = self.scene.getVideoCameras()
                self.pipeline = pipeline.extract(args)
                if len(self.cameras) == 0:
                    raise ValueError("No cameras found. Check if model was trained correctly.")
            except Exception as e:
                print(f"Error loading model: {e}")
                raise
        
        # Viewer state
        self.time_idx = 0
        self.max_time_idx = len(self.cameras) - 1
        self.scale = 1.0
        self.window_name = "4D Gaussian Splatting Immersive Viewer"
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        self.rotation = np.zeros(3, dtype=np.float32)  # Euler angles (rx, ry, rz)
        self.translation = np.zeros(3, dtype=np.float32)  # Camera position
        self.paused = True  # Start paused
        self.playback_speed = 5.0  # Frames per second
        self.last_frame_time = time()
        self.move_speed = 0.1  # Camera movement speed
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Instructions
        self.instructions = [
            "Keys:",
            "  Left/Right: Previous/Next Frame",
            "  W/S: Move Forward/Backward",
            "  A/D: Strafe Left/Right",
            "  Up/Down: Move Up/Down",
            "  +/-: Zoom in/out",
            "  P: Toggle Play/Pause",
            "  R: Reset View",
            "  ESC/Q: Quit",
            "Mouse:",
            "  Drag: Rotate View",
            "  Scroll: Zoom"
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_x = x
            self.last_y = y
            print("Started dragging")
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
            print("Stopped dragging")
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_x
            dy = y - self.last_y
            self.last_x = x
            self.last_y = y
            self.rotation[1] += dx * 0.01  # Yaw
            self.rotation[0] += dy * 0.01  # Pitch
            print(f"Rotation: {self.rotation}")
        elif event == cv2.EVENT_MOUSEWHEEL:
            wheel_delta = flags >> 16
            if wheel_delta > 0:
                self.scale *= 1.1
                print(f"Zoom in: {self.scale:.2f}x")
            else:
                self.scale /= 1.1
                print(f"Zoom out: {self.scale:.2f}x")

    def get_euler_rotation_matrix(self, euler_angles):
        rx, ry, rz = euler_angles
        Rx = np.array([[1, 0, 0], [0, np.cos(rx), -np.sin(rx)], [0, np.sin(rx), np.cos(rx)]])
        Ry = np.array([[np.cos(ry), 0, np.sin(ry)], [0, 1, 0], [-np.sin(ry), 0, np.cos(ry)]])
        Rz = np.array([[np.cos(rz), -np.sin(rz), 0], [np.sin(rz), np.cos(rz), 0], [0, 0, 1]])
        R = Rz @ Ry @ Rx
        return torch.tensor(R, dtype=torch.float32, device="cuda")

    def apply_view_transforms(self, view):
        try:
            view.FoVx /= self.scale
            view.FoVy /= self.scale
            world_view_transform = view.world_view_transform
            c2w = torch.inverse(world_view_transform)
            R = c2w[:3, :3].T
            t = c2w[:3, 3]
            user_rot = self.get_euler_rotation_matrix(self.rotation)
            R_new = user_rot @ R
            # Apply translation
            t_new = t + torch.tensor(self.translation, dtype=torch.float32, device="cuda")
            view.world_view_transform = torch.from_numpy(getWorld2View2(R_new.numpy(), t_new.numpy())).float().cuda()
            return view
        except Exception as e:
            print(f"Error in apply_view_transforms: {e}")
            return view

    def render_current_frame(self):
        if 0 <= self.time_idx < len(self.cameras):
            try:
                view = self.cameras[self.time_idx]
                view = self.apply_view_transforms(view)
                start_time = time()
                rendering = render(view, self.gaussians, self.pipeline, self.background, cam_type=self.cam_type)["render"]
                render_time = time() - start_time
                image = rendering.detach().cpu().numpy()
                image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)
                image = np.ascontiguousarray(image)
                if image.shape[2] == 3:
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.putText(image, f"Frame: {self.time_idx}/{self.max_time_idx}", (10, 30), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Render Time: {render_time:.3f}s", (10, 60), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Zoom: {self.scale:.2f}x", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Position: {self.translation}", (10, 120), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"{'Paused' if self.paused else 'Playing'}", (10, 150), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                for i, line in enumerate(self.instructions):
                    cv2.putText(image, line, (image.shape[1] - 300, 30 + i*25), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                return image
            except Exception as e:
                print(f"Error rendering frame {self.time_idx}: {e}")
                return None
        return None

    def run(self):
        print("Starting immersive viewer. Press 'Q' or 'ESC' to exit, 'P' to toggle play/pause.")
        while True:
            try:
                # Automatic playback
                if not self.paused:
                    current_time = time()
                    if current_time - self.last_frame_time >= 1.0 / self.playback_speed:
                        self.time_idx = (self.time_idx + 1) % (self.max_time_idx + 1)
                        self.last_frame_time = current_time
                        print(f"Frame: {self.time_idx}/{self.max_time_idx}")
                
                # Render current frame
                image = self.render_current_frame()
                if image is not None:
                    cv2.imshow(self.window_name, image)
                
                # Process keyboard input
                key = cv2.waitKey(10) & 0xFF
                if key != 255:  # Only print valid key presses
                    print(f"Key pressed: {key}")
                
                if key == 27 or key == ord('q'):  # ESC or Q
                    print("Exiting viewer")
                    break
                elif key == ord('p'):  # Toggle play/pause
                    self.paused = not self.paused
                    print(f"{'Paused' if self.paused else 'Playing'}")
                elif key in [83, 100, ord('d')]:  # Right arrow, numpad right, or 'd'
                    self.time_idx = min(self.time_idx + 1, self.max_time_idx)
                    print(f"Frame: {self.time_idx}/{self.max_time_idx}")
                elif key in [81, 97, ord('a')]:  # Left arrow, numpad left, or 'a'
                    self.time_idx = max(self.time_idx - 1, 0)
                    print(f"Frame: {self.time_idx}/{self.max_time_idx}")
                elif key == ord('w'):  # Move forward
                    forward = torch.tensor([0, 0, -1], dtype=torch.float32, device="cuda") @ self.get_euler_rotation_matrix(self.rotation)
                    self.translation += forward.cpu().numpy() * self.move_speed
                    print(f"Position: {self.translation}")
                elif key == ord('s'):  # Move backward
                    backward = torch.tensor([0, 0, 1], dtype=torch.float32, device="cuda") @ self.get_euler_rotation_matrix(self.rotation)
                    self.translation += backward.cpu().numpy() * self.move_speed
                    print(f"Position: {self.translation}")
                elif key in [82, ord('e')]:  # Up arrow or 'e'
                    self.translation += np.array([0, self.move_speed, 0], dtype=np.float32)
                    print(f"Position: {self.translation}")
                elif key in [84, ord('c')]:  # Down arrow or 'c'
                    self.translation -= np.array([0, self.move_speed, 0], dtype=np.float32)
                    print(f"Position: {self.translation}")
                elif key == ord('+') or key == ord('='):  # Zoom in
                    self.scale *= 1.1
                    print(f"Zoom: {self.scale:.2f}x")
                elif key == ord('-') or key == ord('_'):  # Zoom out
                    self.scale /= 1.1
                    print(f"Zoom: {self.scale:.2f}x")
                elif key == ord('r'):  # Reset view
                    self.scale = 1.0
                    self.rotation = np.zeros(3, dtype=np.float32)
                    self.translation = np.zeros(3, dtype=np.float32)
                    print("View reset")
            except Exception as e:
                print(f"Error in main loop: {e}")
                continue
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        viewer = ImmersiveViewer()
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
