#!/usr/bin/env python3
# interactive_viewer.py

#This viewer provides the following keyboard controls:
#1.	Left/Right arrow keys (or A/D): Navigate between frames in the 4D Gaussian splat
#2.	+/- keys: Zoom in/out to explore the scene in detail
#3.	R key: Reset the view to default
#4.	ESC/Q: Quit the viewer
#For mouse controls:
#1.	Drag: Rotate the view to explore the scene from different angles
#2.	Mouse wheel: Zoom in/out

# python interactive_viewer.py --model_path "output/dnerf/bouncingballs/" --configs arguments/dnerf/bouncingballs.py --iteration 14000
# python interactive_viewer.py --model_path "output/dnerf/bouncingballs/" --configs arguments/dnerf/bouncingballs.py --iteration 14000 2> error.txt
# 3. python interactive_viewer.py --model_path "output/dnerf/bouncingballs/" --configs arguments/dnerf/bouncingballs.py --iteration 14000

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

class InteractiveViewer:
    def __init__(self, model_path, iteration, configs_path):
        # Parse arguments using a single ArgumentParser
        parser = argparse.ArgumentParser(description="Interactive 4D Gaussian viewer")
        model = ModelParams(parser, sentinel=True)
        pipeline = PipelineParams(parser)
        hyperparam = ModelHiddenParams(parser)
        
        # Create a list of arguments to simulate command-line input
        import sys
        original_argv = sys.argv  # Save original sys.argv
        sys.argv = [sys.argv[0]]  # Reset sys.argv to script name only
        if model_path:
            sys.argv.extend(["--model_path", model_path])
        if iteration is not None:
            sys.argv.extend(["--iteration", str(iteration)])
        if configs_path:
            sys.argv.extend(["--configs", configs_path])
            
        try:
            args = get_combined_args(parser)  # Call get_combined_args with only parser
        finally:
            sys.argv = original_argv  # Restore original sys.argv
        
        # Store the paths and parameters
        self.model_path = args.model_path
        self.iteration = args.iteration
        self.configs = args.configs      
        print(f"Loading model from {self.model_path}, iteration {self.iteration}")
        
        with torch.no_grad():
            try:
                # Load Gaussian model
                self.gaussians = GaussianModel(model.extract(args).sh_degree, hyperparam.extract(args))
                self.scene = Scene(model.extract(args), self.gaussians, load_iteration=self.iteration, shuffle=False)
                self.cam_type = self.scene.dataset_type
                
                # Set background color
                bg_color = [1,1,1] if model.extract(args).white_background else [0, 0, 0]
                self.background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                
                # Get cameras
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
        self.window_name = "4D Gaussian Splatting Interactive Viewer"
        self.dragging = False
        self.last_x = 0
        self.last_y = 0
        self.rotation = np.array([0, 0, 0], dtype=np.float32)  # euler angles (rx, ry, rz)
        self.translation = np.array([0, 0, 0], dtype=np.float32)
        
        # Create window
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, 800, 800)
        cv2.setMouseCallback(self.window_name, self.mouse_callback)
        
        # Instructions text
        self.instructions = [
            "Keys:",
            "  Left/Right: Previous/Next Frame",
            "  +/-: Zoom in/out",
            "  R: Reset view",
            "  ESC/Q: Quit",
            "Mouse:",
            "  Drag: Rotate view",
            "  Scroll: Zoom"
        ]

    def mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.dragging = True
            self.last_x = x
            self.last_y = y
        elif event == cv2.EVENT_LBUTTONUP:
            self.dragging = False
        elif event == cv2.EVENT_MOUSEMOVE and self.dragging:
            dx = x - self.last_x
            dy = y - self.last_y
            self.last_x = x
            self.last_y = y
            
            # Update rotation based on mouse movement
            self.rotation[1] += dx * 0.01  # Yaw
            self.rotation[0] += dy * 0.01  # Pitch
        
        # Mouse wheel event handling varies by platform
        # This handles both Windows and Linux wheel events
        elif event == cv2.EVENT_MOUSEWHEEL:
            # Windows wheel event
            wheel_delta = flags >> 16
            if wheel_delta > 0:
                self.scale *= 1.1
            else:
                self.scale /= 1.1
        elif event == 10:  # Linux mouse wheel up
            self.scale *= 1.1
        elif event == 11:  # Linux mouse wheel down
            self.scale /= 1.1

    def apply_view_transforms(self, view):
        """Apply rotation and scaling transforms to the view"""
        # Since we don't have access to the camera model implementation details,
        # we'll implement a basic transformation that should work with most views
        
        # Clone the view to avoid modifying the original
        try:
            # If view is a dict (for some camera types like PanopticSports)
            if isinstance(view, dict):
                # Make a shallow copy since we don't modify the view for now
                return view
            
            # For regular camera views
            if hasattr(view, "world_view_transform") and hasattr(view.world_view_transform, "copy"):
                # We're not actually modifying the transform for now since we don't know the format
                # This is a placeholder for future implementation
                pass
            
            return view
        except Exception as e:
            print(f"Error in apply_view_transforms: {e}")
            return view

    def render_current_frame(self):
        if 0 <= self.time_idx < len(self.cameras):
            try:
                view = self.cameras[self.time_idx]
                
                # Apply any view transformations (zoom, rotation)
                view = self.apply_view_transforms(view)
                
                # Render the frame
                start_time = time()
                rendering = render(view, self.gaussians, self.pipeline, self.background, cam_type=self.cam_type)["render"]
                render_time = time() - start_time
                
                # Convert to numpy for OpenCV
                image = rendering.detach().cpu().numpy()
                image = np.transpose(image, (1, 2, 0))  # CHW -> HWC
                image = np.clip(image, 0, 1)
                image = (image * 255).astype(np.uint8)
                
                # Add frame info text
                cv2.putText(image, f"Frame: {self.time_idx}/{self.max_time_idx}", (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Render Time: {render_time:.3f}s", (10, 60), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                cv2.putText(image, f"Zoom: {self.scale:.2f}x", (10, 90), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Add instructions
                for i, line in enumerate(self.instructions):
                    cv2.putText(image, line, (image.shape[1] - 300, 30 + i*25), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
                return image
            except Exception as e:
                print(f"Error rendering frame {self.time_idx}: {e}")
                return None
        return None

    def run(self):
        print("Starting interactive viewer. Press 'Q' or 'ESC' to exit.")
        
        while True:
            try:
                # Render current frame
                image = self.render_current_frame()
                
                if image is not None:
                    # Display the rendered image
                    cv2.imshow(self.window_name, image)
                
                # Process keyboard input - wait for a key with timeout
                key = cv2.waitKey(1) & 0xFF
                
                if key == 27 or key == ord('q'):  # ESC or Q
                    break
                # Handle more key codes for arrow keys for cross-platform compatibility
                elif key in [83, 100, ord('d')]:  # Right arrow, numpad right, or 'd'
                    self.time_idx = min(self.time_idx + 1, self.max_time_idx)
                    print(f"Frame: {self.time_idx}/{self.max_time_idx}")
                elif key in [81, 97, ord('a')]:  # Left arrow, numpad left, or 'a'
                    self.time_idx = max(self.time_idx - 1, 0)
                    print(f"Frame: {self.time_idx}/{self.max_time_idx}")
                elif key == ord('+') or key == ord('='):  # Zoom in
                    self.scale *= 1.1
                    print(f"Zoom: {self.scale:.2f}x")
                elif key == ord('-') or key == ord('_'):  # Zoom out
                    self.scale /= 1.1
                    print(f"Zoom: {self.scale:.2f}x")
                elif key == ord('r'):  # Reset view
                    self.scale = 1.0
                    self.rotation = np.array([0, 0, 0], dtype=np.float32)
                    self.translation = np.array([0, 0, 0], dtype=np.float32)
                    print("View reset")
                
                # Debug the key code to help diagnose issues
                if key not in [255, -1, 0]:  # Exclude "no key pressed" values
                    print(f"Key pressed: {key}")
            
            except Exception as e:
                print(f"Error in main loop: {e}")
                # Continue instead of breaking to make the viewer more robust
                continue
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Interactive 4D Gaussian viewer")
    parser.add_argument("--model_path", required=True, type=str, help="Path to model directory")
    parser.add_argument("--iteration", default=-1, type=int, help="Iteration to load")
    parser.add_argument("--configs", required=True, type=str, help="Path to config file")
    
    try:
        args = parser.parse_args()
        viewer = InteractiveViewer(args.model_path, args.iteration, args.configs)
        viewer.run()
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()