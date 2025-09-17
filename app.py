#!/usr/bin/env python3
from flask import Flask, render_template, request, jsonify, send_from_directory
import os
import json
import torch
import numpy as np
from scene import Scene
from gaussian_renderer import render, GaussianModel
from arguments import ModelParams, PipelineParams, ModelHiddenParams, get_combined_args

app = Flask(__name__)

# Model cache
loaded_models = {}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/models')
def list_models():
    output_dir = "output"
    models = []
    
    if os.path.exists(output_dir):
        for category in os.listdir(output_dir):
            category_path = os.path.join(output_dir, category)
            if os.path.isdir(category_path):
                for model_name in os.listdir(category_path):
                    model_path = os.path.join(category_path, model_name)
                    if os.path.isdir(model_path):
                        models.append({
                            "id": f"{category}/{model_name}",
                            "name": model_name,
                            "category": category,
                            "path": model_path
                        })
    
    return jsonify(models)

@app.route('/api/load_model', methods=['POST'])
def load_model():
    data = request.json
    model_id = data['modelId']
    iteration = data.get('iteration', -1)
    
    if model_id in loaded_models:
        return jsonify({
            "success": True,
            "frames": len(loaded_models[model_id]['cameras'])
        })
    
    try:
        model_path = os.path.join("output", model_id)
        config_path = os.path.join("arguments", f"{model_id}.py")
        
        # Create args
        parser = argparse.ArgumentParser()
        args = parser.parse_args([])
        args.model_path = model_path
        args.iteration = iteration
        args.configs = config_path if os.path.exists(config_path) else None
        
        # Set default params for 4D Gaussian model
        args.kplanes_config = {
            'grid_dimensions': 2,
            'input_coordinate_dim': 4,
            'output_coordinate_dim': 32,
            'resolution': [64, 64, 64, 75]
        }
        args.multires = [1, 2]
        args.net_width = 64
        args.defor_depth = 0
        
        # Extract model parameters
        model_params = ModelParams(None, sentinel=True)
        pipeline_params = PipelineParams(None)
        hyperparam = ModelHiddenParams(None)
        
        # Load model
        gaussians = GaussianModel(model_params.extract(args).sh_degree, hyperparam.extract(args))
        scene = Scene(model_params.extract(args), gaussians, load_iteration=iteration, shuffle=False)
        cameras = scene.getVideoCameras()
        pipeline = pipeline_params.extract(args)
        
        # Cache model
        loaded_models[model_id] = {
            'gaussians': gaussians,
            'scene': scene,
            'cameras': cameras,
            'pipeline': pipeline,
            'background': torch.tensor([0, 0, 0], dtype=torch.float32, device="cuda")
        }
        
        return jsonify({
            "success": True,
            "frames": len(cameras)
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        })

@app.route('/api/render_frame', methods=['POST'])
def render_frame():
    data = request.json
    model_id = data['modelId']
    frame_idx = data['frameIdx']
    camera_params = data.get('camera', {})
    
    if model_id not in loaded_models:
        return jsonify({
            "success": False,
            "error": "Model not loaded"
        })
    
    try:
        model_data = loaded_models[model_id]
        
        # Get camera for frame
        view = model_data['cameras'][frame_idx]
        
        # Apply custom camera parameters if provided
        if camera_params:
            # Transform camera based on rotation, position, etc.
            # Implementation depends on your camera model
            pass
        
        # Render frame
        result = render(view, model_data['gaussians'], 
                      model_data['pipeline'], 
                      model_data['background'])
        
        # Convert to image
        image = result["render"].detach().cpu().numpy()
        image = np.transpose(image, (1, 2, 0))
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        # Convert to base64 for sending to client
        import base64
        import cv2
        _, buffer = cv2.imencode('.png', image)
        img_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            "success": True,
            "image": f"data:image/png;base64,{img_base64}"
        })
    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({
            "success": False,
            "error": str(e)
        })

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description="4D Gaussian Web Viewer")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=5000, help="Port to listen on")
    parser.add_argument("--debug", action="store_true", help="Enable debug mode")
    
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=args.debug)