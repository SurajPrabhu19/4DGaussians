#!/usr/bin/env python3
import os
import subprocess
import argparse
import numpy as np
import json
from tqdm import tqdm

def run_colmap_sfm(image_dir, output_dir):
    """
    Run COLMAP Structure-from-Motion on image sequence to get camera parameters
    and sparse point cloud
    
    Args:
        image_dir: Directory containing input frames
        output_dir: Directory to save COLMAP output
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create COLMAP database
    db_path = os.path.join(output_dir, "database.db")
    
    print("Running feature extraction...")
    subprocess.check_call([
        "colmap", "feature_extractor",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--ImageReader.camera_model", "SIMPLE_PINHOLE"
    ])
    
    print("Running feature matching...")
    subprocess.check_call([
        "colmap", "sequential_matcher",
        "--database_path", db_path
    ])
    
    # Create sparse directory
    sparse_dir = os.path.join(output_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    
    print("Running SfM reconstruction...")
    subprocess.check_call([
        "colmap", "mapper",
        "--database_path", db_path,
        "--image_path", image_dir,
        "--output_path", sparse_dir
    ])
    
    # Convert to format compatible with 4D Gaussian Splatting
    print("Converting to readable format...")
    subprocess.check_call([
        "colmap", "model_converter",
        "--input_path", os.path.join(sparse_dir, "0"),
        "--output_path", os.path.join(output_dir, "sparse"),
        "--output_type", "TXT"
    ])
    
    # Export point cloud
    ply_path = os.path.join(output_dir, "points3D.ply")
    subprocess.check_call([
        "colmap", "model_converter",
        "--input_path", os.path.join(sparse_dir, "0"),
        "--output_path", ply_path,
        "--output_type", "PLY"
    ])
    
    return os.path.join(output_dir, "sparse"), ply_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run camera tracking and SfM")
    parser.add_argument("--images", type=str, required=True, help="Input image directory")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    
    args = parser.parse_args()
    run_colmap_sfm(args.images, args.output)