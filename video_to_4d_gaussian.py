#!/usr/bin/env python3
import os
import argparse
import subprocess
import time
from datetime import datetime

def create_directory_structure(video_path):
    """Create directory structure for the project based on video name"""
    video_name = os.path.basename(video_path).split('.')[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    project_dir = f"projects/{video_name}_{timestamp}"
    
    # Create directories
    os.makedirs(f"{project_dir}/frames", exist_ok=True)
    os.makedirs(f"{project_dir}/colmap", exist_ok=True)
    os.makedirs(f"{project_dir}/audio", exist_ok=True)
    
    return project_dir, video_name

def extract_audio(video_path, output_dir):
    """Extract audio from video file"""
    audio_path = os.path.join(output_dir, "audio.wav")
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vn", "-acodec", "pcm_s16le",
        "-ar", "44100", "-ac", "2",
        audio_path
    ]
    subprocess.call(cmd)
    return audio_path

def process_video(video_path, output_dir, target_fps=None):
    """Process video: extract frames and run pipeline"""
    project_dir, video_name = create_directory_structure(video_path)
    
    print(f"Project created at: {project_dir}")
    print(f"Processing video: {video_path}")
    
    # Step 1: Extract frames
    print("\n=== Step 1: Extracting frames ===")
    frames_dir = os.path.join(project_dir, "frames")
    cmd = ["python", "video_to_frames.py", 
           "--video", video_path, 
           "--output", frames_dir]
    if target_fps:
        cmd.extend(["--fps", str(target_fps)])
    subprocess.call(cmd)
    
    # Step 2: Extract audio
    print("\n=== Step 2: Extracting audio ===")
    audio_path = extract_audio(video_path, os.path.join(project_dir, "audio"))
    
    # Step 3: Run camera tracking
    print("\n=== Step 3: Running camera tracking ===")
    colmap_dir = os.path.join(project_dir, "colmap")
    subprocess.call([
        "python", "camera_tracking.py",
        "--images", frames_dir,
        "--output", colmap_dir
    ])
    
    # Step 4: Downsample point cloud for faster training
    print("\n=== Step 4: Downsampling point cloud ===")
    point_cloud_path = os.path.join(colmap_dir, "points3D.ply")
    downsampled_path = os.path.join(project_dir, "points3D_downsampled.ply")
    subprocess.call([
        "python", "scripts/downsample_point.py",
        point_cloud_path, downsampled_path
    ])
    
    # Step 5: Train 4D Gaussian model
    print("\n=== Step 5: Training 4D Gaussian model ===")
    expname = f"video/{video_name}"
    subprocess.call([
        "python", "train_4d_gaussians.py",
        "--scene_dir", project_dir,
        "--expname", expname
    ])
    
    print(f"\n=== Processing complete! ===")
    print(f"Model saved to: output/{expname}")
    print(f"Run the web viewer to visualize: python app.py")
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert video to 4D Gaussian scene")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--fps", type=float, default=None, 
                      help="Target FPS for extraction (default: original video FPS)")
    
    args = parser.parse_args()
    process_video(args.video, "projects", args.fps)