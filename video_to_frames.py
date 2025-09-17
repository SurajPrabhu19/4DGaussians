#!/usr/bin/env python3
import os
import cv2
import numpy as np
import argparse
from tqdm import tqdm

def extract_frames(video_path, output_dir, fps=None):
    """
    Extract frames from video at specified fps or original fps
    
    Args:
        video_path: Path to input video
        output_dir: Directory to save extracted frames
        fps: Target frames per second (None for original fps)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video {video_path}")
    
    # Get video properties
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Determine frame extraction rate
    target_fps = fps if fps is not None else orig_fps
    frame_interval = max(1, round(orig_fps / target_fps))
    
    print(f"Video info: {total_frames} frames, {orig_fps} fps")
    print(f"Extracting at {target_fps} fps (every {frame_interval} frames)")
    
    # Extract frames
    frame_idx = 0
    output_idx = 0
    with tqdm(total=total_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            if frame_idx % frame_interval == 0:
                output_path = os.path.join(output_dir, f"frame_{output_idx:06d}.png")
                cv2.imwrite(output_path, frame)
                output_idx += 1
                
            frame_idx += 1
            pbar.update(1)
    
    cap.release()
    print(f"Extracted {output_idx} frames to {output_dir}")
    return output_idx

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Extract frames from video")
    parser.add_argument("--video", type=str, required=True, help="Path to input video")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--fps", type=float, default=None, help="Target FPS (default: original video FPS)")
    
    args = parser.parse_args()
    extract_frames(args.video, args.output, args.fps)