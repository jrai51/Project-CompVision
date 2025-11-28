import torch
from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import os
from PIL import Image
import numpy as np
import cv2
from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser(description="Run depth estimation on all image files (png, jpg, jpeg) within a directory and \
                                 save raw relative depth estimate (normalized per image) to a specified directory as .npy files.")

parser.add_argument("--input_dir", help="Input images file directory")
parser.add_argument("--output_dir",help="Output depth estimate files directory.")

args = parser.parse_args()

if __name__ == "__main__": 
    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint = "depth-anything/Depth-Anything-V2-base-hf"
    processor = AutoImageProcessor.from_pretrained(checkpoint, use_fast=True)
    model = AutoModelForDepthEstimation.from_pretrained(checkpoint).to(device)


    if not args.input_dir and not args.output_dir:
        input_dir = "val_selection_cropped/test_images_dontuse"  # Replace with your input directory
        output_dir = "val_selection_cropped/test_depth_estimates_dontsave"  # Replace with your output directory

    else:
        input_dir = args.input_dir
        output_dir = args.output_dir

    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    for filename in tqdm(os.listdir(input_dir)):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            
            # Load the image
            image = Image.open(image_path).convert("RGB")
            w, h = image.size  # PIL gives (width, height)

            inputs = processor(images=image, return_tensors="pt").to(device)

            # Perform depth estimation
            with torch.no_grad():
                outputs = model(**inputs) # 

            post_processed = processor.post_process_depth_estimation(outputs,
                                                                    target_sizes=[(h, w)],  # ensure SAME resolution as RGB
                                                                    )
            predicted_depth = post_processed[0]["predicted_depth"]  # (H, W), float32-ish

            # move to cpu + numpy
            depth_np = predicted_depth.detach().cpu().numpy().astype("float32")

            # print(f"depth_np size: {depth_np.shape} \n rgb image size: {(h, w)}")

            
            # per-image normalization to [0,1] for stability -- These are NOT true depths, only structure
            d_min, d_max = depth_np.min(), depth_np.max()
            depth_np = (depth_np - d_min) / (d_max - d_min + 1e-8)
            
            # Construct output path and save
            output_filename = os.path.splitext(filename)[0] + "_depth_estimate.npy"
            output_path = os.path.join(output_dir, output_filename)
            np.save(output_path, depth_np)

            # print(f"Processed {filename} and saved depth map to {output_path}")