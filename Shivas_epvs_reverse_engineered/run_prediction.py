''' modify the scipt below to run the prediction for all the images in the image_dir
    the script will read the images from image_dir, run the prediction and save the masks in the masks_dir'''

''' SCRIPT MOSTLY GENERATED BY CURSOR AND CLAUDE SONNET PLEASE USE WITH CAUTION'''

import os
from pathlib import Path
import json
import subprocess

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def run_prediction(config):
    image_dir = Path(config['image_dir'])
    masks_dir = Path(config['masks_dir'])
    masks_dir.mkdir(parents=True, exist_ok=True)

    for image_file in image_dir.glob('*.nii'):
        # Build the command
        command = [
            "python", "./predict_one_file.py",
        ]
        
        if config.get("verbose", False):
            command.append("--verbose")
        
        command.extend(["--gpu", str(config["gpu"])])
        
        # Add each model path
        for model in config["models"]:
            model_path = Path(config['model_dir']) / model
            command.extend(["-m", str(model_path)])
        
        # Add input and output image paths
        input_image_path = str(image_file)
        output_image_path = str(masks_dir / f"{image_file.stem}_mask.nii")
        
        command.extend(["-i", input_image_path])
        command.extend(["-o", output_image_path])
        
        # Execute the command
        subprocess.run(command)

if __name__ == "__main__":
    config = load_config("config.json")
    run_prediction(config)

