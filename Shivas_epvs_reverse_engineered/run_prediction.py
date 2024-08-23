import json
import subprocess

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def run_prediction(config):
    # Build the command
    command = [
        "python", "./predict_one_file.py",
    ]
    
    if config.get("verbose", False):
        command.append("--verbose")
    
    command.extend(["--gpu", str(config["gpu"])])
    
    # Add each model path
    for model in config["models"]:
        model_path = f"{config['model_dir']}/{model}"
        command.extend(["-m", model_path])
    
    # Add input and output image paths
    input_image_path = f"{config['image_dir']}/{config['input_image']}"
    output_image_path = f"{config['output_dir']}/{config['output_image']}"
    
    command.extend(["-i", input_image_path])
    command.extend(["-o", output_image_path])
    
    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    config = load_config("config.json")
    run_prediction(config)
