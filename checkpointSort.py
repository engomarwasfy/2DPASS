import glob
import json
import os

import torch


def check_points_sort(checkpoint_dir,save_path=None,load_path=None):
    # Define the directory where your checkpoints are saved
    #checkpoint_dir = 'default3'
    # Get a list of all checkpoint files in the directory
    if load_path is not None:
        with open(load_path) as json_file:
            report = json.load(json_file)
        return report

    checkpoint_files = glob.glob(os.path.join(checkpoint_dir, '*.ckpt'))
    # Instantiate the ModelCheckpoint callback
    report = {
        "checkpoints": []
    }
    # Load a saved checkpoint
    for checkpoint_file in checkpoint_files:
        checkpoint = torch.load(checkpoint_file)
        callbacks = checkpoint['callbacks']

        # Access the current score from the loaded checkpoint
        for key, value in callbacks.items():
            if 'current_score' in value:
                current_score = value['current_score'].item()
                entry = {
                    "path": checkpoint_file,
                    "miou": current_score
                }
                report['checkpoints'].append(entry)
                break
    # Sort the list of dictionaries based on accuracy
    sorted_list = sorted(report["checkpoints"], key=lambda x: x["miou"], reverse=True)
    # Create a new dictionary with the sorted list
    sorted_dict = {
        "checkpoints": sorted_list
    }
    for entry in sorted_dict['checkpoints']:
        print(entry['path'], entry['miou'])

    if save_path is not None:
        with open(save_path, "w") as json_file:
            json.dump(sorted_dict, json_file, indent=4)
        print(f"Sorted dict has been saved to {save_path}")

    return sorted_dict






