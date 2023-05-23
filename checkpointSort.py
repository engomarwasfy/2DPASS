from pytorch_lightning.callbacks import ModelCheckpoint
import torch
import glob
import os
# Define the directory where your checkpoints are saved
checkpoint_dir = 'default2'

# Get a list of all checkpoint files in the directory
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




