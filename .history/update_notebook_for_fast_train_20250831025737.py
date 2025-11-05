#!/usr/bin/env python3
"""
Update test_inference_epoch10.ipynb to work better with fast_train_orion.py
"""

import json
from pathlib import Path

def update_notebook():
    notebook_path = Path("test_inference_epoch10.ipynb")
    
    # Read the notebook
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    # Find and update the cell with model_dir
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code' and 'source' in cell:
            source = ''.join(cell['source'])
            if 'model_dir = Path("orion_patches_model")' in source:
                print("Found model_dir cell, updating...")
                
                # Update the source
                new_source = []
                for line in cell['source']:
                    if 'model_dir = Path("orion_patches_model")' in line:
                        new_source.append('# Common paths:\n')
                        new_source.append('# - For train_orion_patches.py: "orion_patches_model_aug28" \n')
                        new_source.append('# - For fast_train_orion.py: "fast_orion_model_run" or similar\n')
                        new_source.append('model_dir = Path("fast_orion_model_run")  # Change this to your actual output directory\n')
                    else:
                        new_source.append(line)
                
                cell['source'] = new_source
                break
    
    # Add a configuration cell at the top
    config_cell = {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Test Inference for Fast Training Model\n",
            "\n",
            "This notebook tests inference with models trained using `fast_train_orion.py`.\n",
            "\n",
            "**Before running:**\n",
            "1. Update `model_dir` below to point to your training output directory\n",
            "2. Update `patchnum` to choose which patch to test\n",
            "3. Make sure `pairs_dir` points to your core patches directory\n",
            "\n",
            "**Compatible with:**\n",
            "- `fast_train_orion.py` checkpoints (LightweightUNet)\n",
            "- `train_orion_patches.py` checkpoints (HE2OrionUNet/Advanced)\n"
        ]
    }
    
    # Insert the config cell at the beginning
    notebook['cells'].insert(0, config_cell)
    
    # Save the updated notebook
    with open(notebook_path, 'w') as f:
        json.dump(notebook, f, indent=2)
    
    print(f"Updated {notebook_path}")
    print("Key changes:")
    print("- Added configuration instructions at the top")
    print("- Updated default model_dir to 'fast_orion_model_run'")
    print("- Added comments about common paths")

if __name__ == "__main__":
    update_notebook()
