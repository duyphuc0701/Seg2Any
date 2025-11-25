import sys
import os
sys.path.append(os.getcwd())

import torch
from dataset.synmirror_dataset import SynMirrorDataset
import matplotlib.pyplot as plt
import numpy as np

def test_synmirror_loading():
    print("Testing SynMirrorDataset loading...")
    
    # Use the path provided by the user
    image_root = "d:/APCSThesisCode/Seg2Any/data/SynMirror"
    
    # Check if directory exists, if not, maybe we can't run the test fully but we can check class init
    if not os.path.exists(image_root):
        print(f"Dataset root {image_root} not found. Skipping actual data loading test.")
        return

    dataset = SynMirrorDataset(
        image_root=image_root,
        resolution=512
    )
    
    print(f"Dataset length: {len(dataset)}")
    
    if len(dataset) > 0:
        item = dataset[0]
        print("Keys in item:", item.keys())
        print("Pixel values shape:", item['pixel_values'].shape)
        print("Label shape:", item['label'].shape)
        print("Regional captions:", item['regional_captions'])
        print("Global caption:", item['global_caption'])
        
        # Save visualization
        # Denormalize pixel values for visualization
        img = item['pixel_values'].permute(1, 2, 0).numpy()
        img = (img * 0.5 + 0.5) * 255
        img = img.astype(np.uint8)
        
        # Visualize label
        label = item['label'].numpy()
        
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.imshow(img)
        plt.title("Image")
        
        plt.subplot(1, 2, 2)
        plt.imshow(label[0], cmap='jet')
        plt.title("Label")
        
        os.makedirs("test_results", exist_ok=True)
        plt.savefig("test_results/synmirror_sample.png")
        print("Saved visualization to test_results/synmirror_sample.png")
    else:
        print("Dataset is empty.")

if __name__ == "__main__":
    test_synmirror_loading()
