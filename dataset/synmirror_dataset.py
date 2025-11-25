import os
from typing import Any, Callable, Dict, List, Optional, Union
import cv2
import numpy as np
from pathlib import Path
import torch
import pandas as pd
from torch.nn import functional as F
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from tqdm import tqdm

from src.pipelines import FluxRegionalPipeline
from utils.utils import mask2box
from utils.visualizer import Visualizer

CLASSNAMES = {
    1: 'mirror frame',
    2: 'object',
    3: 'reflection'
}

class SynMirrorDataset(Dataset):
    def __init__(
        self,
        image_root,
        # segm_root is not used but kept for compatibility if needed, or we can ignore it
        segm_root=None, 
        is_group_bucket=False,
        cache_root=None,
        resolution: Union[List, int] = 512,
        cond_scale_factor: int = 1,
    ):
        super(SynMirrorDataset, self).__init__()
        self.image_root = image_root
        self.resolution = [resolution, resolution] if isinstance(resolution, int) else resolution
        self.cond_resolution = [self.resolution[0] // cond_scale_factor, self.resolution[1] // cond_scale_factor]

        # SynMirror structure: root/subfolder/image.png, etc.
        # We scan for all subfolders containing image.png
        self.data = []
        if not os.path.exists(image_root):
             # Just a warning or empty list if path doesn't exist yet (e.g. user hasn't mounted it)
             pass

        # Walk through directory to find samples
        # Assuming one level of subfolders or recursive search
        if os.path.exists(image_root):
            for root, dirs, files in os.walk(image_root):
                if 'image.png' in files:
                    # Check for required masks
                    if 'mirror_mask.png' in files and 'object_mask.png' in files and 'reflection_mask.png' in files:
                        self.data.append(root)
        
        if len(self.data) == 0:
            print(f"Warning: No valid samples found in {image_root}")

        self.image_transforms = transforms.Compose(
            [
                transforms.Resize(self.resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.cond_transforms = transforms.Compose(
            [
                transforms.Resize(self.cond_resolution, interpolation=transforms.InterpolationMode.BILINEAR),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )

        self.visualizer = Visualizer()
        
        # Group bucket logic is omitted for simplicity unless requested, 
        # as it requires pre-computing token lengths which might be complex for custom datasets without cache.
        # If needed, we can add it back. For now, we disable it or handle it simply.
        if is_group_bucket:
             print("Warning: Group bucketing is not fully implemented for SynMirrorDataset yet. Ignoring.")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample_path = self.data[idx]
        
        image_path = os.path.join(sample_path, 'image.png')
        mirror_mask_path = os.path.join(sample_path, 'mirror_mask.png')
        object_mask_path = os.path.join(sample_path, 'object_mask.png')
        reflection_mask_path = os.path.join(sample_path, 'reflection_mask.png')
        caption_path = os.path.join(sample_path, 'object_caption.txt')

        # Load Image
        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size

        # Load Masks
        mirror_mask = np.array(Image.open(mirror_mask_path).convert('L'))
        object_mask = np.array(Image.open(object_mask_path).convert('L'))
        reflection_mask = np.array(Image.open(reflection_mask_path).convert('L'))

        # Create Unified Segmentation Map
        # 0: Background, 1: Mirror, 2: Object, 3: Reflection
        # Priority: Reflection > Object > Mirror (later overwrites earlier)
        segm_map = np.zeros_like(mirror_mask, dtype=np.int64)
        
        # Assuming masks are binary (0 or 255) or similar. Threshold just in case.
        segm_map[mirror_mask > 127] = 1
        segm_map[object_mask > 127] = 2
        segm_map[reflection_mask > 127] = 3

        # Load Caption
        global_caption = ""
        if os.path.exists(caption_path):
            with open(caption_path, 'r') as f:
                global_caption = f.read().strip()

        boxes = []
        cat_names = []
        label = []
        regional_captions = []

        label_id_list = np.unique(segm_map).tolist()

        for label_id in label_id_list:
            if label_id == 0:  # 0 is unlabel/background
                continue
            
            class_name = CLASSNAMES[label_id]
            mask = segm_map == label_id
            
            x0, y0, x1, y1 = mask2box(mask)
            # Normalize box coordinates
            box = np.array([
                x0 / img_w,
                y0 / img_h,
                x1 / img_w,
                y1 / img_h,
            ])
            
            boxes.append(box)
            label.append(mask)
            cat_names.append(class_name)
            # Use class name as regional caption for now, or global caption if appropriate?
            # ADE20K uses class name. Let's stick to that for regional prompts.
            regional_captions.append(class_name)

        if len(regional_captions) == 0:
            # If no labels found (empty image?), try another sample
            return self.__getitem__(np.random.randint(len(self)))

        label = np.stack(label, axis=0)  # n,h,w
        label = torch.from_numpy(label)
        label = label[None, ...]
        label = F.interpolate(label.float(), size=self.resolution, mode='nearest-exact')
        label = label[0, ...].long()  # n,h,w

        pixel_values = self.image_transforms(image)  # c,h,w

        cond_pixel_values = np.zeros([label.shape[-2], label.shape[-1], 3], dtype=np.uint8)
        cond_pixel_values = self.visualizer.draw_contours(
            cond_pixel_values,
            label.cpu().numpy(),
            thickness=1,
            colors=[(255, 255, 255), ] * len(regional_captions)
        )
        cond_pixel_values = Image.fromarray(cond_pixel_values)
        cond_pixel_values = self.cond_transforms(cond_pixel_values)

        return {
            "label": label,
            "regional_captions": regional_captions,
            "global_caption": global_caption,
            "pixel_values": pixel_values,
            "cond_pixel_values": cond_pixel_values,
            "image_name": os.path.basename(sample_path), # Use folder name or image name
            "image_path": image_path,
            "boxes": boxes
        }
