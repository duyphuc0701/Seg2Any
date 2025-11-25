import h5py
import numpy as np
from PIL import Image
from pathlib import Path

def extract_image_from_hdf5(hdf5_path: str, output_path: str):
    """extract the image from the hdf5_path file"""

    hdf5_data = h5py.File(hdf5_path, "r")

    colors = np.array(hdf5_data["colors"], dtype=np.uint8)

    img = Image.fromarray(colors[0])
    img.save(output_path)