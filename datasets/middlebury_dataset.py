from __future__ import absolute_import, division, print_function

import struct
import os
import skimage.transform
import numpy as np
from PIL import Image

import torch.utils.data as data
import torch

from kitti_utils import read_pfm

from torchvision.transforms.functional import resize, to_tensor

def pil_loader(path):
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

class MiddleburyDataset(data.Dataset):
    """Middlebury dataset loader
    """
    def __init__(self,
                 data_path,
                 height,
                 width,
                 img_ext='.jpg'):
        super(MiddleburyDataset, self).__init__()

        self.data_path = data_path
        self.height = height
        self.width = width

        dirs = os.listdir(self.data_path)

        self.image_paths = np.array([[os.path.join(self.data_path, d, "im{}.png".format(i)) for d in dirs] for i in range(2)]).flatten()
        self.disp_paths = np.array([[os.path.join(self.data_path, d, "disp{}.pfm".format(i)) for d in dirs] for i in range(2)]).flatten()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, index):
        """Returns a raw color image and corresponding depth from the dataset as a tuple.

        The color image will be cropped and resized to (self.width, self.height).
        The depth will only be cropped.
        """

        image = np.array(pil_loader(self.image_paths[index]))
        calib_file = os.path.join(os.path.dirname(self.image_paths[index]), "calib.txt")
        depth = read_pfm(calib_file, self.disp_paths[index])

        # Resize and crop image to be of size (self.width, self.height)
        aspect_current = image.shape[0] / image.shape[1]
        aspect_goal = self.width / self.height
        if aspect_current > aspect_goal:
            image = np.delete(image, slice(int(aspect_goal / aspect_current * image.shape[0]), image.shape[0]), axis=0)
            depth = np.delete(depth, slice(int(aspect_goal / aspect_current * depth.shape[0]), depth.shape[0]), axis=0)
        else:
            image = np.delete(image, slice(int(aspect_current / aspect_goal * image.shape[1]), image.shape[1]), axis=1)
            depth = np.delete(depth, slice(int(aspect_current / aspect_goal * depth.shape[1]), depth.shape[1]), axis=1)

        image = Image.fromarray(image)
        image = to_tensor(resize(image, size=(self.height, self.width), interpolation=Image.ANTIALIAS))
        # image = torch.from_numpy(np.array(image))

        return (image, depth)
