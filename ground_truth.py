import os
import sys

import numpy as np
import PIL.Image as pil
import matplotlib as mpl
import matplotlib.cm as cm
from kitti_utils import generate_depth_map
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter

assert(len(sys.argv) >= 4)

data_path = sys.argv[1]
image_path = sys.argv[2]
output_path = sys.argv[3]

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

with open(image_path) as f:
    filenames = f.readlines()
    for i in range(len(filenames)):
        filename = filenames[i]
        line = filename.split()
        folder = line[0]
        frame_id = int(line[1])
        side = line[2]

        calib_dir = os.path.join(data_path, folder.split("/")[0])
        velo_filename = os.path.join(
            data_path,
            folder,
            "velodyne_points/data",
            "{:010d}.bin".format(frame_id)
        )

        gt_depth = generate_depth_map(calib_dir, velo_filename, 2, True).astype(np.float32)
        gt_disp = np.zeros(gt_depth.shape)

        gt_disp[gt_depth > MIN_DEPTH] = 1.0 / gt_depth[gt_depth > MIN_DEPTH]

        h, w = gt_disp.shape

        all_points = np.transpose(np.mgrid[0:h, 0:w], (1, 2, 0))
        points = all_points[gt_disp != 0.0]
        values = gt_disp[gt_disp != 0.0]
        samples = all_points[gt_disp == 0.0]

        gt_disp[gt_disp == 0.0] = griddata(points, values, samples, method="nearest")
        gt_disp = gaussian_filter(gt_disp, sigma=3)

        normalizer = mpl.colors.Normalize(vmin=gt_disp.min(), vmax=gt_disp.max())
        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_im = (mapper.to_rgba(gt_disp)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        name_dest_im = os.path.join(output_path, "gt-{:010d}.png".format(frame_id))
        im.save(name_dest_im)

