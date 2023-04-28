import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib as mpl
import matplotlib.cm as cm

import PIL.Image as pil

from kitti_utils import read_pfm

assert(len(sys.argv) >= 2)

data_path = sys.argv[1]

total_points = 0
total_files = 0

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

n = 200
histogram = np.zeros(n)

for dirname, subdirs, files in os.walk(data_path):
    for f in files:
        if not f in ["disp0.pfm" or "disp1.pfm"]:
            continue

        if not "Motorcycle" in dirname or not f == "disp0.pfm":
            continue

        pfm_file = os.path.join(dirname, f)
        calib_file = os.path.join(dirname, "calib.txt")

        gt_disp = read_pfm(calib_file, pfm_file).astype(np.float32)
        # gt_disp = 1.0 / gt_depth
        gt_disp[gt_disp > 1.0 / MIN_DEPTH] = 0.0
        gt_disp[gt_disp < 1.0 / MAX_DEPTH] = 1.0 / MAX_DEPTH

        vmax = np.percentile(gt_disp, 95)
        normalizer = mpl.colors.Normalize(vmin=gt_disp.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap="magma")
        colormapped_im = (mapper.to_rgba(gt_disp)[:, :, :3] * 255).astype(np.uint8)
        im = pil.fromarray(colormapped_im)

        im.save("motorcycle.png")

        # histogram += np.histogram(gt_depth[gt_depth > MIN_DEPTH], bins=n, range=(MIN_DEPTH, MAX_DEPTH))[0]

        # total_points += np.sum(gt_depth > MIN_DEPTH)
        
    total_files += len(files)

print("Average points: {}".format(total_points / total_files))

fig = plt.figure()
ax = fig.add_subplot(111)

histogram = histogram / total_points

ax.bar(np.histogram_bin_edges([], bins=n, range=(MIN_DEPTH, MAX_DEPTH))[:-1], histogram, width=(MAX_DEPTH-MIN_DEPTH)/n)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

plt.savefig("histogram_middlebury.png")
np.save("histogram_middlebury.npy", histogram)
