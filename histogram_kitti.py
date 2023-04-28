import os
import sys

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

from kitti_utils import generate_depth_map

assert(len(sys.argv) >= 2)

data_path = sys.argv[1]

total_points = 0
total_files = 0

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

n = 200
histogram = np.zeros(n)

for dirname, subdirs, files in os.walk(data_path):
    if not dirname.endswith("velodyne_points/data"):
        continue

    for f in files:
        calib_dir = "/".join(dirname.split("/")[:-3])
        velo_file = os.path.join(dirname, f)

        gt_depth = generate_depth_map(calib_dir, velo_file, 2, True).astype(np.float32)

        histogram += np.histogram(gt_depth[gt_depth > MIN_DEPTH], bins=n, range=(MIN_DEPTH, MAX_DEPTH))[0]

        total_points += np.sum(gt_depth > MIN_DEPTH)
        
    total_files += len(files)

print("Average points: {}".format(total_points / total_files))

fig = plt.figure()
ax = fig.add_subplot(111)

histogram = histogram / total_points

ax.bar(np.histogram_bin_edges([], bins=n, range=(MIN_DEPTH, MAX_DEPTH))[:-1], histogram, width=(MAX_DEPTH-MIN_DEPTH)/n)
ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

plt.savefig("histogram_kitti.png")
np.save("histogram_kitti.npy", histogram)
