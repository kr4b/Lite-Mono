import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

histogram_path1 = "histogram_kitti.npy"
histogram_path2 = "histogram_middlebury.npy"

MIN_DEPTH = 1e-3
MAX_DEPTH = 80

n = 200
display_bins = 200#int(55/80*n)

histogram1 = np.load(histogram_path1)
histogram2 = np.load(histogram_path2)

fig = plt.figure()
ax = fig.add_subplot(111)
xs = np.histogram_bin_edges([], bins=n, range=(MIN_DEPTH, MAX_DEPTH))[:-1][:display_bins]
width = (MAX_DEPTH-MIN_DEPTH)/n

ax.bar(xs, histogram1[:display_bins], width=width, alpha=0.5)
ax.bar(xs, histogram2[:display_bins], width=width, alpha=0.5)
ax.legend(["Kitti", "Middlebury"])
ax.set_xlabel("Depth")

ax.yaxis.set_major_formatter(ticker.PercentFormatter(xmax=1))

plt.savefig("histogram_combined.png")
