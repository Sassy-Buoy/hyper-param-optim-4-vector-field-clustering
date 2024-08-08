""" This script creates a numpy array of labels for the labelled plots. """

import os
import numpy as np

# create an array like labels, but with all zeros
lit_labels = []
for label in os.listdir("data/labelled_plots"):
    for image in os.listdir(f"data/labelled_plots/{label}"):
        i = image.split(".")[0].split("_")[1]
        i = int(i)
        while len(lit_labels) <= i:
            lit_labels.append(0)
        lit_labels[i] = label

np.save("data/labels.npy", lit_labels)
