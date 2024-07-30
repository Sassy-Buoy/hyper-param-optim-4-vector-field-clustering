"""bleh"""

import os
import numpy as np

# load labels from labels.npy
labels = np.load("labels.npy")

# create an array like labels, but with all zeros
lit_labels = [0 for i in range(len(labels))]

for label in os.listdir("plots"):
    for image in os.listdir(f"plots/{label}"):
        i = image.split(".")[0]
        i = i.split("_")[1]
        lit_labels[int(i)] = label

print(lit_labels)

np.save("labels.npy", lit_labels)
