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

categories = list(set(lit_labels))
categories.sort()

lables = np.zeros_like(labels)

for i, label in enumerate(lit_labels):
    labels[i] = categories.index(label)


np.save("labels.npy", labels)

# write categories and indices to file
with open("categories.txt", "w", encoding="utf-8") as file:
    for i, category in enumerate(categories):
        file.write(f"{i}: {category}\n")
