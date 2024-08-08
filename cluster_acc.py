""" Clustering accuracy evaluation
"""

import numpy as np
from sklearn import metrics

# read labels from file.npy
labels = np.load("data/labels.npy")


def purity(labels_pred, labels_true=labels):
    """Compute the purity of the clustering"""
    confusion_matrix = metrics.cluster.contingency_matrix(
        labels_true, labels_pred)
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


def adj_rand_index(labels_pred, labels_true=labels):
    """Compute the adjusted Rand index"""
    return metrics.adjusted_rand_score(labels_true, labels_pred)
