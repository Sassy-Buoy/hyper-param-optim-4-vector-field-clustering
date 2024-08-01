""" Clustering accuracy evaluation
"""

import numpy as np
from sklearn import metrics

# read labels from file.npy
labels = np.load("labels.npy")


def purity(labels_pred, labels_true=labels):
    """Compute the purity of the clustering"""
    # compute the confusion matrix
    confusion_matrix = metrics.cluster.contingency_matrix(
        labels_true, labels_pred)
    # return the purity
    return np.sum(np.amax(confusion_matrix, axis=0)) / np.sum(confusion_matrix)


def adj_rand_index(labels_pred, labels_true=labels):
    """Compute the adjusted Rand index"""
    # compute the confusion matrix
    return metrics.adjusted_rand_score(labels_true, labels_pred)
