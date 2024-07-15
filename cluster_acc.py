""" Clustering accuracy evaluation
"""

import numpy as np

# read labels from file.npy
labels = np.load("labels.npy")


def purity(labels_pred, labels_true=labels):
    """Compute the purity of the clustering"""
    # compute the confusion matrix
    n = len(labels_true)
    confusion_matrix = np.zeros((n, n))
    for i in range(n):
        confusion_matrix[labels_pred[i], labels_true[i]] += 1

    # compute the purity
    return np.sum(np.amax(confusion_matrix, axis=0)) / n


def adj_rand_index(labels_pred, labels_true=labels):
    """Compute the adjusted Rand index"""
    # compute the confusion matrix
    n = len(labels_true)
    confusion_matrix = np.zeros((n, n))
    for i in range(n):
        confusion_matrix[labels_pred[i], labels_true[i]] += 1

    # compute the adjusted Rand index
    n_ij = np.sum(confusion_matrix**2)
    n_i = np.sum(confusion_matrix, axis=1)**2
    n_j = np.sum(confusion_matrix, axis=0)**2
    n_choose_2 = n * (n - 1) / 2
    expected = np.outer(n_i, n_j) / n_choose_2
    return (n_ij - np.sum(expected)) / (0.5 * (n_i + n_j) - np.sum(expected))
