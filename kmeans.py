"""
To run k-means, just call the function:
kmeans(k, data, min_vals, max_vals, max_iter=50)
with parameters:
k - number of clusters
d - data set, should be ndarray of shape (n, d) where n is the number
of data entries and d is the number of dimensions of each data entry
min_vals, max_vals - array like lists that store min and max values
for each dimension
max_iter - maximal number of iterations.
kmeans(...) returns ndarray of shape (n, d) containing labels for
every data entry.
"""


from random import randint
import numpy as np


def data_preprocess(data):
    """
    Standarize the data set.
    """

    data = np.unique(data, axis=0)
    means = np.mean(data, axis=0)
    stds = np.std(data, axis=0)
    for i in range(len(data)):
        for dim in range(data.shape[1]):
            data[i][dim] = (data[i][dim] - means[dim])/stds[dim]


def kmeans(k, data, min_vals, max_vals, max_iter=50):
    """
    Perform k-means algorithm. k is the number of expected clusters.
    Expects data to be ndarray of shape (n, d), where n is the number
    of data points in the set and d is a number of dimensions.
    min_vals and max_vals are lists of min and max values for each
    dimension, therefore they should be array like of size d.
    Returns ndarray of shape (n, d) containing labels for each data point.
    """

    data_preprocess(data)

    dimensions = data.shape[1]
    min_vals = list(min_vals)
    max_vals = list(max_vals)

    centroids = get_centroids(k, dimensions, min_vals, max_vals)
    old_centroids = np.zeros(centroids.shape)
    it = 0

    while not end_condition_true(centroids, old_centroids, it, max_iter):

        # find nearest centroid for each data point
        labels = get_labels(data, centroids)

        old_centroids = centroids

        # update centroids
        centroids = update_centroids(data, labels, centroids)
        it += 1

    return labels


def end_condition_true(centroids, old_centroids, it, max_iter):
    """
    Check whether the end condition for the k-means algorithm is met.
    That is, whether centroids don't change or the maximum number of
    iterations has been reached.
    """

    return np.array_equal(centroids, old_centroids) or it >= max_iter


def get_centroids(k, dimensions, min_vals, max_vals):
    """
    Returns ndarray of shape (k, d), where k is the number of randomly
    placed centroids and d is the number of dimensions.
    """

    centroids = []

    # randomly place k centroids in the data space
    while len(centroids) < k:
        centroid = tuple(randint(min_vals[dim], max_vals[dim])
                         for dim in range(dimensions))
        if centroid not in centroids:
            centroids.append(centroid)

    return np.array(centroids)


def get_labels(data, centroids):
    """
    Find nearest centroid to each data point. Returns ndarray of shape (n, d)
    containing labels (nearest centroid) of each data point.
    """

    # i-th row of the distances matrix represents distances from each data
    # point to the i-th centroid.
    distances = np.empty(shape=(centroids.shape[0], data.shape[0]))
    for i, c in enumerate(centroids):
        x = data - c
        distances[i] = np.linalg.norm(x, axis=1)

    # i-th element in the nearest_idc matrix represents the index number
    # of the nearest centroid.
    nearest_idc = np.argmin(distances, axis=0)

    return np.array([centroids[i] for i in nearest_idc])


# TODO make this function faster
def update_centroids(data, labels, centroids):
    """
    Update current centroids with respect to the points' labels.
    Returns ndarray of shape (k, d), where k is the number of centroids
    and d is the number of dimensions.
    """

    dimensions = data.shape[1]

    sums = np.zeros(shape=(centroids.shape[0], dimensions), dtype=int)
    labels_occ = np.zeros(shape=(centroids.shape[0], 1))

    for d, l in zip(data, labels):
        for dim in range(dimensions):
            c_idx = get_centroid_idx(centroids, l)
            sums[c_idx, dim] += d[dim]
            labels_occ[c_idx] += 1

    # if there was a centroid with no data points labeled with it,
    # don't update it
    for i, occ in enumerate(labels_occ):
        if occ == 0:
            sums[i] = centroids[i]
        else:
            for dim in range(dimensions):
                sums[i, dim] = np.round(sums[i, dim]/occ)
    return sums


def get_centroid_idx(centroids, c):
    """
    Returns the index of a given centroid c. Assumes that centroids
    is the ndarray of shape (k, d) where k is a number of centroids
    and d is a number od dimensions.
    """

    return centroids.tolist().index(c.tolist())
