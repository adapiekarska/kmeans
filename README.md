# kmeans
KMeans python implementation with use example.

To see the example, which is the image color reduction, run clustering_example.py

To use k-means algorithm in your code just call the function:

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

Copyright 2018, Ada Piekarska, All rights reserved

