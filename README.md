# KMeans

## General description

K-Means python implementation with usage example. Coded for learning purposes.

This project contains two main files:

* **kmeans module** can be used in code by incorporating it in the project and usual importing.
* **clustering_example.py** is a small interactive console application created for demonstration purposes.

## Getting started

### Using kmeans module in your code

To use k-means algorithm from module kmeans in your code just call the function:

```
kmeans(k, data, min_vals, max_vals, max_iter=50)
```

with following parameters:
* **k** - number of clusters,
* **d** - data set, should be ndarray of shape (n, d) where n is the number of data entries, and d is the number of dimensions of each data entry,
* **min_vals, max_vals** - array like lists that store min and max values for each dimension,
* **max_iter** - maximal number of iterations.

kmeans() returns ndarray of shape (n, d) containing labels for every entry from the data set.

### Using clustering_example.py

To see the example of clustering that utilizes kmeans module, run:

```
clustering_example.py
```

To run this example project, Python 3.6 is needed.


*Copyright 2018, Ada Piekarska, All rights reserved*

