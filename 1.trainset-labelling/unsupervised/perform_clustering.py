import argparse
import numpy as np
import subprocess
import os, sys
from sklearn.cluster import DBSCAN, MiniBatchKMeans
import time

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Performs clustering from a dataset')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_dataset', type=str, help='Input dataset (.npy array)', required=True)
    required_named.add_argument('-o', '--output_labels', type=str, help='Output labels (.npy array)', required=True)
    parser.add_argument("--clust_algthm", type=str, default="dbscan", choices=["dbscan","minibatch-kmeans","spectral",
        "agglomerartive","optics", "hierarchical"], help="Clustering algorithm")
    parser.add_argument("--n_clusters", type=int, default=0, help="Number of clusters (only for some algorithms)")
    parser.add_argument("--dbscan_eps", type=float, default=0.5, help="DBSCAN parameter: EPS")
    parser.add_argument("--dbscan_min_samples", type=int, default=10, help="DBSCAN parameter: Min number of samples per cluster")
    parser.add_argument("--kmeans_batch_size", type=int, default=10, help="Mini-Batch-Kmeans parameter: Minibatch size")
    args = parser.parse_args()

    # read input parameters
    input_dataset = args.input_dataset
    output_labels = args.output_labels
    clust_algthm = args.clust_algthm

    # read the dataset
    dataset = np.load(input_dataset)

    # perform clustering
    if clust_algthm == "dbscan":
        dbscan_eps = args.dbscan_eps
        dbscan_min_samples = args.dbscan_min_samples
        print("-> Clustering the dataset with DBSCAN ... (n_samples: {}, n_feats: {}, eps: {}, min_samples: {})".format(
            dataset.shape[0], dataset.shape[1], dbscan_eps, dbscan_min_samples))
        time_start = time.time()
        clust = DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples)
        clust.fit(dataset)
        labels = clust.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
        print("Number of clusters found: {}".format(n_clusters))
    elif clust_algthm == "minibatch-kmeans":
        n_clusters = args.n_clusters
        kmeans_batch_size = args.kmeans_batch_size
        print("-> Clustering the dataset with Mini Batch Kmeans ... (n_samples: {}, n_feats: {}, n_clusters: {}, batch_size: {})".format(
            dataset.shape[0], dataset.shape[1], n_clusters, kmeans_batch_size))
        time_start = time.time()
        clust = MiniBatchKMeans(init='k-means++', n_clusters=n_clusters, batch_size=kmeans_batch_size, n_init=10, max_no_improvement=10, verbose=0)
        clust.fit(dataset)
        labels = clust.labels_
        print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))

    # save the results
    #np.save(output_labels, labels)
    np.savetxt(output_labels, labels, delimiter=",", fmt="%d")
