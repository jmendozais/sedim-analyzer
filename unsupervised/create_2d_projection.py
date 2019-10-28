import argparse
import numpy as np
import subprocess
import os, sys
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas
import matplotlib.pyplot as plt
import time
import seaborn

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Creates a 2D projection from a dataset')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_dataset', type=str, help='Input dataset (.npy array)', required=True)
    required_named.add_argument('-o', '--output_plot', type=str, help='Output plot (.png)', required=True)
    parser.add_argument("--apply_pca", action="store_true", default=False, help="Apply PCA dimensionality reduction")
    parser.add_argument("--n_dim_pca", type=int, default=30, help="Number of dimensions for PCA")
    parser.add_argument("--apply_tsne", action="store_true", default=False, help="Apply t-SNE dimensionality reduction")
    parser.add_argument("--perplexity_tsne", type=int, default=30, help="Perpexity for t-sne")
    parser.add_argument("--n_iter_tsne", type=int, default=1000, help="Number of iterations for t-sne")
    parser.add_argument("--output_dataset", type=str, default="", help="Output dataset (.npy)")
    parser.add_argument("--input_labels", type=str, default="", help="Input labels for the dataset (.csv)")
    args = parser.parse_args()

    # read input parameters
    input_dataset = args.input_dataset
    output_plot = args.output_plot
    apply_pca = args.apply_pca
    n_dim_pca = args.n_dim_pca
    apply_tsne = args.apply_tsne
    perplexity_tsne = args.perplexity_tsne
    n_iter_tsne = args.n_iter_tsne
    output_dataset = args.output_dataset
    input_labels = args.input_labels

    # read the dataset
    dataset = np.load(input_dataset)

    # validate some parameters
    if not apply_pca and not apply_tsne and dataset.shape[1] > 2:
        sys.exit("You must choose to apply PCA or t-SNE or both")
    if apply_pca and not apply_tsne and n_dim_pca > 2:
        sys.exit("If you are applying only PCA, n_dim_pca must be 2 (to create the final plot)")
    # if apply_pca and apply_tsne and n_dim_pca <= 2:
    #     sys.exit("If you are applying PCA and t-SNE, n_dim_pca must be greater than 2 (since t-SNE will reduce to 2 dim)")

    if dataset.shape[1] > 2:
        # apply PCA
        if apply_pca:
            print("-> Applying PCA ... (n_samples: {}, n_feats: {}, n_dim_pca: {})".format(
                dataset.shape[0], dataset.shape[1], n_dim_pca))
            time_start = time.time()
            pca = PCA(n_components=n_dim_pca)
            results = pca.fit_transform(dataset)
            print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
        else:
            results = dataset

        # create the t-sne projection
        if apply_tsne:
            n_dim_tsne = 2
            print("-> Applying t-SNE ... (n_samples: {}, n_feats: {}, n_dim_tsne: {}, perplexity: {}, n_iter: {})".format(
                results.shape[0], results.shape[1], n_dim_tsne, perplexity_tsne, n_iter_tsne))
            time_start = time.time()
            tsne = TSNE(n_components=n_dim_tsne, verbose=1, perplexity=perplexity_tsne, n_iter=n_iter_tsne)
            results = tsne.fit_transform(results)
            print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
    else:
        print("Warning: the dataset has only 2 dimensions, no dimensionality reduction will be applied")
        results = dataset

    # load the samples' labels
    if input_labels != "":
        labels = np.loadtxt(input_labels)
    else:
        labels = [1 for i in range(dataset.shape[0])]

    # create a data frame with the results
    data_frame = pandas.DataFrame()
    data_frame['dim-1'] = results[:,0]
    data_frame['dim-2'] = results[:,1]
    # if input_labels != "":
    data_frame['labels'] = labels

    # create the plot
    plt.figure(figsize=(16,10))
    seaborn.scatterplot(
        x="dim-1", y="dim-2",
        hue="labels",
        palette=seaborn.color_palette("hls", len(set(labels))),
        data=data_frame,
        legend="full",
        alpha=0.8
    )
    plt.savefig(output_plot)
    print("Created plot '{}'".format(output_plot))

    if output_dataset != "":
        np.save(output_dataset, results)
        print("Created dataset '{}'".format(output_dataset))
