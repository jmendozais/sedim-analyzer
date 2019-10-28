import argparse
import numpy as np
import os, sys, shutil
from skimage.io import imread
from skimage.feature import (corner_peaks, corner_harris, BRIEF, ORB)
from skimage.color import rgb2gray
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import train_test_split
import time
import csv

#========================================================================================#
# function to extract local image descriptors using ORB (Oriented FAST and rotated BRIEF)
#========================================================================================#
def extract_local_img_descriptors(img, local_desc_algthm, max_n_local_feats_orb):
    img = rgb2gray(img)

    if local_desc_algthm == 'ORB':
        detector_extractor = ORB(n_keypoints=max_n_local_feats_orb)
        detector_extractor.detect_and_extract(img)
        descriptors = detector_extractor.descriptors
    elif local_desc_algthm == 'BRIEF':
        keypoints = corner_peaks(corner_harris(img), min_distance=5)
        extractor = BRIEF()
        extractor.extract(img, keypoints)
        keypoints = keypoints[extractor.mask]
        descriptors = extractor.descriptors

    return descriptors

#========================================================================#
# function to compute the fisher vector from an image given a GMM model
#========================================================================#
def compute_fisher_vector(xx, gmm):
    """Computes the Fisher vector on a set of descriptors.
    Reference
    ---------
    J. Krapac, J. Verbeek, F. Jurie.  Modeling Spatial Layout with Fisher
    Vectors for Image Categorization.  In ICCV, 2011.
    http://hal.inria.fr/docs/00/61/94/03/PDF/final.r1.pdf
    """
    xx = np.atleast_2d(xx)
    N = xx.shape[0]

    # Compute posterior probabilities.
    Q = gmm.predict_proba(xx)  # NxK

    # Compute the sufficient statistics of descriptors.
    Q_sum = np.sum(Q, 0)[:, np.newaxis] / N
    Q_xx = np.dot(Q.T, xx) / N
    Q_xx_2 = np.dot(Q.T, xx ** 2) / N

    # Compute derivatives with respect to mixing weights, means and variances.
    d_pi = Q_sum.squeeze() - gmm.weights_
    d_mu = Q_xx - Q_sum * gmm.means_
    d_sigma = (
        - Q_xx_2
        - Q_sum * gmm.means_ ** 2
        + Q_sum * gmm.covariances_
        + 2 * Q_xx * gmm.means_)

    # Merge derivatives into a vector.
    return np.hstack((d_pi, d_mu.flatten(), d_sigma.flatten()))


#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Extracts the feature vectors from the images using Fisher Vectors')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the images', required=True)
    required_named.add_argument('-o', '--output_file', type=str, help='Output file, numpy array (.npy)', required=True)
    parser.add_argument('--train_perc', type=float, default=0.5, help='Training percentage (to learn the clustering model)')
    parser.add_argument('--local_desc_algthm', type=str, default="ORB", choices=['ORB','BRIEF'], help='Local feature extraction algorithm')
    parser.add_argument("--fisher_vectors_all_imgs", action="store_true", default=False, help="Compute the feature vectors for the entire dataset (including training images)")
    parser.add_argument('--max_n_local_feats_orb', type=int, default=200, help='Maximum number of local features to be used with ORB')
    parser.add_argument('--n_groups_gmm', type=int, default=500, help='Number of groups for the GMM algorithm (number of visual words)')
    parser.add_argument("--use_disk_buffer_feat_vect", action="store_true", default=False, help="Use a disk buffer to store the feature\
                            vectors temporally")
    args = parser.parse_args()

    # read input parameters
    input_folder = args.input_folder
    output_file = args.output_file
    train_perc = args.train_perc
    local_desc_algthm = args.local_desc_algthm
    fisher_vectors_all_imgs = args.fisher_vectors_all_imgs
    max_n_local_feats_orb = args.max_n_local_feats_orb
    n_groups_gmm = args.n_groups_gmm
    use_disk_buffer_feat_vect = args.use_disk_buffer_feat_vect

    # clean previous temp files
    if os.path.isdir("_temp_buffer"):
        shutil.rmtree("_temp_buffer")

    # read the files in the directory
    files = os.listdir(input_folder)
    files.sort()
    files_train, files_test = train_test_split(files, test_size=1.0-train_perc)

    # process the training set
    time_start = time.time()
    files_train_zip = zip(range(1, len(files_train)+1), files_train)
    print("-> Computing {} image descriptors from training set ...".format(local_desc_algthm))
    n_errors = 0
    local_feat_vects = []
    for i, file in files_train_zip:
        print("Processing image {}/{} ...\r".format(i, len(files_train)), end="")
        # read image
        img = imread(os.path.join(input_folder, file))

        # extract local feature vectors
        try:
            feats = extract_local_img_descriptors(img, local_desc_algthm, max_n_local_feats_orb)
            local_feat_vects.append(feats)
        except:
            n_errors += 1
    local_feat_vects = np.concatenate(local_feat_vects, axis=0)
    print('\nDone! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
    if n_errors > 0:
        print("Warning: the {} descriptor could not find any local descriptors in {} images!".format(local_desc_algthm, n_errors))

    # create the GMM model
    time_start = time.time()
    print("\n-> Creating the GMM clustering model ... (n_samples: {}, n_feats_per_samp: {}, n_groups: {})".format(local_feat_vects.shape[0], local_feat_vects.shape[1], n_groups_gmm))
    gmm = GaussianMixture(n_components=n_groups_gmm, covariance_type='diag')
    gmm.fit(local_feat_vects)
    print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))

    # compute the fisher vectors
    if use_disk_buffer_feat_vect:
        os.makedirs("_temp_buffer")
    if fisher_vectors_all_imgs:
        files_test = files

    files_test_zip = zip(range(1, len(files_test)+1), files_test)
    time_start = time.time()
    n_errors = 0
    feat_vects = []
    file_ids = []
    
    print("\n-> Computing the fisher vectors ...")
    for i, file in files_test_zip:
        print("Processing image {}/{} ...\r".format(i, len(files_test)), end="")
        # read image
        img = imread(os.path.join(input_folder, file))

        # extract local feature vectors and compute the fisher vectors on them
        try:
            local_feats = extract_local_img_descriptors(img, local_desc_algthm, max_n_local_feats_orb)
            fisher_vector = np.atleast_2d(compute_fisher_vector(local_feats, gmm))
            if use_disk_buffer_feat_vect:
                np.save(os.path.join("_temp_buffer", "feat_vect_{}.npy".format(i)), fisher_vector)
            else:
                feat_vects.append(fisher_vector)
            file_ids.append([file])
        except:
            n_errors += 1
    feat_vects = np.concatenate(feat_vects, axis=0)
    print('\nDone! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
    if n_errors > 0:
        print("Warning: the {} descriptor could not find any local descriptors in {} images!".format(local_desc_algthm, n_errors))

    # save the vectors
    time_start = time.time()
    print("\n-> Saving the feature vectors ...")
    if use_disk_buffer_feat_vect:
        feat_vects = []
        for i in range(1, len(files_test)+1):
            print("Reading feat vectors from buffer {}/{} ...\r".format(i, len(files_test)), end="")
            filename = os.path.join("_temp_buffer", "feat_vect_{}.npy".format(i))
            if os.path.isfile(filename):
                fisher_vect = np.load(filename)
                feat_vects.append(fisher_vector)
                os.remove(filename)
        feat_vects = np.concatenate(feat_vects, axis=0)
        shutil.rmtree("_temp_buffer")
        print()
    np.save(output_file, feat_vects)
    print('Done! Time elapsed: {:.2f} seconds'.format(time.time()-time_start))
    print("Created file '{}' containing {} feature vectors with {} features each".format(output_file, feat_vects.shape[0], feat_vects.shape[1]))

    # save the file ids
    filename = "{}_ids.csv".format(os.path.join(os.path.dirname(output_file), os.path.basename(output_file).split(".")[0]))
    with open(filename, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerows(file_ids)
    print("Created file '{}' containing the IDs of the images".format(filename))