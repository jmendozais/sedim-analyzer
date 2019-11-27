# Author: Julio Mendoza
import argparse
import numpy as np
import subprocess
import os, sys
from shutil import copyfile
import pandas as pd
import random

def load_folds_as_dataframes(folds_folder):
    files = os.listdir(folds_folder)

    fold_prefixes=set()
    for f in files:
        fold_prefixes.add(os.path.join(folds_folder, f[:-6]))
    fold_prefixes = sorted(list(fold_prefixes))
    print("FOLDS")
    print(fold_prefixes)

    fold_dataframes = []
    for fold_prefix in fold_prefixes:
        data_tr = pd.read_csv(fold_prefix + "tr.csv")
        data_te = pd.read_csv(fold_prefix + "te.csv")
        fold_dataframes.append((data_tr, data_te))

    return fold_dataframes

if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Split the image set into training and validation sets')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_folder', type=str, help='Folder containing the images', required=True)
    required_named.add_argument('-o', '--output_folder', type=str, help='Output folder', required=True)
    required_named.add_argument('-y', '--labels_file', type=str, help='labels', required=True)
    required_named.add_argument('-e', '--file_ext', type=str, help='File extension', required=True)
    args = parser.parse_args()

    if not os.path.isdir(args.output_folder):
        os.makedirs(args.output_folder)

    data = pd.read_csv(args.labels_file, sep=",", header=1)
    imgs = list(data.iloc[:,0])

    major_classes = []
    for i in range(len(imgs)):
        classes = set([data.iloc[i,j] for j in range(1, 5)])
        major_class="None"
        major_class_count=0
        for c in classes:
            count = 0
            for j in range(4):
                if c == data.iloc[i,j+1]:
                    count += 1
            if major_class_count < count:
                major_class_count = count
                major_class = c
        
        major_classes.append(major_class)
        #if major_class not in filenames_by_class:
        #    filenames_by_class[major_class] = []
        #ids_by_class[major_class].append(imgs[i])

    assert data.shape[0] == len(major_classes)
    data['major_class'] = np.array(major_classes)

    # TODO: division por porcentaje
    # TODO: guardar en txts
    from sklearn.model_selection import StratifiedKFold

    skf = StratifiedKFold(n_splits=5)
    target = data['major_class']
    i = 0
    for train_index, test_index in skf.split(target, target):
        data_tr, data_te = data.loc[train_index], data.loc[test_index]
        for j in range(len(data_tr)):
            data_tr.iloc[j,0] = data_tr.iloc[j,0][:-3] + args.file_ext
        for j in range(len(data_te)):
            data_te.iloc[j,0] = data_te.iloc[j,0][:-3] + args.file_ext

        data_tr.to_csv(os.path.join(args.output_folder, "fold-" + str(i) + '-tr.csv'))
        data_te.to_csv(os.path.join(args.output_folder, "fold-" + str(i) + '-te.csv'))
        i += 1
