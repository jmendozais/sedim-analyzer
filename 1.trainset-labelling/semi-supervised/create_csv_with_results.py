import argparse
import numpy as np
import os, sys
import csv
import pandas as pd

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Creates a CSV with the results in the required format')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_csv', type=str, help='Input CSV containing the results (image patch and label)', required=True)
    required_named.add_argument('-o', '--output_csv', type=str, help='Output CSV containing the results (image and labels per patch)', required=True)
    args = parser.parse_args()

    # read input parameters
    input_csv = args.input_csv
    output_csv = args.output_csv

    # read the csv containing the results
    data = pd.read_csv(input_csv, sep=",", header=None)
    imgs_without_patches = list(data.iloc[:,0])
    imgs_without_patches.sort()
    imgs_without_patches = [os.path.splitext(x)[0].split("_") for x in imgs_without_patches]
    imgs_without_patches = list(set(["{}_{}_{}_{}".format(x[0],x[1],x[2],x[3]) for x in imgs_without_patches]))

    # create the output csv
    csv_list = [["Nombre del Archivo", "0 - 25 % de la Imagen", "25 - 50 % de la Imagen", "50 - 75 % de la Imagen", "75 - 100 % de la Imagen"]]
    print("--> Copying the results ...")
    for i in range(len(imgs_without_patches)):
        print("Processing image {}/{} ...\r".format(i+1, len(imgs_without_patches)), end="")
        csv_row = [imgs_without_patches[i]]
        for j in range(1,5):
            filename = "{}_patch_{}.png".format(imgs_without_patches[i], j)
            row = data.loc[data[0] == filename]
            csv_row.append(row.iloc[0,1])
        csv_list.append(csv_row)
    print()

    # save the csv
    with open(output_csv, 'w', newline='') as csv_file:
        wr = csv.writer(csv_file, quoting=csv.QUOTE_ALL)
        wr.writerows(csv_list)
    print("Created file '{}'".format(output_csv))