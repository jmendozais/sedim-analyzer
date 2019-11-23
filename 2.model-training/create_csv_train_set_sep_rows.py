#!/usr/bin/python

# author: Cesar Castelo
# date: Nov 22, 2019
# description: Creates a CSV with the train set labels (one row per patch) from another CSV with the train set labels (one row per image and one column per patch)

import os
import argparse
import pandas as pd

#===============#
# main function
#===============#
if __name__ == "__main__":
    # verify input parameters
    parser = argparse.ArgumentParser('Creates a CSV with the train set labels (one row per patch) from another CSV with one row per image and one column per patch')
    required_named = parser.add_argument_group('required named arguments')
    required_named.add_argument('-i', '--input_csv', type=str, help='Input CSV containing one row per image and one column per patch', required=True)
    required_named.add_argument('-o', '--output_csv', type=str, help='Output CSV containing one row per patch', required=True)
    args = parser.parse_args()

    # read input parameters
    input_csv = args.input_csv
    output_csv = args.output_csv

    # read the input CSV
    data = pd.read_csv(input_csv, delimiter=',', header=0)

    file_list = []
    for index, row in data.iterrows():
        filename = os.path.splitext(row[0])[0]
        file_list.append(["{}_patch_{}.png".format(filename, 1), row[1]])
        file_list.append(["{}_patch_{}.png".format(filename, 2), row[2]])
        file_list.append(["{}_patch_{}.png".format(filename, 3), row[3]])
        file_list.append(["{}_patch_{}.png".format(filename, 4), row[4]])

    pd.DataFrame(file_list, columns=["Patch", "Etiqueta"]).to_csv(output_csv, sep=",", index=False, header=True)