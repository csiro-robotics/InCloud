# PointNetVLAD datasets: based on Oxford RobotCar and Inhouse
# Code adapted from PointNetVLAD repo: https://github.com/mikacuy/pointnetvlad

import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm 
import matplotlib.pyplot as plt 
from glob import glob

FILENAME = "pd_northing_easting.csv"
POINTCLOUD_FOLS = "Ouster"

def output_to_file(output, save_folder, filename):
    file_path = os.path.join(save_folder, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

def construct_query_and_database_sets(folder, save_folder, eval_thresh, time_thresh, file_extension):
    
    # Load data, make distance and time trees
    df_locations = pd.read_csv(os.path.join(folder, FILENAME))
    df_locations['file'] = folder + '/' + POINTCLOUD_FOLS + '/' + df_locations['timestamp'].astype(str) + file_extension
    df_locations['timestamp'] = df_locations['timestamp'] / 1e9

    database = {}
    for index, row in df_locations.iterrows():
        database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                                'easting': row['easting'], 'timestamp': row['timestamp']}
    filename = f'{os.path.basename(folder)}_evaluation_database.pickle'
    output_to_file(database, save_folder, filename)




if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--eval_thresh', type = int, default = 10, help = 'Threshold for positive examples')
    parser.add_argument('--time_thresh', type = int, default = 90, help = 'Threshold for time before re-visit considered')
    parser.add_argument('--file_extension', type = str, default = '.npy', help = 'File extension expected')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save pickle files to')
    args = parser.parse_args()

    # Check dataset root exists, make save dir if doesn't exist
    print('Dataset root: {}'.format(args.dataset_root))
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    
    all_folders = sorted([x for x in glob(os.path.join(base_path, '*', '*')) if 'Sejong' not in x])
    for folder in all_folders:
        construct_query_and_database_sets(
            folder = folder,
            save_folder = args.save_folder, 
            eval_thresh = args.eval_thresh,
            time_thresh = args.time_thresh,
            file_extension = args.file_extension
        )

