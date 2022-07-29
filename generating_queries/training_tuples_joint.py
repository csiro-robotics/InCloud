import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm 
from datasets.oxford import TrainingTuple
# Import test set boundaries
from generating_queries.Oxford.generate_test import P1, P2, P3, P4, check_in_test_set
from generating_queries.Inhouse.generate_test import P5, P6, P7, P8, P9, P10

from generating_queries.Oxford.generate_train import construct_query_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate Baseline training dataset')
    parser.add_argument('--oxford_root', type=str, required=True, help='Oxford root folder')
    parser.add_argument('--mulran_root', type=str, required=True, help='Oxford root folder')
    parser.add_argument('--inhouse_root', type=str, required=True, help='Oxford root folder')
    parser.add_argument('--pos_thresh', type = int, default = 10, help = 'Threshold for positive examples')
    parser.add_argument('--neg_thresh', type = int, default = 50, help = 'Threshold for negative examples')
    parser.add_argument('--file_extension', type = str, default = '.bin', help = 'File extension expected')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save pickle files to')
    args = parser.parse_args()

    # Make save dir if doesn't exist
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Make pandas dataframe
    df_train = pd.DataFrame(columns = ['file', 'northing', 'easting'])

    # Do Oxford
    base_path = args.oxford_root 
    file_extension = '.bin'

    P = [P1, P2, P3, P4]
    FILENAME = "pointcloud_locations_20m_10overlap.csv"
    POINTCLOUD_FOLS = "pointcloud_20m_10overlap"

    folders = sorted(os.listdir(base_path))
    for folder in tqdm(folders, desc = 'Oxford Folders'):
        df_locations = pd.read_csv(os.path.join(base_path, folder, FILENAME), sep=',')
        df_locations['timestamp'] = base_path + '/' + folder + '/' + POINTCLOUD_FOLS + '/' + df_locations['timestamp'].astype(str) + file_extension
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                pass # df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)

    # Do MulRan
    base_path = args.mulran_root
    file_extension = '.npy'

    FILENAME = "pd_northing_easting.csv"
    POINTCLOUD_FOLS = "Ouster"
    ENVS = ['DCC','Riverside']
    RUNS = ['01']

    for ENV in ENVS:
        folders = [f'{ENV}_{RUN}' for RUN in RUNS]
        for folder in folders:
            df_locations = pd.read_csv(os.path.join(base_path, ENV, folder, FILENAME), sep = ',')
            df_locations['timestamp'] = base_path + '/' + ENV + '/' + folder + '/' + POINTCLOUD_FOLS + '/' + df_locations['timestamp'].astype(str) + file_extension
            df_locations = df_locations.rename(columns = {'timestamp': 'file'})
            df_train = pd.concat([df_train, df_locations], ignore_index = True)

    # Do In-House
    base_path = args.inhouse_root
    file_extension = '.bin'

    P = [P5, P6, P7, P8, P9, P10]
    FILENAME = "pointcloud_centroids_10.csv"
    POINTCLOUD_FOLS = "pointcloud_25m_10"

    folders = sorted([x for x in os.listdir(base_path) if 'business' not in x])

    for folder in tqdm(folders):
        df_locations = pd.read_csv(os.path.join(base_path, folder, FILENAME), sep=',')
        df_locations['timestamp'] = base_path + '/' + folder + '/' + POINTCLOUD_FOLS + '/' + df_locations['timestamp'].astype(str) + file_extension
        df_locations = df_locations.rename(columns={'timestamp': 'file'})

        for index, row in df_locations.iterrows():
            if check_in_test_set(row['northing'], row['easting'], P):
                pass #df_test = df_test.append(row, ignore_index=True)
            else:
                df_train = df_train.append(row, ignore_index=True)

    print("Number of training submaps: " + str(len(df_train['file'])))
    construct_query_dict(df_train, args.save_folder, "joint_train_queries.pickle", args.pos_thresh, args.neg_thresh)