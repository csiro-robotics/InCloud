import numpy as np
import os
import pandas as pd
from sklearn.neighbors import KDTree
import pickle
import argparse
from tqdm import tqdm 
from datasets.oxford import TrainingTuple
# For training and test data splits
X_WIDTH = 150
Y_WIDTH = 150

# For University Sector
P5 = [363621.292362, 142864.19756]
P6 = [364788.795462, 143125.746609]
P7 = [363597.507711, 144011.414174]

# For Residential Area
P8 = [360895.486453, 144999.915143]
P9 = [362357.024536, 144894.825301]
P10 = [361368.907155, 145209.663042]

test_regions = {
    'university': [P5,P6,P7],
    'residential': [P8,P9,P10],
    'business': []
}

FILENAME = "pointcloud_centroids_25.csv"
POINTCLOUD_FOLS = "pointcloud_25m_25"

def construct_query_and_database_sets(base_path, folders, save_folder, file_extension, p, output_name):
    database_trees = []
    test_trees = []
    for folder in folders:
        df_database = pd.DataFrame(columns=['file', 'northing', 'easting'])
        df_test = pd.DataFrame(columns=['file', 'northing', 'easting'])

        df_locations = pd.read_csv(os.path.join(base_path, folder, FILENAME), sep=',')
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name == "business":
                df_test = df_test.append(row, ignore_index=True)
            elif check_in_test_set(row['northing'], row['easting'], p):
                df_test = df_test.append(row, ignore_index=True)
            df_database = df_database.append(row, ignore_index=True)

        database_tree = KDTree(df_database[['northing', 'easting']])
        test_tree = KDTree(df_test[['northing', 'easting']])
        database_trees.append(database_tree)
        test_trees.append(test_tree)

    test_sets = []
    database_sets = []
    for folder in folders:
        database = {}
        test = {}
        df_locations = pd.read_csv(os.path.join(base_path, folder, FILENAME), sep=',')
        df_locations['timestamp'] = base_path + '/' + folder + '/' + POINTCLOUD_FOLS + \
                                    '/' + df_locations['timestamp'].astype(str) + file_extension
        df_locations = df_locations.rename(columns={'timestamp': 'file'})
        for index, row in df_locations.iterrows():
            # entire business district is in the test set
            if output_name == "business":
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            elif check_in_test_set(row['northing'], row['easting'], p):
                test[len(test.keys())] = {'query': row['file'], 'northing': row['northing'], 'easting': row['easting']}
            database[len(database.keys())] = {'query': row['file'], 'northing': row['northing'],
                                              'easting': row['easting']}
        database_sets.append(database)
        test_sets.append(test)

    for i in range(len(database_sets)):
        tree = database_trees[i]
        for j in range(len(test_sets)):
            if i == j:
                continue
            for key in range(len(test_sets[j].keys())):
                coor = np.array([[test_sets[j][key]["northing"], test_sets[j][key]["easting"]]])
                index = tree.query_radius(coor, r=25)
                # indices of the positive matches in database i of each query (key) in test set j
                test_sets[j][key][i] = index[0].tolist()

    output_to_file(database_sets, save_folder, f'{output_name}_evaluation_database.pickle')
    output_to_file(test_sets, save_folder, f'{output_name}_evaluation_query.pickle')


def check_in_test_set(northing, easting, points):
    in_test_set = False
    for point in points:
        if point[0] - X_WIDTH < northing < point[0] + X_WIDTH and point[1] - Y_WIDTH < easting < point[1] + Y_WIDTH:
            in_test_set = True
            break
    return in_test_set

def output_to_file(output, save_folder, filename):
    file_path = os.path.join(save_folder, filename)
    with open(file_path, 'wb') as handle:
        pickle.dump(output, handle, protocol=pickle.HIGHEST_PROTOCOL)
    print("Done ", filename)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate evaluation datasets')
    parser = argparse.ArgumentParser(description='Generate Inhouse Training Dataset')
    parser.add_argument('--dataset_root', type=str, required=True, help='Dataset root folder')
    parser.add_argument('--eval_thresh', type = int, default = 25, help = 'Threshold for positive examples')
    parser.add_argument('--file_extension', type = str, default = '.bin', help = 'File extension expected')
    parser.add_argument('--save_folder', type = str, required = True, help = 'Folder to save pickle files to')
    args = parser.parse_args()

    # Check dataset root exists, make save dir if doesn't exist
    print('Dataset root: {}'.format(args.dataset_root))
    assert os.path.exists(args.dataset_root), f"Cannot access dataset root folder: {args.dataset_root}"
    base_path = args.dataset_root
    if not os.path.exists(args.save_folder):
        os.makedirs(args.save_folder)

    # Select runs used for evaluation
    for run in ['business', 'residential', 'university']:
        folders = sorted([x for x in os.listdir(args.dataset_root) if run in x])
        p = test_regions[run]
        construct_query_and_database_sets(base_path, folders, args.save_folder, args.file_extension, p, run)