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

# For Oxford
P1 = [5735712.768124, 620084.402381]
P2 = [5735611.299219, 620540.270327]
P3 = [5735237.358209, 620543.094379]
P4 = [5734749.303802, 619932.693364]
index_list = index_list = [5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 22, 24, 31, 32, 33, 38, 39, 43, 44]

FILENAME = "pointcloud_locations_20m.csv"
POINTCLOUD_FOLS = "pointcloud_20m"

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
    folders = []
    for idx, folder in enumerate(sorted(os.listdir(os.path.join(base_path)))):
        if idx in index_list:
            folders.append(folder)
    construct_query_and_database_sets(base_path, folders, args.save_folder, args.file_extension, [P1,P2,P3,P4], 'Oxford')
   