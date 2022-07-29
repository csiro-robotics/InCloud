# Author: Jacek Komorowski
# Warsaw University of Technology

import numpy as np
import torch
from torch.utils.data import DataLoader
import MinkowskiEngine as ME

from datasets.oxford import OxfordDataset, TrainTransform, TrainSetTransform
from datasets.samplers import BatchSampler
import random 

from torchpack.utils.config import configs 

from torchsparse import SparseTensor
from torchsparse.utils.quantize import sparse_quantize
from torchsparse.utils.collate import sparse_collate

def make_sparse_tensor(lidar_pc, voxel_size=0.05, return_points=False):
    # get rounded coordinates
    lidar_pc = lidar_pc.numpy()
    lidar_pc = np.hstack((lidar_pc, np.zeros((len(lidar_pc),1), dtype=np.float32)))
    coords = np.round(lidar_pc[:, :3] / voxel_size)
    coords -= coords.min(0, keepdims=1)
    feats = lidar_pc

    # sparse quantization: filter out duplicate points
    _, indices = sparse_quantize(coords, return_index=True)
    coords = coords[indices]
    feats = feats[indices]

    # construct the sparse tensor
    inputs = SparseTensor(feats, coords)
    # inputs = sparse_collate([inputs])
    # inputs.C = inputs.C.int()
    if return_points:
        return inputs , feats
    else:
        return inputs


def sparcify_and_collate_list(list_data, voxel_size):
    outputs = []
    for xyzr in list_data:
        outputs.append(make_sparse_tensor(xyzr, voxel_size))
    outputs =  sparse_collate(outputs)
    outputs.C = outputs.C.int()
    return outputs




def make_dataset(pickle_file):
    # Create training and validation datasets

    datasets = {}
    train_transform = TrainTransform(configs.data.aug_mode)
    train_set_transform = TrainSetTransform(configs.data.aug_mode)

    print(f'Creating Dataset from pickle file : {pickle_file}')
    dataset = OxfordDataset(configs.data.dataset_folder, pickle_file, train_transform,
                                      set_transform=train_set_transform)
  
    return dataset


def make_collate_fn(dataset: OxfordDataset, mink_quantization_size=None):
    # set_transform: the transform to be applied to all batch elements
    def collate_fn(data_list):
        # Constructs a batch object
        clouds = [e[0] for e in data_list]
        labels = [e[1] for e in data_list]
        batch = torch.stack(clouds, dim=0)       # Produces (batch_size, n_points, 3) tensor
        if dataset.set_transform is not None:
            # Apply the same transformation on all dataset elements
            batch = dataset.set_transform(batch)

        if mink_quantization_size is None:
            # Not a MinkowskiEngine based model
            batch = {'cloud': batch}
        elif configs.model.name == 'logg3d':
            batch = sparcify_and_collate_list(clouds, mink_quantization_size)
        else:
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=mink_quantization_size)
                    for e in batch]
            coords = ME.utils.batched_coordinates(coords)
            # Assign a dummy feature equal to 1 to each point
            # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
            feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
            batch = {'coords': coords, 'features': feats, 'cloud': batch}

        # Compute positives and negatives mask
        # Compute positives and negatives mask
        positives_mask = [[in_sorted_array(e, dataset.queries[label].positives) for e in labels] for label in labels]
        negatives_mask = [[not in_sorted_array(e, dataset.queries[label].non_negatives) for e in labels] for label in labels]
        positives_mask = torch.tensor(positives_mask)
        negatives_mask = torch.tensor(negatives_mask)

        # Returns (batch_size, n_points, 3) tensor and positives_mask and
        # negatives_mask which are batch_size x batch_size boolean tensors
        return batch, positives_mask, negatives_mask

    return collate_fn


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def make_dataloader(pickle_file, memory):
    """
    Create training and validation dataloaders that return groups of k=2 similar elements

    :return:
    """

    dataset = make_dataset(pickle_file)
    dataset.add_memory(memory)

    train_sampler = BatchSampler(dataset, batch_size=configs.train.batch_size,
                                 batch_size_limit=configs.train.batch_size_limit,
                                 batch_expansion_rate=configs.train.batch_expansion_rate)

    # Reproducibility
    g = torch.Generator()
    g.manual_seed(0)

    # Collate function collates items into a batch and applies a 'set transform' on the entire batch
    train_collate_fn = make_collate_fn(dataset,  configs.model.mink_quantization_size)
    dataloader = DataLoader(dataset, batch_sampler=train_sampler, collate_fn=train_collate_fn,
                                     num_workers=configs.train.num_workers, pin_memory=configs.data.pin_memory,
                                     worker_init_fn = seed_worker, generator = g)

    return dataloader


def in_sorted_array(e: int, array: np.ndarray) -> bool:
    pos = np.searchsorted(array, e)
    if pos == len(array) or pos == -1:
        return False
    else:
        return array[pos] == e
