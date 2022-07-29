import os 
import torch 
import torchsparse
import numpy as np 
from tqdm import tqdm 
import torch.nn.functional as F
import MinkowskiEngine as ME
from torch.utils.data import DataLoader
from torchpack.utils.config import configs 

class EvalDataset:
    def __init__(self, dataset_dict):
        self.set = dataset_dict 
        self.n_points = 4096 
        self.root = configs.data.dataset_folder


    def load_pc(self, filename):
        if not os.path.exists(filename):
            filename = os.path.join(self.root, filename)
        if '.bin' in filename:
            pc = np.fromfile(filename, dtype = np.float64)
            # coords are within -1..1 range in each dimension
            assert pc.shape[0] == self.n_points * 3, "Error in point cloud shape: {}".format(filename)
            pc = np.reshape(pc, (pc.shape[0] // 3, 3))
            pc = torch.tensor(pc, dtype=torch.float)
        elif '.npy' in filename:
            try:
                pc = np.load(filename)[:,:3]
                assert pc.shape[0] == self.n_points, "Error in point cloud shape: {}".format(filename)
                pc = torch.tensor(pc, dtype = torch.float)
            except:
                print(filename)
                pc = np.load(filename)[:,:3]
                assert pc.shape[0] == self.n_points, "Error in point cloud shape: {}".format(filename)
                pc = torch.tensor(pc, dtype = torch.float)
        return pc 

    def __len__(self):
        return len(self.set)

    def __getitem__(self, idx):
        pc = self.load_pc(self.set[idx]['query'])
        return pc 

def get_eval_dataloader(dataset_dict):
    dataset = EvalDataset(dataset_dict)

    def collate_fn(data_list):
        clouds = [e for e in data_list]
        labels = [e for e in data_list]
        batch = torch.stack(clouds, dim = 0)

        if configs.model.mink_quantization_size is None:
            # Not a MinkowskiEngine based model
            batch = {'cloud': batch}
        else:
            coords = [ME.utils.sparse_quantize(coordinates=e, quantization_size=configs.model.mink_quantization_size)
                    for e in batch]
            if configs.model.name == 'logg3d':
                batch = torchsparse.sparse_collate(coords)
                batch.C = batch.C.int()
            else:
                coords = ME.utils.batched_coordinates(coords)
                # Assign a dummy feature equal to 1 to each point
                # Coords must be on CPU, features can be on GPU - see MinkowskiEngine documentation
                feats = torch.ones((coords.shape[0], 1), dtype=torch.float32)
                batch = {'coords': coords, 'features': feats, 'cloud': batch}

        return batch

    dataloader = DataLoader(
        dataset,
        batch_size = configs.eval.batch_size,
        shuffle = False, 
        collate_fn = collate_fn,
        num_workers = configs.train.num_workers
    )

    return dataloader 

@torch.no_grad()
def get_latent_vectors(model, dataset_dict):
    dataloader = get_eval_dataloader(dataset_dict)
    
    model.eval()
    embeddings_list = []

    for idx, batch in enumerate(dataloader):
        batch = {k:v.to('cuda') for k,v in batch.items()}
        embeddings = model(batch)
        embeddings_list += list(embeddings.cpu().numpy())
        
    embeddings_stack = np.vstack(embeddings_list)
    return embeddings_stack
        
