# Author: Jacek Komorowski
# Warsaw University of Technology

import models.minkloc as minkloc
from models.PointNetVlad import PointNetVlad
from torchpack.utils.config import configs 


def model_factory(ckpt = None, device = 'cuda'):
    in_channels = 1

    if 'MinkFPN' in configs.model.name:
        model = minkloc.MinkLoc(
            configs.model.name, 
            in_channels=in_channels,
            feature_size=configs.model.feature_size,
            output_dim=configs.model.output_dim, 
            planes=configs.model.planes,
            layers=configs.model.layers, 
            num_top_down=configs.model.num_top_down,
            conv0_kernel_size=configs.model.conv0_kernel_size)
    elif configs.model.name == 'PointNetVlad':
        model = PointNetVlad(
            num_points = configs.data.num_points,
            global_feat = True,
            feature_transform = True,
            max_pool = False,
            output_dim = configs.model.output_dim)
    elif configs.model.name == 'LoGG3D':
        raise NotImplementedError('LoGG3D not currently implemented; will be implemented in a future commit')
    else:
        raise NotImplementedError('Model not implemented: {}'.format(configs.model.name))
    if ckpt != None:
        model.load_state_dict(ckpt)
    model = model.to(device)

    return model
