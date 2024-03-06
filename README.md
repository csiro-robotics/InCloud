# InCloud: Incremental Learning for Point Cloud Place Recognition
![](figures/InCloud.png?style=centerm)
This repository contains the code implementation used in the IROS2022 paper *InCloud: Incremental Learning for Point Cloud Place Recognition*. \[[arXiv](https://arxiv.org/abs/2203.00807)].  

### Abstract 
Abstract— Place recognition is a fundamental component of robotics, and has seen tremendous improvements through the use of deep learning models in recent years. Networks can experience significant drops in performance when deployed in unseen or highly dynamic environments, and require additional training on the collected data. However naively fine-tuning on new training distributions can cause severe degradation of performance on previously visited domains, a phenomenon known as catastrophic forgetting. In this paper we address the problem of incremental learning for point cloud place recognition and introduce InCloud, a structure-aware distillation-based approach which preserves the higher-order structure of the network’s embedding space. We introduce several challenging new benchmarks on four popular and large-scale LiDAR datasets (Oxford, MulRan, In-house and KITTI) showing broad improvements in point cloud place recognition performance over a variety of network architectures. To the best of our knowledge, this work is the first to effectively apply incremental learning for point cloud place recognition. Data pre-processing, training and evaluation code for this paper can be found at https://github.com/csiro-robotics/InCloud. 

### Repository Contributions
Our contributions in this repository are:

- A pre-processed MulRan dataset with ground plane removed and downsampled to 4096 points to bring in-line with the pre-processing of the Oxford and In-House datasets
- Implementations of LwF, EWC and InCloud for incremental training on three different network architectures:  MinkLoc3D, LoGG3D-Net and PointNetVLAD 
- Implementations of the two incremental learning training protocols  for point cloud place recognition introduced in our paper (*2-Step, 4-Step*)
- Evaluation scripts for the aforementioned datasets and training protocols
- Pre-trained checkpoints for key results in our paper 

If you find this repository useful for your research, please consider citing the paper

```
@inproceedings{knights2022incloud,
  title={Incloud: Incremental learning for point cloud place recognition},
  author={Knights, Joshua and Moghadam, Peyman and Ramezani, Milad and Sridharan, Sridha and Fookes, Clinton},
  booktitle={2022 IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)},
  pages={8559--8566},
  year={2022},
  organization={IEEE}
}
```

## Updates 
- 06/03/2024: Update links for downloads
- 29/07/2022: Initial Commit

## Table of Contents
- [Installation](#installation)
- [Data Preparation](#data-preparation)
 - [Getting Started](#getting-started)
- [Pretrained Models](#models)
- [Acknowledgements](#acknowledgements)

## Installation
Code was tested using Python 3.8 with PyTorch 1.9.1 and MinkowskiEngine 0.5.4 on Ubuntu 20.04 with CUDA 11.1

The following Python packages are required:
* PyTorch (version 1.9.0)
* MinkowskiEngine (version 0.5.4)
* pytorch_metric_learning (version 1.0 or above)
* torchpack
* tensorboard
* pandas


Modify the `PYTHONPATH` environment variable to include an absolute path to the project root folder: 
```
export PYTHONPATH=$PYTHONPATH:/.../.../InCloud
 ```


## Data Preparation 
<a name="data-preparation"></a>
### Oxford & In-House
Two environments in our incremental setup are the Oxford RobotCar and In-house (U.S., R.A., B.D.) datasets introduced in *PointNetVLAD: Deep Point Cloud Based Retrieval for Large-Scale Place Recognition* paper ([paper](https://arxiv.org/pdf/1804.03492)).  For dataset description see PointNetVLAD paper or github repository ([link](https://github.com/mikacuy/pointnetvlad)).

You can download training and evaluation datasets from [here](https://drive.google.com/open?id=1rflmyfZ1v9cGGH0RL4qXRrKhg-8A-U9q) ([alternative link](https://drive.google.com/file/d/1-1HA9Etw2PpZ8zHd3cjrfiZa8xzbp41J/view?usp=sharing)). 

To generate the training and testing pickles for the Oxford and In-House environments respectively, run the code below:

    python generating_queries/Oxford/generate_train.py --dataset_root <path_to_oxford_root>  --save_folder <path_to_saved_pickles>
    python generating_queries/Oxford/generate_test.py  --dataset_root <path_to_oxford_root>  --save_folder <path_to_saved_pickles>
    python generating_queries/Oxford/generate_train.py --dataset_root <path_to_inhouse_root> --save_folder <path_to_saved_pickles>
    python generating_queries/Oxford/generate_test.py  --dataset_root <path_to_inhouse_root> --save_folder <path_to_saved_pickles

`<path_to_oxford_root>` and `<path_to_inhouse_root>` are paths to the root folders for each dataset, e.g. `/data/benchmark_datasets/oxford` and `/data/benchmark_datasets/inhouse_datasets` respectively.

### MulRan 
We also employ the DCC and Riverside environments from the MulRan dataset introduced in *MulRan: Multimodal Range Dataset for Urban Place Recognition* ([paper](https://ieeexplore.ieee.org/document/9197298)).  We modify the provided scans to remove the ground plane, normalize point coordinates between -1 and 1 and downsample to 4096 points to mimic the pre-processing of the Oxford and In-House datasets.  

You can download the pre-processed DCC and Riverside datasets from [here](https://www.dropbox.com/scl/fo/f6x9j2y1r9gmhydmfbskz/h?rlkey=7tndaex2wss5h3xa44v4ll527&dl=0)

To generate the training and testing pickles for DCC and Riverside, run the following scripts:

    python generating_queries/MulRan/generate_train.py --dataset_root <path_to_mulran_root>  --save_folder <path_to_saved_pickles>
    python generating_queries/MulRan/generate_test.py  --dataset_root <path_to_mulran_root>  --save_folder <path_to_saved_pickles>

Where `<path_to_mulran_root>` is the path to the folder containing the DCC and Riverside environments.

## Getting Started 
<a name="getting-started"></a>
To get started training and evaluating with InCloud, first download and generate pickle files for the Oxford, In-House and MulRan datasets as detailed above in Data Preparation.  Then replace the paths to the test pickle files in `config/protocols/2-step.yaml` and `config/protocols/4-step.yaml` with the pickle files generated in the previous step.

### Training
An example training script for InCloud can be found in the bash file found in `scripts/train_MinkLoc3D_Incloud_4step.sh` after making the following changes:

 1. Line 9: Replace the path with the path to your conda installation
 2. Line 10: Replace the environment with the name for your conda environment
 3. Line 12: Replace with the path to your InCloud root directory
 4. Line 13: Replace with your desired save location
 5. Line 22-23: Replace paths to training pickles with the paths to the corresponding training pickles generated by following the instructions in Data Preparation 

Further changes to the training - such as changing network architecture, incremental loss, training weights, or the memory buffer - can be done by changing the appropriate value in the configuration file or input arguments.  See `training/train_incremental.py` for a list of input arguments, and the config files in the `config` folder for a list of adjustable config parameters.

### Evaluation
To evaluate InCloud run the following command:

    python eval/evaluate.py --config config/protocols/<config> --ckpt <path_to_ckpt>
   
   Where `<config>` and `<path_to_ckpt>` are the config file for the evaluation method you wish to evaluate and the path to the checkpoint you wish to evaluate. 


## Pretrained Models
<a name="models"></a>
The following models from the paper are provided for evaluation purposes:

|Architecture  | Protocol | Recall@1 | Link | 
|--|--|--|--|
| MinkLoc3D | 4-Step  | 83.6 | [Link](https://www.dropbox.com/scl/fi/o4mvbmmogj89mskpzlr1j/4-Step-MinkLoc.pth?rlkey=5bkrib5bxxhw1xeihrw60x40l&dl=0) |
| MinkLoc3D | 2-Step  | 87.7 | [Link](https://www.dropbox.com/scl/fi/kxsbr93buxqwmgk6e4ux3/2-Step-MinkLoc.pth?rlkey=jqtukrk4u3gtzbyqq0h69jid5&dl=0) |
| LoGG3D-Net | 4-Step  | 66.0 | [Link](https://www.dropbox.com/scl/fi/8667d7arrkhewm3d3bncn/4-Step-LoGG3D.pth?rlkey=9g1697gtnzetdet3rbx8ez7gj&dl=0) |
| LoGG3D-Net | 2-Step  | 73.9 | [Link](https://www.dropbox.com/scl/fi/lnh33xre7j4q1xhxqn7g9/2-Step-LoGG3D.pth?rlkey=r97h9n8j5sbck76d25v2o7ixr&dl=0) |
| PointNetVLAD | 4-Step  | 56.1 | [Link](https://www.dropbox.com/scl/fi/215wo1z18rwbeqa69sifl/4-Step-VLAD.pth?rlkey=tm9vbc7l72yppmpfvgwd07n1z&dl=0) |
| PointNetVLAD | 2-Step  | 59.7 | [Link](https://www.dropbox.com/scl/fi/l615snsoizhbc1oh9eok5/2-Step-VLAD.pth?rlkey=j4qq0vgb1krcqgivrp8p392d3&dl=0) |

## Acknowledgements
We would like to acknowledge the authors of [MinkLoc3D](https://github.com/jac99/MinkLoc3D) for their excellent codebase which has been used as a starting point for this project.  We would also like to thank the authors of [Avalanche](https://github.com/ContinualAI/avalanche) for their implementation of incremental learning approaches LwF and EWC, and the authors of [PointNetVlad-Pytorch](https://github.com/cattaneod/PointNetVlad-Pytorch) and [LoGG3D-NET](https://github.com/csiro-robotics/LoGG3D-Net) for their implementations of these network backbones.
