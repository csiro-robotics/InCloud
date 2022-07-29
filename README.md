# InCloud: Incremental Learning for Point Cloud Place Recognition
![InCloud](./media/overview.jpg")
This repository contains the code implementation used in the IROS2022 paper *InCloud: Incremental Learning for Point Cloud Place Recognition*. \[[arXiv](https://arxiv.org/abs/2203.00807)].  Our contributions in this repository are:

- A pre-processed MulRan dataset with ground plane removed and downsampled to 4096 points to bring in-line with the pre-processing of the Oxford and In-House datasets
- Implementations of LwF, EWC and InCloud for incremental training on three different network architectures (MinkLoc3D, LoGG3D-Net, PointNetVLAD) and two different training protocols introduced in our paper (*2-Step, 4-Step*)
- Pre-trained checkpoints for key results in our paper 

If you find this repository useful for your research, please consider citing the paper

```
@inproceedings{vid2022logg3d,
  title={InCloud: Incremental Learning for Point Cloud Place Recognition},
  author={Knights, Joshua and Moghadam, Peyman and Ramezani, Milad and Sridharan, Sridha and Fookes, Clinton},
  booktitle={2022 IEEE International Conference on Intelligent Robots and Systems (IROS)},
  year={2022},
  eprint={arXiv preprint arXiv:2203.00807}
}
```

## Updates 
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


Modify the `PYTHONPATH` environment variable to include absolute path to the project root folder: 
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
We also employ the DCC and Riverside environments from the MulRan dataset introduced in *MulRan: Multimodal Range Dataset for Urban Place Recognition* ([paper](https://ieeexplore.ieee.org/document/9197298)).  We modify the provided scans to remove the ground plane, normalize point co-ordinates between -1 and 1 and downsample to 4096 points to mimic the pre-processing of the Oxford and In-House datasets.  

You can download the pre-processed DCC and Riverside datasets from [here](https://cloudstor.aarnet.edu.au/plus/s/6fLYRjl3QjCjRHJ)

To generate the training and testing pickles for DCC and Riverside, run the following scripts:

    python generating_queries/MulRan/generate_train.py --dataset_root <path_to_mulran_root>  --save_folder <path_to_saved_pickles>
    python generating_queries/MulRan/generate_test.py  --dataset_root <path_to_mulran_root>  --save_folder <path_to_saved_pickles>
Where `<path_to_mulran_root>` is the path to the folder containing the DCC and Riverside environments.

## Getting Started 
<a name="getting-started"></a>
To get started training and evaluating with InCloud, first download and generate pickle files for the Oxford, In-House and MulRan datasets as detailed above in Data Preparation.  Then replace the paths to the test pickle files in `config/protocols/2-step.yaml` and `config/protocols/4-step.yaml` with the pickle files generated in the previous step.

### Training
To train InCloud on MinkLoc3D, use the bash file found in `scripts/train_MinkLoc3D_Incloud.sh` after making the following changes:

 1. Line 9: Replace the path with the path to your conda installation
 2. Line 10: Replace the environment with the name for your conda environment
 3. Line 12: Replace with the path to your InCloud root directory
 4. Line 13: Replace with your desired save location
 5. Line 22-23: Replace paths to training pickles with the paths to the corresponding training pickles generated by following the instructions in Data Preparation 

### Evaluation
To evaluate InCloud run the following command:

    python eval/evaluate.py --config config/protocols/<config> --ckpt <path_to_ckpt>
   
   Where `<config>` and `<path_to_ckpt>` are the config file for the evaluation method you wish to evaluate and the path to the checkpoint you wish to evaluate. 


## Pretrained Models
<a name="models"></a>
The following models from the paper are provided for evaluation purposes:

|Architecture  | Protocol | Recall@1 | Link | 
|--|--|--|--|
| MinkLoc3D | 4-Step  | 83.6 | [Link](https://cloudstor.aarnet.edu.au/plus/s/pfy3G8IWM6zHKDm) |
| MinkLoc3D | 2-Step  | 87.7 | [Link](https://cloudstor.aarnet.edu.au/plus/s/O3wT94juNCGQsfd) |
| LoGG3D-Net | 4-Step  | 66.0 | [Link](https://cloudstor.aarnet.edu.au/plus/s/45sgreIQqCJ223r) |
| LoGG3D-Net | 2-Step  | 73.9 | [Link](https://cloudstor.aarnet.edu.au/plus/s/tRCzUUcSUWmQk7C) |
| PointNetVLAD | 4-Step  | 56.1 | [Link](https://cloudstor.aarnet.edu.au/plus/s/JLaaqUMaMlju1R7) |
| PointNetVLAD | 2-Step  | 59.7 | [Link](https://cloudstor.aarnet.edu.au/plus/s/pWZoZN6YQntXWza) |

## Acknowledgements
We would like to acknowledge the authors of [MinkLoc3D](https://github.com/jac99/MinkLoc3D) for their excellent codebase which has been used as a starting point for this project.  We would also like to thank the authors of [Avalanche](https://github.com/ContinualAI/avalanche) for their implementation of incremental learning approaches LwF and EWC, and the authors of [PointNetVlad-Pytorch](https://github.com/cattaneod/PointNetVlad-Pytorch) and [LoGG3D-NET]() for their implementations of these network backbones.
