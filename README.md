# MOPA: Modular Object Navigation with PointGoal Agents

This is an implementation of our paper [MOPA: Modular Object Navigation with PointGoal Agents](https://openaccess.thecvf.com/content/WACV2024/html/Raychaudhuri_MOPA_Modular_Object_Navigation_With_PointGoal_Agents_WACV_2024_paper.html). [webpage](https://3dlg-hcvc.github.io/mopa)

![](docs/images/task_viz.gif)

## Architecture Overview

![](docs/images/architecture.png)


## Installing dependencies:


This code is tested on python 3.8.13, pytorch v1.11.0 and CUDA V11.2. Install pytorch from https://pytorch.org/ according to your machine configuration.

```
conda create -n mon python=3.8 cmake=3.14.0
conda activate mon
```

This code uses forked versions of [habitat-sim](https://github.com/sonia-raychaudhuri/habitat-sim) and [habitat-lab](https://github.com/sonia-raychaudhuri/habitat-lab). 

#### Installing habitat-sim:

##### For headless machines with GPU
```
git clone git@github.com:sonia-raychaudhuri/habitat-sim.git
cd habitat-sim
python -m pip install -r requirements.txt
python setup.py build_ext --parallel 4 install --headless --bullet 
```

##### For machines with attached display
```
git clone git@github.com:sonia-raychaudhuri/habitat-sim.git
cd habitat-sim
python -m pip install -r requirements.txt
python setup.py build_ext --parallel 4 install --bullet 
```

#### Installing habitat-lab:
```
git clone git@github.com:sonia-raychaudhuri/habitat-lab.git
cd habitat-lab
pip install -e .
```

## Setup

Clone the repository and install the requirements:

```
git clone git@github.com:3dlg-hcvc/mopa.git
cd mopa
python -m pip install -r requirements.txt
```
### Downloading data and checkpoints
Download HM3D scenes [here](https://aihabitat.org/datasets/hm3d) and place the data in: `mopa/data/scene_datasets/hm3d`. 

Download objects:
```
wget -O multion_cyl_objects.zip "https://aspis.cmpt.sfu.ca/projects/multion-challenge/2022/challenge/dataset/multion_cyl_objects"
wget -O multion_real_objects.zip "https://aspis.cmpt.sfu.ca/projects/multion-challenge/2022/challenge/dataset/multion_real_objects"
```

Extract them under `mopa/data`.

Download the dataset.

```
# Replace {n} with 1, 3, 5 for 1ON, 3ON & 5ON respectively; Replace {obj_type} with CYL or REAL for Cylinder and Real/Natural objects respectively; Replace {split} with minival, val or train for different data splits.

wget -O {n}_ON_{obj_type}_{split}.zip "https://aspis.cmpt.sfu.ca/projects/multion-challenge/2022/challenge/dataset/{n}_ON_{obj_type}_{split}"
```

Extract them and place them inside `mopa/data` in the following format:

```
mopa/
  data/
    scene_datasets/
      hm3d/
          ...
    multion_cyl_objects/
        ...
    multion_real_objects/
        ...
    5_ON_CYL/
        train/
            content/
                ...
            train.json.gz
        minival/
            content/
                ...
            minival.json.gz
        val/
            content/
                ...
            val.json.gz
    5_ON_REAL/
        train/
            content/
                ...
            train.json.gz
        minival/
            content/
                ...
            minival.json.gz
        val/
            content/
                ...
            val.json.gz
```

## Usage

### Pre-trained models
Download the pretrained PointNav model, trained on HM3D [here](https://github.com/facebookresearch/habitat-matterport3d-dataset/tree/main/pointnav_comparison#pre-trained-models) and update the path [here](https://github.com/3dlg-hcvc/mopa/blob/main/baselines/config/pointnav/hier_w_proj_pred_sem_map.yaml#L12) and [here](https://github.com/3dlg-hcvc/mopa/blob/main/baselines/config/pointnav/hier_w_proj_pred_sem_map.yaml#L92).

Download the following checkpoints for Object Detection and place under mopa/data/object_detection_models:

```
wget "https://aspis.cmpt.sfu.ca/projects/multion/mopa/pretrained_models/obj_det_real.zip"
wget "https://aspis.cmpt.sfu.ca/projects/multion/mopa/pretrained_models/obj_det_cylinder.zip"
  wget "https://aspis.cmpt.sfu.ca/projects/multion/mopa/pretrained_models/knn_colors.zip"
```

### Evaluation
Evaluation will run on the `3_ON` val set by default. 

```
# For evaluating with OraSem agent on 3ON cylinders dataset
python run.py  --exp-config baselines/config/pointnav/hier_w_proj_ora_sem_map.yaml --run-type eval

# For evaluating with OraSem agent on 3ON real/natural objects dataset
python run.py  --exp-config baselines/config/pointnav/hier_w_proj_ora_sem_map_real.yaml --run-type eval

# For evaluating with PredSem agent on 3ON cylinders dataset
python run.py  --exp-config baselines/config/pointnav/hier_w_proj_pred_sem_map.yaml --run-type eval

# For evaluating with PredSem agent on 3ON real/natural objects dataset
python run.py  --exp-config baselines/config/pointnav/hier_w_proj_pred_sem_map_real.yaml --run-type eval

```

## Citation
>Sonia Raychaudhuri, Tommaso Campari, Unnat Jain, Manolis Savva, Angel X. Chang, 2023. MOPA: Modular Object Navigation with PointGoal Agents. [PDF]()

## Bibtex
```
  @misc{raychaudhuri2023mopa,
      title={MOPA: Modular Object Navigation with PointGoal Agents}, 
      author={Sonia Raychaudhuri and Tommaso Campari and Unnat Jain and Manolis Savva and Angel X. Chang},
      year={2023},
      eprint={2304.03696},
      archivePrefix={arXiv},
      primaryClass={cs.RO}
}
```

## Acknowledgements
The members at SFU were supported by Canada CIFAR AI Chair grant, Canada Research Chair grant, NSERC Discovery Grant and a research grant by Facebook AI Research. Experiments at SFU were enabled by support from WestGrid and Compute Canada. This repository is built upon [Habitat Lab](https://github.com/facebookresearch/habitat-lab) and [multiON](https://github.com/3dlg-hcvc/multiON).
