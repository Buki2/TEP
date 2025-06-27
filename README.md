# TEP

Code and dataset for the ACL 2025 paper:
**"Walk in Others' Shoes with a Single Glance: Human-Centric Visual Grounding with Top-View Perspective Transformation"**

## Introduction

We propose a **human-centric visual grounding (HVG) task**, which requires robots to interpret human-centric instructions and identify the corresponding objects from their own perspectives. To support this task:

- We construct an **InterRef dataset** to evaluate perspective-taking abilities
- We propose a **top-view enhanced perspective transformation (TEP) method**, which bridges human and robot perspectives via an intermediate top-view representation.

This repository provides the dataset and code for TEP.

## Dataset

You can access the InterRef dataset (images and annotation files) via [Google Drive](https://drive.google.com/drive/folders/1G5Xi5KDul2my9MV8E1OB6K7dsS3irSJ-?usp=sharing).

Each annotation file is a dictionary with the format:
```
{
  "sample ID": [
    "input image name",
    "ground-truth label of human position",
    "input instruction",
    [ground-truth bounding box: x, y, w, h]
  ]
}
```

## Environment Setup


1. Install GLIP
```
git clone https://github.com/sachit-menon/GLIP.git
cd GLIP
python setup.py clean --all build develop --user
```

2. Install required Python packages
```
pip install yacs timm einops ftfy dill omegaconf backoff openai numpy==1.23.5
```

3. Verify GLIP installation
```
import contextlib
import os

with contextlib.redirect_stderr(open(os.devnull, "w")):  # Do not print nltk_data messages when importing
    from maskrcnn_benchmark.engine.predictor_glip import GLIPDemo, to_image_list, create_positive_map, create_positive_map_label_to_token_from_positive_map
```

4. Download model checkpoints
```
mkdir -p ./TEP/model/glip_mebow/pretrained_models/GLIP/checkpoints
mkdir -p ./TEP/model/glip_mebow/pretrained_models/GLIP/configs
mkdir -p ./TEP/model/glip_mebow/pretrained_models/depth

wget -P ./TEP/model/glip_mebow/pretrained_models/GLIP/checkpoints https://huggingface.co/GLIPModel/GLIP/resolve/main/glip_large_model.pth
wget -P ./TEP/model/glip_mebow/pretrained_models/GLIP/configs https://raw.githubusercontent.com/microsoft/GLIP/main/configs/pretrain/glip_Swin_L.yaml
wget -P ./TEP/model/glip_mebow/pretrained_models/depth https://github.com/isl-org/MiDaS/releases/download/v3/dpt_large_384.pt
```

5. Install MEBOW. Download the trained [HBOE model](https://pennstateoffice365-my.sharepoint.com/:f:/g/personal/czw390_psu_edu/EoXLPTeNqHlCg7DgVvmRrDgB_DpkEupEUrrGATpUdvF6oQ?e=CQQ2KY), and then place it under the folder `glip_mebow/models/model_hboe.pth`
```
cd ./TEP/model/glip_mebow/MEBOW
pip install -r requirements.txt
cd cocoapi/PythonAPI
python3 setup.py install --user
```

## Inference

Run the following command to perform inference:
```
python test.py
```

## Citation

If you find our work helpful, please cite our paper:

```bibtex
@inproceedings{conf/acl25/BuXZ/tep,
  author = {Yuqi Bu and Xin Wu and Zirui Zhao and Yi Cai and David Hsu and Qiong Liu},
  title = {Walk in Others' Shoes with a Single Glance: Human-Centric Visual Grounding with Top-View Perspective Transformation},
  booktitle = {ACL},
  year = {2025},
}
```

## Acknowledgement

We sincerely thank the authors of the following repositories for generously sharing their code:
- https://github.com/sachit-menon/GLIP
- https://github.com/isl-org/MiDaS
- https://github.com/ChenyanWu/MEBOW
