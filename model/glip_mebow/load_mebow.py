from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os
import pprint
import shutil

import cv2
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
from tensorboardX import SummaryWriter

import model.glip_mebow.tools._init_paths
from config import cfg
from config import update_config

from core.loss import JointsMSELoss
from core.loss import DepthLoss
from core.loss import hoe_diff_loss
from core.loss import Bone_loss

from core.function import train
from core.function import validate

from model.glip_mebow.lib.utils.utils import get_optimizer
from model.glip_mebow.lib.utils.utils import save_checkpoint
from model.glip_mebow.lib.utils.utils import create_logger
from model.glip_mebow.lib.utils.utils import get_model_summary

import dataset
import model.glip_mebow.models
from PIL import Image

from tqdm import tqdm
import json

from model.glip_mebow.lib.models.pose_hrnet import PoseHighResolutionNet

class MEBOW_Args():
    def __init__(self) -> None:
        self.cfg = 'model/glip_mebow/experiments/coco/segm-4_lr1e-3.yaml'
        self.opts = []

        self.modelDir = ''
        self.logDir = ''
        self.dataDir =''
        self.prevModelDir = ''
        self.device = 'cpu'
        self.img_path = ''

def visual_mebow(image_path, person_bbox):  # xywh
    args = MEBOW_Args()
    update_config(cfg, args)

    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    model = PoseHighResolutionNet(cfg).to(args.device)
    model.eval()
    model.load_state_dict(torch.load(cfg.TEST.MODEL_FILE, map_location=torch.device(args.device)), strict=True)
    # Data loading code
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
    )
    transform = transforms.Compose([
            transforms.ToTensor(),
            normalize])

    img = cv2.imread(image_path, cv2.IMREAD_COLOR | cv2.IMREAD_IGNORE_ORIENTATION)
    # crop
    x, y, w, h = person_bbox
    img = img[y:y+h, x:x+w, :]

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (192, 256))

    input = transform(img).unsqueeze(0)
    input = input.float()

    model.eval()
    _, hoe_output = model(input)
    ori = torch.argmax(hoe_output[0]) * 5

    pred = 0
    if ori > (180 - 11.25) and ori < (180 + 11.25):
        pred = '3'
    elif ori > (157.5 - 11.25) and ori < (157.5 + 11.25):
        pred = '4'
    elif ori > (135 - 11.25) and ori < (135 + 11.25):
        pred = '5'
    elif ori > (202.5 - 11.25) and ori < (202.5 + 11.25):
        pred = '2'
    elif ori > (225 - 11.25) and ori < (225 + 11.25):
        pred = '1'
    
    elif ori > (90 - 11.25) and ori < (90 + 11.25):
        pred = '7'
    elif ori > (45 - 11.25) and ori < (45 + 11.25):
        pred = '9'
    elif ori > (112.5 - 11.25) and ori < (112.5 + 11.25):
        pred = '6'
    elif ori > (67.5 - 11.25) and ori < (67.5 + 11.25):
        pred = '8'
    elif ori > (22.5 - 11.25) and ori < (22.5 + 11.25):
        pred = '10'
    
    elif ori > (270 - 11.25) and ori < (270 + 11.25):
        pred = '14'
    elif ori > (315 - 11.25) and ori < (315 + 11.25):
        pred = '12'
    elif ori > (292.5 - 11.25) and ori < (292.5 + 11.25):
        pred = '13'
    elif ori > (247.5 - 11.25) and ori < (247.5 + 11.25):
        pred = '15'
    elif ori > (337.5 - 11.25) and ori < (337.5 + 11.25):
        pred = '11'
    
    elif ori < 11.25 and ori > 348.75:
        pred = 'R'
    
    return pred