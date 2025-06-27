import re
import os
os.environ['CONFIG_NAMES'] = 'my_config'

import tensorflow as tf
from PIL import Image
from torchvision import transforms
import torch
import requests
from model.glip_mebow.vision_processes import forward
from model.glip_mebow.configs import config
from IPython.display import display

def load_image(path):
    if path.startswith("http://") or path.startswith("https://"):
        image = Image.open(requests.get(path, stream=True).raw).convert('RGB')
        image = transforms.ToTensor()(image)
    else:
        image = Image.open(path).convert('RGB')
        image = transforms.ToTensor()(image)
    return image

def object_detection_wo_margin(object_category, im=None, threshold_detection=0.5, threshold_prior=0):
    # if object_name in ["object", "objects"]:
    #     # all_object_coordinates = forward('maskrcnn', image)[0]
    # else:
    global image
    area_ori = image.shape[2] * image.shape[1]
    if im is None:
        im = image

    if object_category == 'person':
        object_category = 'people'  # GLIP does better at people than person

    all_object_coordinates = forward('glip', im, object_category, confidence_threshold=threshold_detection).tolist()

    iw = im.shape[2]
    ih = im.shape[1]
    area = iw * ih
    new_all_object_coordinates = []
    for o in all_object_coordinates: # GLIP output is [left, lower, right, upper]
        x = max(0, o[0])
        y = max(0, ih - o[3])
        w = min(iw-x, o[2] - o[0])
        h = min(ih-y, o[3] - o[1])
        if threshold_prior > 0 and (((w * h) / area_ori) < threshold_prior):
            continue
        new_all_object_coordinates.append([x,y,w,h])
    
    if is_print:
        print('object {} found {}'.format(object_category, len(new_all_object_coordinates)))

    return new_all_object_coordinates

def compute_depth_single(object_bbox):
    global image
    depth_map = forward('depth', image)
    depth = depth_map[object_bbox[1]:object_bbox[1]+object_bbox[3],object_bbox[0]:object_bbox[0]+object_bbox[2]]
    # if depth.min().item() < 0:
    #     print(torch.topk(depth, k=3))
    #     # v, idx = torch.sort(depth)
    #     # print(v[:3])
    #     # print(v[-3:])
    return [depth.min().item(), depth.max().item(), depth.median().item()]

def visual_glip(code, image_path):
    match_text = re.findall(r'find_object\((.*?)\)', code)
    obj_list = []
    obj_list.extend(['person', 'table'])
    for i in match_text:
        if i.startswith('name'):
            i_match_text = re.findall(r'\"(.*?)\"', i)
            if i_match_text:
                obj_list.extend(i_match_text)
            else:
                i_match_text = re.findall(r'\'(.*?)\'', i)
                if i_match_text:
                    obj_list.extend(i_match_text)
        else:
            try:
                obj_list.append(eval(i))
            except Exception as e:
                print(id)
                print(e)
                try:
                    i_match_text = re.findall(r'\"(.*?)\"', i)
                    if i_match_text:
                        obj_list.extend(i_match_text)
                    else:
                        i_match_text = re.findall(r'\'(.*?)\'', i)
                        if i_match_text:
                            obj_list.extend(i_match_text)
                except Exception as e:
                    print(id)
                    print(e)
    # return obj_list

    ## GLIP
    global is_print
    is_print = False

    global image
    image = load_image(image_path)
    detection_results = []
    for obj_name in obj_list:
        if isinstance(obj_name, tuple) or isinstance(obj_name, list):  # has attribute_names
            if obj_name[1]:
                cate_name = obj_name[1] + ' ' + obj_name[0]
            else:
                cate_name = obj_name[0]
        else:
            cate_name = obj_name
        obj_box = object_detection_wo_margin(cate_name, threshold_detection=0.5)
        for item in obj_box:
            obj = []
            obj.append(cate_name)
            obj.append(item)
            obj.append(compute_depth_single(item))
            detection_results.append(obj)
    
    return detection_results