import os
import datetime
from model.TEP import grounding_steps
import cv2
import json

class Args:
    def __init__(self) -> None:
        self.prompt_path = './prompt/'

        self.llm_grounding = 'gpt-4'
        self.llm_top_person = 'gpt-4-vision-preview'
        self.llm_top_object = 'gpt-4-vision-preview'
        self.llm_top_reason = 'gpt-4-vision-preview'
        self.llm_temperature = 0
        self.llm_max_tokens = 1024

        self.flag_write_cache = True
        cache_path = './cache/' + datetime.datetime.now().strftime('%y%m%d_%H%M') + '/'
        self.cache_path = cache_path
        self.flag_execute_code = True

def image_preprocess(image_path):
    max_size = 512
    img = cv2.imread(image_path)
    img_h, img_w, _ = img.shape
    if img_w >= img_h and img_w > max_size:
        resize_w = max_size
        resize_h = int(max_size * img_h / img_w)
        new_img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    elif img_w < img_h and img_h > max_size:
        resize_h = max_size
        resize_w = int(max_size * img_w / img_h)
        new_img = cv2.resize(img, (resize_w, resize_h), interpolation=cv2.INTER_AREA)
    else:
        return image_path, img_w, img_h
    new_image_path = image_path[:-4] + '_resiezed.jpg'
    cv2.imwrite(new_image_path, new_img)
    return new_image_path, resize_w, resize_h

def xywh2xyxy_norm(box, img_w, img_h):
    x, y, w, h =box
    xmin = x / img_w
    ymin = y / img_h
    xmax = (x + w) / img_w
    ymax = (y + h) / img_h
    return [xmin, ymin, xmax, ymax]

def tep_method(image_path, text):

    args = Args()
    if args.flag_write_cache:
        if not os.path.exists(args.cache_path):
            os.makedirs(args.cache_path)

    new_image_path, img_w, img_h = image_preprocess(image_path)

    try:
        final_answer = grounding_steps(args, text, new_image_path)
        final_answer_norm = xywh2xyxy_norm(final_answer, img_w, img_h)
    except Exception as e:
        print(e)
    
    if final_answer_norm:
        record = {}
        record['img'] = image_path
        record['text'] = text
        record['box'] = final_answer_norm
        with open(args.cache_path + 'record.json', 'w') as f:
            json.dump(record, f)
    
        return final_answer_norm
    else:
        return [0, 0, 0, 0]