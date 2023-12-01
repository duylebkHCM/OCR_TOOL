import os
import json
import glob
from pathlib import Path
from PIL import Image
from rotation import rotator
from extractor import extractor, utils
from extractor.utils import order_points, to_quad_fast, to_4points, to_quad_slow
import cv2
from scipy.spatial import distance
import numpy as np
import PIL
import collections
from copy import deepcopy
from typing import Union, List
import yaml
from itertools import tee, combinations


def pairwise(iterable):
    # pairwise('ABCDEFG') --> AB BC CD DE EF FG
    a, b = tee(iterable)
    next(b, None)
    return zip(a, b)

def get_text_image(coord, image, pad_ratio=0.02):
    coord = utils.pad_points(coord, pad_ratio)
    
    tl, tr, br, bl = coord
    w = (
        distance.euclidean(np.asarray(tl), np.asarray(tr))
        + distance.euclidean(np.asarray(bl), np.asarray(br))
    ) / 2
    h = (
        distance.euclidean(np.asarray(tl), np.asarray(bl))
        + distance.euclidean(np.asarray(tr), np.asarray(br))
    ) / 2
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    M = cv2.getPerspectiveTransform(src, dst)
    text_image = cv2.warpPerspective(image, M, dsize=(int(w), int(h)))
    text_image = Image.fromarray(text_image)
    return text_image


def get_all_ext(dir: Union[str, List[str]], exclude = ['.json']):
    if isinstance(dir, list):
        exts = set()
        for d in dir:
            for f_ in os.listdir(d):
                ext = Path(f_).suffix.lower()
                if ext not in exclude:
                    exts.add(ext)
    else:
        exts = set()
        for f_ in os.listdir(d):
                ext = Path(f_).suffix.lower()
                if ext not in exclude:
                    exts.add(ext)

    exts = list(exts)
    return exts

import sys

yaml_path = sys.argv[1:]

with open(yaml_path[0], 'r') as f:
    config = yaml.safe_load(f)

rotator_obj = rotator.LabelRotator(
    ignore_wrong_text=False,
    device='cpu',
    text_ratio=5,
    hor_text_ratio=5
)

regions = list(config['regions'].keys())


extractor_obj = extractor.LabelExtractor(
                regions=regions,
                to_quad=True,
                to_crop=False,
                pad_ratio=0.02,
                is_fast=True
)

logging_path = Path(config['log_path'])
if not logging_path.exists():
    logging_path.mkdir(parents=True)
    
save_dir = Path(config['save_path'])

if not save_dir.exists():
    save_dir.mkdir(parents=True)

if config['process_only']:
    data_subset = open(config['process_only'], 'r').readlines()
    process_files = [line.strip().split('\t') for line in data_subset]
else:
    process_files = Path(config['raw_data_path']).rglob(f"{config['DIR_PREFIX']}*/*.json")
    
error_file = 'error_file.txt'
error_file = logging_path.joinpath(error_file)

error_box = 'error_box.txt'
error_box = logging_path.joinpath(error_box)

out_none_labels = 'none_value_img.txt'
out_none_labels = logging_path.joinpath(out_none_labels)

error_files = open(error_file, 'w')
error_boxs = open(error_box, 'w')
none_value_img = open(out_none_labels, 'w')

lst_dirs = list(Path(config['raw_data_path']).glob(f"{config['DIR_PREFIX']}*"))

exts = get_all_ext(lst_dirs)

print('Extension', exts)
total_textlines = 0


for f_ in process_files:
    if isinstance(f_, list):
        f_path = f_[0]
        save_field = f_[1]
        f_ = Path(f_path)
    else:
        save_field = None
        
    for ext in exts:
        img_path = f_.with_suffix(f'{ext}')
        if img_path.exists():
            img_ext = ext
            break

    img_name = img_path.name
    img_ext = img_path.suffix
    
    try:
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        error_files.write(str(f_) + '\n')
        print('Error cannot extract card region')
        continue

    with open(str(f_), 'r', encoding='utf-8') as f:
        anno = json.load(f)

    try:
        extracted_imgs, extracted_json_infos, region_names = extractor_obj.extract(img, anno)
    except Exception as e:
        error_files.write(str(f_) + '\n')
        print('Error cannot extract card region')
        continue
    
    if len(extracted_imgs) == 0 or len(extracted_json_infos) == 0:
        error_files.write(str(f_) + '\n')
        print('Error cannot extract card region')
        continue

    for extracted_img, extracted_json_info, region_name in zip(extracted_imgs, extracted_json_infos, region_names):
        new_shape = []
        exclude_field_by_region = config['regions'][region_name]
        
        for box in extracted_json_info['shapes']:
            if box['label'] in regions:
                continue
            
            if exclude_field_by_region is not None and any([box['label'].upper().startswith(field) for field in exclude_field_by_region]):
                continue
            
            points = [tuple(point) for point in box['points']]
            
            if box['shape_type'] == 'rectangle':
                points = to_4points(points)

            elif box['shape_type'] == 'polygon' and len(points) > 4:
                points = to_quad_fast(
                    points,
                    extracted_img.shape[0],
                    extracted_img.shape[1]
                )
            
            elif box['shape_type'] == 'line' and len(points)> 4:
                print(img_path)
                print(img_path.with_suffix('.json'))
                print(box['label'])
                print(points)
                collapse_point = None
                remove_point = None
                
                for (point1, point2) in combinations(points, 2):
                    if np.abs(point1[0] - point2[0]) < 3 and np.abs(point1[1]-point2[1]) < 3:
                        collapse_point = point1
                        remove_point = point2
                        break
                
                new_point = [collapse_point] + [point for idx, point in enumerate(points) if point[0]!=remove_point[0] and point[1]!=remove_point[1]]
                print(new_point)
                points = order_points(new_point)
                
            points = order_points(points)
            box['points'] = points
            new_shape.append(box)
        
        extracted_json_info.update({'shapes':new_shape})
        
        rotated_img, rotated_json_info = rotator_obj.rotate(extracted_img, extracted_json_info)
        boxes = rotated_json_info['shapes']
        labels = [box['label'] for box in boxes]
        dupplicate_labels = {item.upper(): count for item, count in collections.Counter(labels).items() if count > 1}

        
        for box in boxes:         
            if save_field and box['label'] != save_field:
                continue
            
            if config['extract_img_only']:
                label = box['label'].upper()
                label_dir = save_dir.joinpath(region_name).joinpath(box['label'].upper())
                if not label_dir.exists():
                    label_dir.mkdir(parents=True)

                img_save_path = str(label_dir.joinpath(save_img_name))
                points = [tuple(point) for point in box['points']]
                tl_img = get_text_image(points, rotated_img, pad_ratio=config['pad_info'].get(region_name, None))
                tl_img.save(img_save_path)
                
                continue
            
            value = box.get('value', None)
            if value is None:
                print(f'Error box {box["label"]} has none value')
                error_boxs.write(str(f_) + '\t' + box['label'] + '\n')
                continue
            elif value == '':
                print(f'Error box {box["label"]} has empty value')
                none_value_img.write(str(f_) + '\t' + box['label'] + '\n')
                value = "None"
                continue
            else:
                label = box['label'].upper()

                save_img_name = None
                if label in dupplicate_labels:
                    idx = deepcopy(dupplicate_labels[label])
                    save_img_name = img_name[:-4] + f'_{idx}{img_ext}'
                    dupplicate_labels[label] = idx - 1
                else:
                    save_img_name = img_name

                label_dir = save_dir.joinpath(region_name).joinpath(box['label'].upper())

                if not label_dir.exists():
                    label_dir.mkdir(parents=True)

                img_save_path = str(label_dir.joinpath(save_img_name))
                label_save_path = str(label_dir.joinpath(save_img_name).with_suffix('.txt'))

                if not os.path.exists(img_save_path):
                    points = [tuple(point) for point in box['points']]
                    tl_img = get_text_image(points, rotated_img, pad_ratio=[0.02, 0.02, 0.1, 0.2])
                    tl_img.save(img_save_path)
                
                    with open(label_save_path, 'w', encoding='utf-8') as f_value:
                        f_value.write(value)

                    total_textlines += 1
                else:
                    assert os.path.exists(label_save_path)
                    assert os.path.exists(img_save_path)

error_files.close()
error_boxs.close()
none_value_img.close()
print('Total textlines', total_textlines)
