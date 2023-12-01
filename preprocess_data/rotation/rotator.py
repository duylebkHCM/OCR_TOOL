import cv2
import math
import yaml
import torch
import numpy as np
import imgaug.augmenters as iaa

from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .model import MobileNetV3Small

from imgaug.augmentables import Keypoint
from imgaug.augmentables import KeypointsOnImage


class LabelRotator:
    def __init__(
        self,
        weight_path: str = 'rotation/weight/best_model_9_loss=-0.0002.pt',
        text_ratio: int = 20,  # text_width / text_height
        text_height: int = 32,
        hor_text_ratio: float = 2.,  # text_width / text_height
        max_texts: int = 10,
        ignore_wrong_text: bool = False,
        ignore_regions: Optional[List[str]] = None,
        device: str = 'cpu',
    ) -> None:
        self.device = device
        self.max_texts = max_texts
        self.text_ratio = text_ratio
        self.text_height = text_height
        self.hor_text_ratio = hor_text_ratio

        self.ignore_regions = ignore_regions
        self.ignore_wrong_text = ignore_wrong_text

        self.ratio_croper = iaa.CropToAspectRatio(text_ratio, position='right-bottom')  # crop text image to fixed ratio
        self.ratio_padder = iaa.PadToAspectRatio(text_ratio, position='right-bottom')  # pad text image to fixed ratio

        self.model = MobileNetV3Small(num_classes=4)
        self.model.load_state_dict(state_dict=torch.load(f=weight_path, map_location='cpu'))
        self.model.eval().to(device)

    def rotate(self, image: np.ndarray, json_info: dict) -> Tuple[np.ndarray, dict]:
        angle = self.get_angle_document(image, json_info)
        rotator = iaa.Rot90(k=-int(angle / 90), keep_size=False)

        for shape in json_info['shapes']:
            keypoints = [Keypoint(x=point[0], y=point[1]) for point in shape['points']]
            kps_on_image = KeypointsOnImage(keypoints=keypoints, shape=image.shape)
            kps_on_image = rotator(keypoints=kps_on_image)
            shape['points'] = [[keypoint.x, keypoint.y] for keypoint in kps_on_image.keypoints]

        image = rotator(image=image)

        if angle % 180:
            json_info['imageWidth'], json_info['imageHeight'] = json_info['imageHeight'], json_info['imageWidth']

        return image, json_info

    def get_angle_document(self, image: np.ndarray, json_info: dict) -> int:
        '''get angle of document'''
        hor_texts, ver_texts = [], []

        for text in self.get_texts(json_info):
            points = self.order_points(text)
            text_ratio = self.compute_text_ratio(points)  # w / h

            if text_ratio > self.hor_text_ratio:
                hor_texts.append(points)
            elif 1 / (text_ratio + 1e-6) > self.hor_text_ratio:
                ver_texts.append(points)

        if (not len(hor_texts)) and (not len(ver_texts)):  # angle = 0 if image has no horizontal texts and vertical texts
            return 0

        if len(hor_texts) > len(ver_texts):
            texts = sorted(hor_texts, key=lambda text: self.distance(text[0], text[1]), reverse=True)
            angle = 0
        else:
            texts = sorted(ver_texts, key=lambda text: self.distance(text[0], text[3]), reverse=True)
            angle = 90

        if len(texts) > self.max_texts:
            texts = texts[(len(texts) - self.max_texts) // 2: (len(texts) + self.max_texts) // 2]

        text_samples = []
        for points in texts:
            w = int(round((self.distance(points[0], points[1]) + self.distance(points[3], points[2])) / 2))
            h = int(round((self.distance(points[0], points[3]) + self.distance(points[1], points[2])) / 2))

            # crop text line from original image
            M = cv2.getPerspectiveTransform(
                src=np.float32(points), dst=np.float32([[0, 0], [w, 0], [w, h], [0, h]])
            )
            text = cv2.warpPerspective(src=image, M=M, dsize=(w, h))
            # rotate text image to horizontal image if angle of text is 90
            if angle == 90:
                text = np.rot90(text, k=1)

            # crop or pad text line to required ratio
            text = text.astype(np.float32)
            if w / (h + 1e-6) > self.text_ratio:
                text = self.ratio_croper(image=text)
                text = (text - text.mean()) / text.std() if text.std() else np.zeros_like(text)
            else:
                text = (text - text.mean()) / text.std() if text.std() else np.zeros_like(text)
                text = self.ratio_padder(image=text)

            # convert to tensor
            text = cv2.resize(text, dsize=(int(round(self.text_height * self.text_ratio)), self.text_height))
            text = torch.from_numpy(np.ascontiguousarray(text)).to(self.device)
            text = text.float().permute(2, 0, 1).contiguous()

            text_samples.append(text)

        text_samples = torch.stack(text_samples, dim=0).to(self.device)  # N x 3 x H x W
        with torch.no_grad():
            preds = self.model(text_samples).squeeze(1).sigmoid()  # N

        if preds.round().sum() / (preds.numel() + 1e-6) > 0.5:  # voting (0: 0 degree, 1: 180 degree)
            angle += 180

        return angle

    def get_texts(self, json_info: dict) -> List[List[List[Tuple[float, float]]]]:
        '''get all text line region from json file'''
        texts = []
        for shape in json_info['shapes']:
            # if shape is ignored regions, will be continued to next shape
            if self.ignore_regions and (shape['label'] in self.ignore_regions):
                continue

            if shape['shape_type'] == 'polygon':
                points = shape['points']
                if len(points) != 4:  # not a 4 points polygon (not a true text line)
                    if self.ignore_wrong_text:
                        continue
                    else:
                        raise RuntimeError(f"{json_info['imagePath']} with {shape['label']} must be 4 points")

            elif shape['shape_type'] == 'rectangle':
                points = self.to_4points(shape['points'])

            else:
                continue

            texts.append(points)

        return texts

    def compute_text_ratio(self, points: List[Tuple[float, float]]) -> float:
        w = (self.distance(points[0], points[1]) + self.distance(points[3], points[2])) / 2
        h = (self.distance(points[0], points[3]) + self.distance(points[1], points[2])) / 2
        return w / (h + 1e-6)

    def order_points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        assert len(points) == 4, 'Length of points must be 4'
        ls = sorted(points, key=lambda p: p[0])[:2]  # two points at left side
        rs = sorted(points, key=lambda p: p[0])[2:]  # two points at right side
        tl, bl = sorted(ls, key=lambda p: p[1])      # top point and bottom point in left side
        tr, br = sorted(rs, key=lambda p: p[1])      # top point and bottom point in right side
        return [tl, tr, br, bl]

    def to_4points(self, points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
        x1, y1 = points[0][0], points[0][1]
        x2, y2 = points[1][0], points[1][1]
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    def distance(self, point1, point2):
        return math.sqrt((point1[0] - point2[0]) ** 2 + (point1[1] - point2[1]) ** 2)
