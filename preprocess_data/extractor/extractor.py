import copy
from pathlib import Path
from typing import Dict, List, Tuple, Union

import numpy as np
from shapely.geometry import Point, Polygon

from .utils import (
    centroid_poly,
    to_4points,
    to_crop,
    to_perspective,
    to_quad_fast,
    to_quad_slow,
    to_warp,
)


class LabelExtractor:
    def __init__(
        self,
        regions: List[str],
        pad_ratio: Union[List[float], float] = [
            0.0,
            0.0,
            0.0,
            0.0,
        ],  # left, right, top, bottom
        is_fast: bool = True,
        to_quad: bool = True,
        to_crop: bool = False,
    ) -> None:
        self.to_quad = to_quad
        self.to_crop = to_crop
        self.is_fast = is_fast
        self.regions = regions
        self.pad_ratio = pad_ratio

    def extract(
        self, image: np.ndarray, json_info: Dict
    ) -> Tuple[List[np.ndarray], List[Dict]]:
        warped_json_infos = []

        (
            warped_images,
            warped_persps,
            warped_locs,
            region_names,
        ) = self.get_warped_regions(image, json_info)

        for idx, (warped_image, warped_persp, warped_loc) in enumerate(
            zip(warped_images, warped_persps, warped_locs)
        ):
            warped_json_info = {}
            warped_json_info["version"] = json_info.get("version", "4.5.7")
            warped_json_info["flags"] = json_info.get("flags", {})
            warped_json_info[
                "imagePath"
            ] = f"{Path(json_info['imagePath']).stem}_{idx}{Path(json_info['imagePath']).suffix}"
            warped_json_info["imageData"] = json_info["imageData"]
            warped_json_info["imageHeight"] = warped_image.shape[0]
            warped_json_info["imageWidth"] = warped_image.shape[1]
            warped_json_info["lineColor"] = json_info["lineColor"]
            warped_json_info["fillColor"] = json_info["fillColor"]
            warped_json_info["shapes"] = []

            warped_loc = Polygon(warped_loc).buffer(0)
            for shape in json_info["shapes"]:
                if shape["shape_type"] == "rectangle":
                    points = to_4points(shape["points"])
                    shape_type = "polygon"
                    center_point = centroid_poly(points)
                elif shape["shape_type"] == "polygon":
                    points = [tuple(point) for point in shape["points"]]
                    shape_type = shape["shape_type"]
                    center_point = centroid_poly(points)
                elif shape["shape_type"] == "line":
                    points = [tuple(point) for point in shape["points"]]
                    shape_type = shape["shape_type"]
                    center_point = (
                        (points[0][0] + points[1][0]) / 2,
                        (points[0][1] + points[1][1]) / 2,
                    )
                else:
                    continue

                # change coordinate of all component inside regions by warp perspective matrix
                points = np.int32(to_perspective(points, warped_persp))

                add_warp_condition = warped_loc.contains(Point(center_point))
                polygon_additional_condition = (
                    shape["shape_type"] == "polygon"
                    and len(points) >= 4
                    and add_warp_condition
                    and warped_loc.area > Polygon(points).area
                )

                if polygon_additional_condition or add_warp_condition:
                    warped_shape = copy.deepcopy(shape)
                    warped_shape["points"] = points.tolist()
                    warped_shape["shape_type"] = shape_type
                    warped_json_info["shapes"].append(warped_shape)

                # add card region to label
                if (shape["label"] in self.regions) and (
                    shape["label"]
                    not in [
                        warped_shape["label"]
                        for warped_shape in warped_json_info["shapes"]
                    ]
                ):
                    location = Polygon(points.tolist()).buffer(0)
                    if (
                        warped_loc.intersection(location).area
                        / warped_loc.union(location).area
                        > 0.9
                    ):
                        warped_shape = copy.deepcopy(shape)
                        warped_shape["points"] = points.tolist()
                        warped_shape["shape_type"] = shape_type
                        warped_json_info["shapes"].append(warped_shape)

            warped_json_infos.append(warped_json_info)

        return warped_images, warped_json_infos, region_names

    def get_warped_regions(
        self, image: np.ndarray, json_info: Dict
    ) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        warped_images, warped_persps, warped_locs, region_names = [], [], [], []

        height, width = json_info["imageHeight"], json_info["imageWidth"]

        for info in json_info["shapes"]:
            region_name, region_coords, region_type = (
                info["label"],
                info["points"],
                info["shape_type"],
            )
            if region_type == "rectangle":
                region_coords = to_4points(region_coords)

            if (region_name in self.regions) and (len(region_coords) >= 4):
                if self.to_quad:
                    if self.is_fast:
                        warped_coords = to_quad_fast(
                            points=region_coords,
                            image_height=height,
                            image_width=width,
                            pad_ratio=self.pad_ratio,
                        )
                    else:
                        warped_coords = to_quad_slow(
                            points=region_coords, pad_ratio=self.pad_ratio
                        )

                elif self.to_crop:
                    warped_coords = to_crop(
                        points=region_coords,
                        image_height=height,
                        image_width=width,
                        pad_ratio=self.pad_ratio,
                    )
                else:
                    raise ValueError(
                        "Must be choose warp method is to_quad or to_crop."
                    )

                if warped_coords is not None:
                    warped_image, warped_persp = to_warp(
                        image=image, points=warped_coords
                    )

                    region_names.append(region_name)
                    warped_images.append(warped_image)
                    warped_persps.append(warped_persp)
                    warped_locs.append(warped_coords)

        return warped_images, warped_persps, warped_locs, region_names
