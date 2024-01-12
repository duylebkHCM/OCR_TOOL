import itertools
import random
from typing import List, Tuple, Union

import cv2
import numpy as np
import sympy
from scipy.spatial import distance
from shapely.geometry import Point, Polygon, box


def to_crop(
    points: List[Tuple[float, float]],
    image_height: int,
    image_width: int,
    pad_ratio: Union[List[float], Tuple[float, float], float] = 0.0,
) -> List[Tuple[float, float]]:
    if isinstance(pad_ratio, tuple) or (
        isinstance(pad_ratio, list) and len(pad_ratio) == 2
    ):
        pad_ratio = random.uniform(*pad_ratio)

    x1 = min(points, key=lambda p: p[0])[0]  # x top-left
    y1 = min(points, key=lambda p: p[1])[1]  # y top-left
    x2 = max(points, key=lambda p: p[0])[0]  # x bottom-right
    y2 = max(points, key=lambda p: p[1])[1]  # y bottom-right

    ex1 = max(0, x1 - pad_ratio * (x2 - x1))
    ey1 = max(0, y1 - pad_ratio * (y2 - y1))
    ex2 = min(image_width, x2 + pad_ratio * (x2 - x1))
    ey2 = min(image_height, y2 + pad_ratio * (y2 - y1))

    box = [(ex1, ey1), (ex2, ey2)]

    return to_4points(box)


def to_quad_slow(
    points: List[Tuple[float, float]],
    pad_ratio: List[float] = [0.0, 0.0, 0.0, 0.0],  # left, right, top, bottom
) -> List[Tuple[float, float]]:
    polygon = Polygon(points)
    assert polygon.is_valid, "polygon is invalid."

    # create all possible lines from convex points
    lines = []
    for i in range(len(points)):
        line = sympy.Line2D(
            sympy.Point2D(points[i]), sympy.Point2D(points[(i + 1) % len(points)])
        )
        lines.append(line)

    # get candiate quadrangles from all lines
    candidates = []
    for line1, line2, line3, line4 in itertools.combinations(lines, 4):
        point1 = line1.intersection(line2)
        point2 = line2.intersection(line3)
        point3 = line3.intersection(line4)
        point4 = line4.intersection(line1)

        four_points = [point1, point2, point3, point4]
        if all(
            [
                (len(point) == 1 and isinstance(point[0], sympy.Point2D))
                for point in four_points
            ]
        ):
            candidate = [point[0].coordinates for point in four_points]
            if Polygon(candidate).is_valid:
                candidates.append(candidate)

    # choose candidate quadrangle which has the highest iou with polygon
    chosen_quad = max(
        candidates,
        key=lambda x: Polygon(x).intersection(polygon).area
        / Polygon(x).union(polygon).area,
    )

    chosen_quad = pad_points(points=order_points(chosen_quad), pad_ratio=pad_ratio)

    return chosen_quad


def to_quad_fast(
    points: List[Tuple[float, float]],
    image_height: int,
    image_width: int,
    pad_ratio: List[float] = [0.0, 0.0, 0.0, 0.0],  # left, right, top, bottom
) -> List[Tuple[float, float]]:
    poly = Polygon(points)
    assert poly.is_valid, "polygon is invalid."

    # get convex hull of poly, and remove end point of convex (because begin point is same end point)
    points = list(poly.convex_hull.boundary.coords)[:-1]

    # set boundary for intersection points, intersection points of two lines can not be out of boundary
    boundary = box(-image_width, -image_height, 2 * image_width, 2 * image_height)

    candidates = []
    for x, y, z, t in itertools.combinations(range(len(points)), 4):
        lines = [
            [points[x], points[(x + 1) % len(points)]],
            [points[y], points[(y + 1) % len(points)]],
            [points[z], points[(z + 1) % len(points)]],
            [points[t], points[(t + 1) % len(points)]],
        ]

        candidate = []
        for i in range(4):
            point = intersect_point(lines[i], lines[(i + 1) % 4])
            if (
                (not point)
                or (point in candidate)
                or (not boundary.contains(Point(point)))
            ):
                break
            candidate.append(point)

        if len(candidate) == 4 and Polygon(order_points(candidate)).is_valid:
            candidates.append(order_points(candidate))

    # choose candidate quadrangle which has the highest iou with polygon
    chosen_quad = max(
        candidates,
        key=lambda x: Polygon(x).intersection(poly).area / Polygon(x).union(poly).area,
    )

    chosen_quad = pad_points(points=chosen_quad, pad_ratio=pad_ratio)

    return chosen_quad


def intersect_point(
    line1: List[Tuple[float, float]], line2: List[Tuple[float, float]]
) -> Tuple[float, float]:
    a1 = line1[1][1] - line1[0][1]
    b1 = line1[0][0] - line1[1][0]
    a2 = line2[1][1] - line2[0][1]
    b2 = line2[0][0] - line2[1][0]

    determinant = a1 * b2 - a2 * b1

    if determinant == 0:
        return None

    c1 = (a1 / determinant) * line1[0][0] + (b1 / determinant) * line1[0][1]
    c2 = (a2 / determinant) * line2[0][0] + (b2 / determinant) * line2[0][1]

    x = b2 * c1 - b1 * c2
    y = a1 * c2 - a2 * c1

    return (x, y)


def to_warp(
    image: np.ndarray,
    points: List[Tuple[float, float]],
) -> Tuple[np.ndarray, np.ndarray]:
    points = [sympy.Point2D(*point) for point in points]

    avg_width = (
        float(points[0].distance(points[1])) + float(points[2].distance(points[3]))
    ) / 2
    avg_height = (
        float(points[0].distance(points[3])) + float(points[1].distance(points[2]))
    ) / 2

    rectangle = np.float32(
        [[0, 0], [avg_width, 0], [avg_width, avg_height], [0, avg_height]]
    )
    quadrangle = np.float32([point.coordinates for point in points])

    M = cv2.getPerspectiveTransform(quadrangle, rectangle)
    warp = cv2.warpPerspective(image, M, (int(avg_width), int(avg_height)))

    return warp, M


def pad_points(
    points: List[Tuple[float, float]],
    pad_ratio: Union[List[float], Tuple[float, float], float] = 0.0,
) -> List[Tuple[float, float]]:
    tl, tr, br, bl = np.array(points)  # convert list to vector

    if isinstance(pad_ratio, float):
        pad_ratio = [pad_ratio] * 4
    elif isinstance(pad_ratio, tuple):
        pad_ratio = [random.uniform(*pad_ratio)] * 4
    elif isinstance(pad_ratio, list) and len(pad_ratio) == 2:
        pad_ratio = [random.uniform(*pad_ratio)] * 4

    pad_left, pad_right, pad_top, pad_bot = pad_ratio

    # extending horizontally
    h_tl = tl - (tr - tl) * pad_left
    h_tr = tr + (tr - tl) * pad_right
    h_bl = bl - (br - bl) * pad_left
    h_br = br + (br - bl) * pad_right

    # extending vertically
    v_tl = h_tl - (h_bl - h_tl) * pad_top
    v_bl = h_bl + (h_bl - h_tl) * pad_bot
    v_tr = h_tr - (h_br - h_tr) * pad_top
    v_br = h_br + (h_br - h_tr) * pad_bot

    return [tuple(v_tl), tuple(v_tr), tuple(v_br), tuple(v_bl)]


def to_4points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    x1, y1 = points[0][0], points[0][1]
    x2, y2 = points[1][0], points[1][1]
    return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]


def to_perspective(
    points: List[Tuple[float, float]], matrix: np.ndarray
) -> List[Tuple[float, float]]:
    points = np.float32([[point[0], point[1], 1] for point in points])
    points = matrix.dot(points.T)
    points = points / points[-1, :]
    points = points[:2, :].T.tolist()
    return points


def centroid_poly(points: List[Tuple[float, float]]) -> Tuple[float, float]:
    centroid = tuple(Polygon(points).centroid.coords[0])
    return centroid


def order_points(points: List[Tuple[float, float]]) -> List[Tuple[float, float]]:
    assert len(points) == 4, "Length of points must be 4"
    tl = min(points, key=lambda p: p[0] + p[1])
    br = max(points, key=lambda p: p[0] + p[1])
    tr = max(points, key=lambda p: p[0] - p[1])
    bl = min(points, key=lambda p: p[0] - p[1])

    return [tl, tr, br, bl]
