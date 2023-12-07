import importlib
import os
import typing
from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Union

import numpy as np
from PIL.Image import Image as PILImage

from text_renderer.effect import Effects
from text_renderer.layout import Layout
from text_renderer.layout.same_line import SameLineLayout

if typing.TYPE_CHECKING:
    from text_renderer.corpus import Corpus


@dataclass
class PerspectiveTransformCfg:
    """
    Base class for PerspectiveTransform
    """

    x: float = 10
    y: float = 5
    z: float = 1.5
    scale: int = 1
    fovy: int = 50

    @abstractmethod
    def get_xyz(self) -> Tuple[int, int, int]:
        pass


@dataclass
class FixedPerspectiveTransformCfg(PerspectiveTransformCfg):
    def get_xyz(self) -> Tuple[float, float, float]:
        return 15, 15, 1.2


@dataclass
class UniformPerspectiveTransformCfg(PerspectiveTransformCfg):
    """
    x,y,z are uniform distributed
    """

    def get_xyz(self) -> Tuple[float, float, float]:
        x = np.random.uniform(-self.x, self.x)
        y = np.random.uniform(-self.y, self.y)
        z = np.random.uniform(-self.z, self.z)
        return x, y, z


@dataclass
class NormPerspectiveTransformCfg(PerspectiveTransformCfg):
    """
    x,y,z are normally distributed
    """

    def cliped_rand_norm(self, mu=0, sigma3: float = 1):
        """
        :param mu: mean
        :param sigma3: 99% (mu-3*sigma, mu+3*sigma)
        :return:
            float
        """
        # 标准差
        sigma = sigma3 / 3
        dst = sigma * np.random.randn() + mu
        dst = np.clip(dst, 0 - sigma3, sigma3)
        return dst

    def get_xyz(self) -> Tuple[float, float, float]:
        x = self.cliped_rand_norm(0, self.x)
        y = self.cliped_rand_norm(0, self.y)
        z = self.cliped_rand_norm(0, self.z)
        return x, y, z


class TextColorCfg:
    """
    Base class for TextColorCfg
    """

    @abstractmethod
    def get_color(self, bg_img: PILImage) -> Tuple[int, int, int, int]:
        pass


@dataclass
class FixedTextColorCfg(TextColorCfg):
    # For generate effect/layout example
    def get_color(self, bg_img: PILImage) -> Tuple[int, int, int, int]:
        alpha = 255
        text_color = (255, 50, 0, alpha)

        return text_color


@dataclass
class SimpleTextColorCfg(TextColorCfg):
    """
    Randomly use mean value of background image
    """

    alpha: Tuple[int, int] = (110, 255)

    def get_color(self, bg_img: PILImage) -> Tuple[int, int, int, int]:
        np_img = np.array(bg_img)

        # Calculate the luminance of the background color
        bg_luminance = np.mean(np_img)

        # Ensure a significant contrast by choosing a luminance that is far from the background
        fg_luminance = np.random.uniform(0.2, 0.8) * bg_luminance

        alpha = np.random.randint(*self.alpha)

        # Generate random RGB values with a luminance significantly different from the background
        r = int(np.random.uniform(0.2, 0.8) * fg_luminance)
        g = int(np.random.uniform(0.2, 0.8) * fg_luminance)
        b = int(np.random.uniform(0.2, 0.8) * fg_luminance)

        text_color = (r, g, b, alpha)

        return text_color


# noinspection PyUnresolvedReferences
@dataclass
class RenderCfg:
    """

    Parameters
    ----------
    corpus : Corpus
    corpus_effects : Effects
        Effects apply on text mask image of each corpus.
        Effects used at this stage must return changed bbox of text if it modified it.
    bg_dir : Path
        Background image directory
    pre_load_bg_img : bool
        True: Load all image into memory
    layout : Layout
        Layout will applied if corpus is a List
    perspective_transform : PerspectiveTransformCfg
        Apply Perspective Transform
    layout_effects : Effects
        Effects apply on merged text mask image output by Layout.
    render_effects : Effects
        Effects apply on final image.
    height : int
        Resize(keep ratio) image to height, set -1 disables resize
    gray : bool
        Save image as gray image
    text_color_cfg : TextColorCfg
        If not None, will overwrite text_color_cfg in CorpusCfg
        useful to set same text color when use multi corpus
    return_bg_and_mask: bool
    """

    corpus: "Corpus"
    corpus_effects: Effects = None
    bg_dir: Path = None
    pre_load_bg_img: bool = True
    layout: Layout = SameLineLayout()
    perspective_transform: PerspectiveTransformCfg = None
    layout_effects: Effects = None
    render_effects: Effects = None
    height: int = -1
    gray: bool = True
    text_color_cfg: TextColorCfg = None
    return_bg_and_mask: bool = False


@dataclass
class GenerateCfg:
    """
    Parameters
    ----------
    render_cfg : RenderCfg
        Configuration of Render
    """
    cfg_name: str = None
    render_cfg: RenderCfg = None


def import_module_from_file(full_path_to_module):
    """
    Import a module given the full path/filename of the .py file

    https://stackoverflow.com/questions/28836713/from-folder-name-import-variable-python-3-4-2
    """
    module = None
    try:

        # Get module name and path from full path
        module_dir, module_file = os.path.split(full_path_to_module)
        module_name, module_ext = os.path.splitext(module_file)
        # Get module "spec" from filename
        spec = importlib.util.spec_from_file_location(module_name, full_path_to_module)

        module = spec.loader.load_module()

    except Exception as ec:
        # Simple error printing
        # Insert "sophisticated" stuff here
        print(ec)

    finally:
        return module
