# All Effect/Layout example config
# 1. Run effect_layout_example.py, generate images in effect_layout_image
# 2. Update README.md
import inspect
import os
from pathlib import Path
from typing import List

from text_renderer.effect import *
from text_renderer.corpus import *
from text_renderer.config import (
    RenderCfg,
    GenerateCfgV2,
    SimpleTextColorCfg,
    FixedPerspectiveTransformCfg,
)
from text_renderer.effect.curve import Curve
import imgaug.augmenters as iaa

__all__ = ['dropout_rand', 'dropout_horizontal', 'dropout_vertical', 'line']

# __all__ = ['blur', 'gaussblur']

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
BG_DIR = CURRENT_DIR / "bg_new"

font_cfg = dict(
    font_dir=CURRENT_DIR / "font",
    font_size=(29, 30),
)

def base_cfg(text_paths: List[Path], name=None):
    assert len(text_paths) == 1
    
    return GenerateCfgV2(
        num_image=1,
        save_image_name=os.path.basename(text_paths[0])[:-4]  + name,
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            corpus=EnumCorpus(
                EnumCorpusCfg(
                    text_paths=text_paths,
                    chars_file=CURRENT_DIR / "char" / "mz_vocab.txt",
                    text_color_cfg=SimpleTextColorCfg(),
                    **font_cfg,
                ),
            ),
            pre_load_bg_img=True,
            gray=False,
            height=32
        ),
    )


def dropout_rand(text_paths: List[Path]):
    cfg = base_cfg(text_paths, inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutRand(p=1, dropout_p=(0.3, 0.5)))
    return cfg


def dropout_horizontal(text_paths):
    cfg = base_cfg(text_paths, inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        DropoutHorizontal(p=1, num_line=2, thickness=3)
    )
    return cfg


def dropout_vertical(text_paths):
    cfg = base_cfg(text_paths, inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(DropoutVertical(p=1, num_line=15))
    return cfg


def line(text_paths):
    poses = [
        "top",
        "bottom",
        "left",
        "right",
        "top_left",
        "top_right",
        "bottom_left",
        "bottom_right",
        "horizontal_middle",
        "vertical_middle",
    ]
    cfgs = []
    skip_idx = [8, 9]
    for i, pos in enumerate(poses):
        pos_p = [0] * len(poses)
        if i not in skip_idx:
            pos_p[i] = 1
        else:
            continue
        cfg = base_cfg(text_paths, f"{inspect.currentframe().f_code.co_name}_{pos}")
        cfg.render_cfg.corpus_effects = Effects(
            Line(p=1, thickness=(3, 4), line_pos_p=pos_p)
        )
        cfgs.append(cfg)
    return cfgs


def emboss(text_paths):

    cfg = base_cfg(text_paths, inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        [
            ImgAugEffect(aug=iaa.Emboss(alpha=(0.9, 1.0), strength=(1.5, 1.6))),
        ]
    )
    return cfg


def blur(text_paths):

    cfg = base_cfg(text_paths, inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        [
            ImgAugEffect(
                aug=iaa.Sequential([
                    iaa.Sometimes(
                        1.0,
                        iaa.MotionBlur()
                    )
                ])
            ),
        ]
    )
    return cfg

def gaussblur(text_paths):
    cfg = base_cfg(text_paths, inspect.currentframe().f_code.co_name)
    cfg.render_cfg.corpus_effects = Effects(
        [
            ImgAugEffect(
                aug=iaa.Sequential([
                    iaa.Sometimes(
                        1.0,
                        iaa.GaussianBlur(sigma=(0,3))
                    )
                ])
            ),
        ]
    )
    return cfg