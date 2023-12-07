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
    GenerateCfg,
    SimpleTextColorCfg
)
import imgaug.augmenters as iaa

__all__ = [
    "dropout_rand",
    "dropout_horizontal",
    "dropout_vertical",
    "line",
    "blur",
    "gaussblur",
    "padding"
]

CURRENT_DIR = Path(os.path.abspath(os.path.dirname(__file__)))
BG_DIR = CURRENT_DIR / "bg"
TEXT_DIR = CURRENT_DIR / "text"

font_cfg = dict(
    font_dir=CURRENT_DIR / "font",
    font_size=(29, 30),
)

def base_cfg():    
    return GenerateCfg(
        render_cfg=RenderCfg(
            bg_dir=BG_DIR,
            corpus=EnumCorpus(
                EnumCorpusCfg(
                    text_paths=list(TEXT_DIR.rglob("*.txt")),
                    chars_file=CURRENT_DIR / "char" / "vocab.txt",
                    text_color_cfg=SimpleTextColorCfg(),
                    filter_by_chars=False,
                    **font_cfg,
                ),
            ),
            pre_load_bg_img=True,
            gray=False,
            height=32
        ),
    )


def dropout_rand():
    cfg = base_cfg()
    cfg.render_cfg.corpus_effects = Effects(DropoutRand(p=1, dropout_p=(0.3, 0.5)))
    cfg.cfg_name=inspect.currentframe().f_code.co_name
    return cfg


def dropout_horizontal():
    cfg = base_cfg()
    cfg.render_cfg.corpus_effects = Effects(
        DropoutHorizontal(p=1, num_line=2, thickness=3)
    )
    cfg.cfg_name=inspect.currentframe().f_code.co_name
    return cfg


def dropout_vertical():
    cfg = base_cfg()
    cfg.render_cfg.corpus_effects = Effects(DropoutVertical(p=1, num_line=15))
    cfg.cfg_name=inspect.currentframe().f_code.co_name
    return cfg


def line():
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
        cfg = base_cfg()
        cfg.render_cfg.corpus_effects = Effects(
            Line(p=1, thickness=(3, 4), line_pos_p=pos_p)
        )
        cfg.cfg_name=f'{inspect.currentframe().f_code.co_name}_{pos}'
        cfgs.append(cfg)
    return cfgs


def blur():
    cfg = base_cfg()
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
    cfg.cfg_name=inspect.currentframe().f_code.co_name
    return cfg


def gaussblur():
    cfg = base_cfg()
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
    cfg.cfg_name=inspect.currentframe().f_code.co_name
    return cfg


def padding():
    cfg = base_cfg()
    cfg.render_cfg.corpus_effects = Effects(
        [
            Padding(p=1.0, h_ratio=(0.25, 0.5))
        ]
    )
    cfg.cfg_name=inspect.currentframe().f_code.co_name
    return cfg
