# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This software may be used and distributed in accordance with
# the terms of the DINOv3 License Agreement.

import logging
from typing import Sequence, Optional, Tuple

import torch
from torchvision.transforms import v2

import cv2
import math
import numpy as np
from PIL import Image


logger = logging.getLogger("dinov3")


def make_interpolation_mode(mode_str: str) -> v2.InterpolationMode:
    return {mode.value: mode for mode in v2.InterpolationMode}[mode_str]


class GaussianBlur(v2.RandomApply):
    """
    Apply Gaussian Blur to the PIL image.
    """

    def __init__(self, *, p: float = 0.5, radius_min: float = 0.1, radius_max: float = 2.0):
        # NOTE: torchvision is applying 1 - probability to return the original image
        keep_p = 1 - p
        transform = v2.GaussianBlur(kernel_size=9, sigma=(radius_min, radius_max))
        super().__init__(transforms=[transform], p=keep_p)


# Use timm's names
IMAGENET_DEFAULT_MEAN = (0.485, 0.456, 0.406)
IMAGENET_DEFAULT_STD = (0.229, 0.224, 0.225)

TOUTIAO_DEFAULT_MEAN = (0.0, 0.0, 0.0)
TOUTIAO_DEFAULT_STD = (0.0, 0.0, 0.0)

CROP_DEFAULT_SIZE = 224
RESIZE_DEFAULT_SIZE = int(256 * CROP_DEFAULT_SIZE / 224)


def make_normalize_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Normalize(mean=mean, std=std)


def make_base_transform(
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Normalize:
    return v2.Compose(
        [
            v2.ToDtype(torch.float32, scale=True),
            make_normalize_transform(mean=mean, std=std),
        ]
    )


# This roughly matches torchvision's preset for classification training:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L6-L44
def make_classification_train_transform(
    *,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    hflip_prob: float = 0.5,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
):
    transforms_list = [v2.ToImage(), v2.RandomResizedCrop(crop_size, interpolation=interpolation)]
    if hflip_prob > 0.0:
        transforms_list.append(v2.RandomHorizontalFlip(hflip_prob))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built classification train transform\n{transform}")
    return transform


def make_resize_transform(
    *,
    resize_size: int,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
):
    assert not (resize_square and resize_large_side), "These two options can not be set together"
    if resize_square:
        logger.info("resizing image as a square")
        size = (resize_size, resize_size)
        transform = v2.Resize(size=size, interpolation=interpolation)
        return transform
    elif resize_large_side:
        logger.info("resizing based on large side")
        transform = v2.Resize(size=None, max_size=resize_size, interpolation=interpolation)
        return transform
    else:
        transform = v2.Resize(resize_size, interpolation=interpolation)
        return transform


# Derived from make_classification_eval_transform() with more control over resize and crop
def make_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    resize_square: bool = False,
    resize_large_side: bool = False,  # Set the larger side to resize_size instead of the smaller
    interpolation: v2.InterpolationMode = v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    transforms_list = [v2.ToImage()]
    resize_transform = make_resize_transform(
        resize_size=resize_size,
        resize_square=resize_square,
        resize_large_side=resize_large_side,
        interpolation=interpolation,
    )
    transforms_list.append(resize_transform)
    if crop_size:
        transforms_list.append(v2.CenterCrop(crop_size))
    transforms_list.append(make_base_transform(mean, std))
    transform = v2.Compose(transforms_list)
    logger.info(f"Built eval transform\n{transform}")
    return transform


# This matches (roughly) torchvision's preset for classification evaluation:
#   https://github.com/pytorch/vision/blob/main/references/classification/presets.py#L47-L69
def make_classification_eval_transform(
    *,
    resize_size: int = RESIZE_DEFAULT_SIZE,
    crop_size: int = CROP_DEFAULT_SIZE,
    interpolation=v2.InterpolationMode.BICUBIC,
    mean: Sequence[float] = IMAGENET_DEFAULT_MEAN,
    std: Sequence[float] = IMAGENET_DEFAULT_STD,
) -> v2.Compose:
    return make_eval_transform(
        resize_size=resize_size,
        crop_size=crop_size,
        interpolation=interpolation,
        mean=mean,
        std=std,
        resize_square=False,
        resize_large_side=False,
    )


def voc2007_classification_target_transform(label, n_categories=20):
    one_hot = torch.zeros(n_categories, dtype=int)
    for instance in label.instances:
        one_hot[instance.category_id] = True
    return one_hot


def imaterialist_classification_target_transform(label, n_categories=294):
    one_hot = torch.zeros(n_categories, dtype=int)
    one_hot[label.attributes] = True
    return one_hot


def get_target_transform(dataset_str):
    if "VOC2007" in dataset_str:
        return voc2007_classification_target_transform
    elif "IMaterialist" in dataset_str:
        return imaterialist_classification_target_transform
    return None


class ResizeShortEdge:
    def __init__(self, short_edge=256, align=16, resample=Image.BICUBIC):
        self.short_edge = short_edge
        self.align = align
        self.resample = resample
    
    def __call__(self, img: Image.Image) -> Image.Image:
        resize_size = self.detect_size(img.size)
        return img.resize(resize_size, self.resample)
    
    def detect_size(self, size: Tuple[int, int]) -> Tuple[int, int]:
        w, h = size
        
        if h < w:
            new_h = self.short_edge
            new_w = math.ceil(w * self.short_edge / h / self.align) * self.align
        else:
            new_w = self.short_edge
            new_h = math.ceil(h * self.short_edge / w / self.align) * self.align
        
        return (new_w, new_h)


class BorderCropper:
    """基于梯度检测 + 黑色验证的图像黑边裁剪器"""

    def __init__(
        self,
        search_ratio: float = 0.5,
        gradient_scale: float = 1.0,
        coverage_thresh: float = 0.8,
        black_thresh: int = 25,
        none_black_ratio_thresh: float = 0.05,
    ):
        self.search_ratio = search_ratio
        self.gradient_scale = gradient_scale
        self.coverage_thresh = coverage_thresh
        self.black_thresh = black_thresh
        self.none_black_ratio_thresh = none_black_ratio_thresh

    def __call__(self, image: Image.Image) -> Image.Image:
        bbox = self.detect_bbox(image)
        if bbox is None:
            return image
        
        return image.crop(bbox)

    def detect_bbox(self, image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
        gray = np.asarray(image.convert("L"))
        h, w = gray.shape
        area = h * w

        row_coverage, col_coverage = self._compute_coverage(gray)
        h_margin, w_margin = int(h * self.search_ratio), int(w * self.search_ratio)
        black_mask = gray <= self.black_thresh

        row_black_count = black_mask.sum(axis=1)
        col_black_count = black_mask.sum(axis=0)

        top_cumsum = row_black_count[:h_margin].cumsum()
        bottom_cumsum = row_black_count[-h_margin:][::-1].cumsum()
        left_cumsum = col_black_count[:w_margin].cumsum()
        right_cumsum = col_black_count[-w_margin:][::-1].cumsum()

        top = self._detect_boundary(row_coverage[:h_margin], top_cumsum, area, w)
        bottom = h - self._detect_boundary(row_coverage[-h_margin:][::-1], bottom_cumsum, area, w)
        left = self._detect_boundary(col_coverage[:w_margin], left_cumsum, area, h)
        right = w - self._detect_boundary(col_coverage[-w_margin:][::-1], right_cumsum, area, h)

        if top >= bottom or left >= right:
            return None
        
        return (left, top, right, bottom) # (x1, y1, x2, y2)

    def _compute_coverage(self, gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
        grad = np.hypot(gx, gy)

        positive_grad = grad[grad > 0]
        if positive_grad.size == 0:
            return np.zeros(gray.shape[0]), np.zeros(gray.shape[1])

        grad_thresh = positive_grad.mean() * self.gradient_scale
        high_grad_mask = grad > grad_thresh
        return high_grad_mask.mean(axis=1), high_grad_mask.mean(axis=0)

    def _detect_boundary(self, coverage: np.ndarray, cumsum: np.ndarray, area: int, axis_len: int) -> int:
        mask = np.where(coverage > self.coverage_thresh)[0]
        
        if len(mask) == 0:
            return 0

        none_black_ratios = (mask * axis_len - cumsum[mask]) / area
        final_mask = none_black_ratios <= self.none_black_ratio_thresh

        return mask[final_mask].max() if final_mask.any() else 0
