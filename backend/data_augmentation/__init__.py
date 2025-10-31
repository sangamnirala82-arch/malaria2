"""Data Augmentation Package for Malaria Detection

This package contains various augmentation techniques:
- Albumentations-based transforms
- MixUp augmentation
- CutMix augmentation
- Dataset repetition with variations
"""

from .albumentations_transforms import (
    get_albumentations_transforms,
    apply_albumentations_augmentation,
    create_albumentations_dataset
)
from .mixup import (
    mixup_augmentation,
    create_mixup_dataset
)
from .cutmix import (
    cutmix_augmentation,
    create_cutmix_dataset
)
from .repeating_the_dataset import (
    create_repeated_dataset,
    augment_variations
)

__all__ = [
    'get_albumentations_transforms',
    'apply_albumentations_augmentation',
    'create_albumentations_dataset',
    'mixup_augmentation',
    'create_mixup_dataset',
    'cutmix_augmentation',
    'create_cutmix_dataset',
    'create_repeated_dataset',
    'augment_variations'
]
