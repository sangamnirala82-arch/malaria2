"""
Advanced Augmentation Techniques for Malaria Detection

This package contains optional advanced augmentation methods:
- MixUp: Mixes images and labels
- CutMix: Cuts and pastes image patches
- Albumentations: 70+ advanced transforms
"""

from .mixup import (
    create_mixup_dataset,
    mixup_augmentation,
    get_mixup_info
)

from .cutmix import (
    create_cutmix_dataset,
    cutmix_augmentation,
    get_cutmix_info
)

from .albumentations_transforms import (
    create_albumentations_dataset,
    get_medical_transforms,
    get_aggressive_transforms,
    get_albumentations_info
)

__all__ = [
    # MixUp
    'create_mixup_dataset',
    'mixup_augmentation',
    'get_mixup_info',
    
    # CutMix
    'create_cutmix_dataset',
    'cutmix_augmentation',
    'get_cutmix_info',
    
    # Albumentations
    'create_albumentations_dataset',
    'get_medical_transforms',
    'get_aggressive_transforms',
    'get_albumentations_info',
]


def print_available_augmentations():
    """Print information about all available augmentation techniques."""
    print("\n" + "="*70)
    print("Available Advanced Augmentation Techniques")
    print("="*70)
    
    # MixUp
    mixup_info = get_mixup_info()
    print(f"\n1. {mixup_info['name']}")
    print(f"   {mixup_info['description']}")
    print(f"   Available: {'✅' if mixup_info.get('requires_tfp', True) else '⚠️  (Install tensorflow-probability)'}")
    
    # CutMix
    cutmix_info = get_cutmix_info()
    print(f"\n2. {cutmix_info['name']}")
    print(f"   {cutmix_info['description']}")
    print(f"   Available: {'✅' if cutmix_info.get('requires_tfp', True) else '⚠️  (Install tensorflow-probability)'}")
    
    # Albumentations
    albu_info = get_albumentations_info()
    print(f"\n3. {albu_info['name']}")
    print(f"   {albu_info['description']}")
    print(f"   Available: {'✅' if albu_info['available'] else '⚠️  (Install albumentations)'}")
    
    print("\n" + "="*70)
    print("To enable: Edit config.py and set USE_ADVANCED_AUGMENTATION = True")
    print("="*70 + "\n")


if __name__ == "__main__":
    print_available_augmentations()
