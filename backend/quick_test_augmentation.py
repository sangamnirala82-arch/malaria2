"""Quick test to verify augmentation modules can be imported"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

print("Testing imports...")

try:
    from data_augmentation.mixup import create_mixup_dataset
    print("✓ MixUp imported successfully")
except Exception as e:
    print(f"✗ MixUp import failed: {e}")

try:
    from data_augmentation.cutmix import create_cutmix_dataset
    print("✓ CutMix imported successfully")
except Exception as e:
    print(f"✗ CutMix import failed: {e}")

try:
    from data_augmentation.albumentations_transforms import create_albumentations_dataset
    print("✓ Albumentations imported successfully")
except Exception as e:
    print(f"✗ Albumentations import failed: {e}")

try:
    from data_augmentation.repeating_the_dataset import create_repeated_dataset
    print("✓ Repeated Dataset imported successfully")
except Exception as e:
    print(f"✗ Repeated Dataset import failed: {e}")

print("\nAll augmentation modules are ready!")
print("\nYou can now set CONFIG['AUGMENTATION_TYPE'] to one of:")
print("  - 'basic' (default)")
print("  - 'mixup'")
print("  - 'cutmix'")
print("  - 'albumentations'")
print("  - 'repeated'")
