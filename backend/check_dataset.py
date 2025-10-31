"""
Script to check where TensorFlow Datasets stores the malaria dataset
Run this to find the dataset location on your local PC
"""

import os
import tensorflow_datasets as tfds

def check_dataset_location():
    print("=" * 60)
    print("TensorFlow Datasets - Malaria Dataset Location Checker")
    print("=" * 60)
    
    # Default TensorFlow Datasets directory
    data_dir = os.path.expanduser('~/tensorflow_datasets')
    
    print(f"\n📁 Default TensorFlow Datasets Directory:")
    print(f"   {data_dir}")
    
    # Check if directory exists
    if os.path.exists(data_dir):
        print(f"   ✓ Directory exists")
        
        # List all datasets
        datasets = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]
        if datasets:
            print(f"\n📦 Downloaded Datasets ({len(datasets)}):")
            for dataset in sorted(datasets):
                dataset_path = os.path.join(data_dir, dataset)
                try:
                    # Get size using os.path.getsize for all files
                    total_size = 0
                    for dirpath, dirnames, filenames in os.walk(dataset_path):
                        for filename in filenames:
                            filepath = os.path.join(dirpath, filename)
                            total_size += os.path.getsize(filepath)
                    
                    # Convert to MB
                    size_mb = total_size / (1024 * 1024)
                    print(f"   - {dataset}: {size_mb:.1f} MB")
                except Exception as e:
                    print(f"   - {dataset}: (size unknown)")
        else:
            print(f"\n   No datasets found yet")
    else:
        print(f"   ✗ Directory does not exist yet")
        print(f"   (Will be created on first dataset download)")
    
    # Check specifically for malaria dataset
    print(f"\n🔬 Malaria Dataset:")
    malaria_path = os.path.join(data_dir, 'malaria')
    
    if os.path.exists(malaria_path):
        print(f"   ✓ Found at: {malaria_path}")
        
        # Get detailed info
        version_path = os.path.join(malaria_path, '1.0.0')
        if os.path.exists(version_path):
            files = os.listdir(version_path)
            print(f"   ✓ Version: 1.0.0")
            print(f"   ✓ Files: {len(files)}")
            
            # Check for tfrecord files
            tfrecord_files = [f for f in files if f.endswith('.tfrecord')]
            if tfrecord_files:
                print(f"   ✓ TFRecord shards: {len(tfrecord_files)}")
            
            # Calculate total size
            total_size = 0
            for dirpath, dirnames, filenames in os.walk(malaria_path):
                for filename in filenames:
                    filepath = os.path.join(dirpath, filename)
                    total_size += os.path.getsize(filepath)
            
            size_mb = total_size / (1024 * 1024)
            print(f"   ✓ Total size: {size_mb:.1f} MB")
            
        print(f"\n   ✅ Dataset is ready to use!")
        
    else:
        print(f"   ✗ Not found")
        print(f"   ℹ️  Will be downloaded automatically when you run train.py")
        print(f"   ℹ️  Download size: ~337 MB")
        print(f"   ℹ️  Processed size: ~319 MB")
    
    # Check downloads directory
    downloads_path = os.path.join(data_dir, 'downloads')
    if os.path.exists(downloads_path):
        malaria_download = os.path.join(downloads_path, 'malaria')
        if os.path.exists(malaria_download):
            print(f"\n💾 Downloaded Files:")
            print(f"   ✓ Found at: {malaria_download}")
    
    print("\n" + "=" * 60)
    print("Summary:")
    print("=" * 60)
    
    if os.path.exists(malaria_path):
        print("✅ Malaria dataset is downloaded and ready")
        print("✅ No action needed - your training script will use it automatically")
    else:
        print("ℹ️  Malaria dataset not found locally")
        print("ℹ️  It will be automatically downloaded when you run:")
        print("   python train.py")
        print("ℹ️  Download time: ~2-5 minutes (depending on internet speed)")
    
    print("\n💡 Tip: This cache location means you only download once,")
    print("   and all your TensorFlow projects can use the same dataset!")
    print("=" * 60)

if __name__ == "__main__":
    check_dataset_location()
