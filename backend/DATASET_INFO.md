# Malaria Dataset Information

## 📍 Dataset Location

**The dataset is NOT stored in your project folder.** TensorFlow Datasets automatically stores data in a system cache directory to avoid re-downloading for multiple projects.

### Current Storage Locations:

```
Dataset Location (on this system):
1. Processed Dataset: ~/tensorflow_datasets/malaria/ (319 MB)
2. Downloaded Files: ~/tensorflow_datasets/downloads/malaria/ (338 MB)
Total Size: ~657 MB
```

### On Your Local PC (When you run it):

**Windows:**
```
C:\Users\<YourUsername>\tensorflow_datasets\malaria\
```

**Mac/Linux:**
```
~/tensorflow_datasets/malaria/
or
/home/<username>/tensorflow_datasets/malaria/
```

## 📊 Dataset Details

- **Total Images**: 27,558 cell images
- **Classes**: 
  - Parasitized (infected with malaria): 13,779 images
  - Uninfected (healthy): 13,779 images
- **Format**: TFRecord (TensorFlow's optimized format)
- **Split**: Single 'train' split (you split it 80/10/10 in the code)
- **Image Type**: Thin blood smear cell images
- **Version**: 1.0.0

## 🔍 Dataset Structure

```
~/tensorflow_datasets/malaria/1.0.0/
├── dataset_info.json                    # Dataset metadata
├── features.json                        # Feature definitions
├── label.labels.txt                     # Class labels
├── malaria-train.tfrecord-00000-of-00004   # Data shard 1 (80 MB)
├── malaria-train.tfrecord-00001-of-00004   # Data shard 2 (80 MB)
├── malaria-train.tfrecord-00002-of-00004   # Data shard 3 (80 MB)
└── malaria-train.tfrecord-00003-of-00004   # Data shard 4 (80 MB)
```

## 📥 How Dataset Download Works

When you run `python train.py` on your local PC:

1. **First Time**: 
   - Downloads dataset (~337 MB) from NIH website
   - Extracts and converts to TFRecord format
   - Stores in `~/tensorflow_datasets/malaria/`
   - Takes ~2-5 minutes depending on internet speed

2. **Subsequent Runs**:
   - Automatically detects existing dataset
   - No re-download needed
   - Loads instantly from cache

## 🎯 Why Not in Project Folder?

TensorFlow Datasets uses a **global cache** for several good reasons:

✅ **Avoids Duplication** - One copy shared across all your ML projects
✅ **Saves Disk Space** - No need to store 657 MB per project
✅ **Version Management** - TensorFlow handles dataset versions automatically
✅ **Faster Development** - Switch between projects without re-downloading

## 📋 Dataset Classes

```
0: Parasitized (infected with malaria parasite)
1: Uninfected (healthy cell)
```

## 🔬 Dataset Source

- **Provider**: National Library of Medicine (NLM), NIH
- **Paper**: "Pre-trained convolutional neural networks as feature extractors toward improved malaria parasite detection"
- **Authors**: Rajaraman et al., 2018
- **Journal**: PeerJ
- **URL**: https://lhncbc.nlm.nih.gov/publication/pub9932

## 💡 Checking Dataset Location on Your PC

Run this Python command to find where the dataset is stored:

```python
import tensorflow_datasets as tfds
import os

# Find dataset location
data_dir = os.path.expanduser('~/tensorflow_datasets')
print(f"Dataset directory: {data_dir}")

# Check if malaria dataset exists
malaria_path = os.path.join(data_dir, 'malaria')
if os.path.exists(malaria_path):
    print(f"Malaria dataset found at: {malaria_path}")
    # Get size
    import subprocess
    size = subprocess.check_output(['du', '-sh', malaria_path]).split()[0].decode('utf-8')
    print(f"Dataset size: {size}")
else:
    print("Malaria dataset not yet downloaded")
```

## 🗑️ Deleting Dataset (if needed)

If you want to free up space or re-download:

**Windows:**
```powershell
Remove-Item -Recurse -Force "$env:USERPROFILE\tensorflow_datasets\malaria"
```

**Mac/Linux:**
```bash
rm -rf ~/tensorflow_datasets/malaria
```

The dataset will automatically re-download next time you run training.

## 📦 Alternative: Copy Dataset to Project (Optional)

If you want the dataset IN your project folder for portability:

```bash
# Create data directory in project
mkdir -p /app/backend/data

# Copy dataset
cp -r ~/tensorflow_datasets/malaria /app/backend/data/

# Then modify data_loader.py to use local path:
# tfds.load('malaria', data_dir='/app/backend/data', ...)
```

**Note**: This is NOT recommended because:
- Takes up more space
- Makes your project folder large
- Defeats the purpose of TensorFlow Datasets caching

## ✅ Summary

- ✅ Dataset is downloaded and ready (657 MB total)
- ✅ Stored in TensorFlow's cache directory (standard practice)
- ✅ No action needed - your code will find it automatically
- ✅ On your local PC, it will download to `~/tensorflow_datasets/` on first run
- ✅ Subsequent runs will use the cached version

The dataset is working perfectly! You don't need to move or copy anything. 🎉
