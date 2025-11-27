# WSI Patch Classification Pipeline

A three-stage pipeline for training a deep learning model to detect tumor regions in whole-slide histopathology images.

---

## Pipeline Overview

```
┌─────────────────────┐      ┌────────────────────────┐      ┌─────────────────────┐
│  1. build_index.py  │ ──▶  │ 1.5 build_sampled_index│ ──▶  │  2. save_patches.py │
│                     │      │        .py             │      │                     │
│  Scans slides and   │      │  Subsamples normal     │      │  Extracts actual    │
│  annotations,       │      │  patches to create a   │      │  image patches and  │
│  identifies tissue  │      │  balanced index        │      │  stores them        │
│  regions and tumor  │      │                        │      │                     │
│  annotations        │      │                        │      │                     │
└─────────────────────┘      └────────────────────────┘      └─────────────────────┘
            │                           │                           │
            ▼                           ▼                           ▼
   index_level_N.pkl           index_level_N_sampled.pkl       patches_lmdb/
                                                              (or .h5 / PNGs)
                                                                        │
                                                                        ▼
                                                           ┌─────────────────────┐
                                                           │     3. train.py     │
                                                           │  Train classifier   │
                                                           │  & evaluate model   │
                                                           └─────────────────────┘
                                                                        │
                                                                        ▼
                                                     best.pt (model), slide_index_test.pkl

```

---


### Annotation Format

Annotations should be XML files (e.g., from ASAP annotation tool) with tumor regions marked as polygons:

```xml
<Annotation PartOfGroup="Tumor">
    <Coordinates>
        <Coordinate X="1234.5" Y="5678.9"/>
        ...
    </Coordinates>
</Annotation>
```

---

## Step 1: Build Slide Index

### Purpose

Scans all whole-slide images, detects tissue regions using Otsu thresholding, parses tumor annotations, and creates a coordinate index of valid patches.

### Script: `build_index.py`

### Inputs

| Input | Description |
|-------|-------------|
| `images/` | Directory containing `.tif` whole-slide images |
| `annotations/` | Directory containing `.xml` tumor annotations (optional) |

### Configuration Parameters

| Parameter | Description | Example |
|-----------|-------------|---------|
| `images_dir` | Path to slide images | `'images'` |
| `annotations_dir` | Path to annotation XMLs | `'annotations'` |
| `level` | Low-res level for tissue detection | `7` |
| `target_level` | Level for patch extraction | `0-6` |
| `patch_size` | Patch dimensions in pixels | `256` |
| `base_mag` | Base magnification of scanner | `40.0` |

### Usage

Edit the script parameters at the bottom, then run:

```bash
python build_index.py
```

Or modify the loop to generate indices for specific levels:

```python
for target_level in range(0, 7):
    index = build_slide_index(
        images_dir='images',
        annotations_dir='annotations',
        level=7,               # working level (low-res)
        target_level=target_level,
        patch_size=256,
        base_mag=40.0
    )
    with open(f'index_level_{target_level}.pkl', 'wb') as f:
        pickle.dump(index, f)
```

### Output

| Output | Description |
|--------|-------------|
| `index_level_N.pkl` | Pickle file containing slide index dictionary |

### Output Structure

```python
{
    'slide_id': {
        'slide_path': '/path/to/slide.tif',
        'label': 'patient001',
        'annotation_path': '/path/to/annotations.xml',  # or None
        'level': 4,                    # target extraction level
        'work_level': 7,               # tissue detection level
        'magnification': 2.5,          # effective magnification
        'visualisation_coords': [...], # (row, col, is_tumor) at work_level
        'patches': [                   # (y, x, is_tumor) at target_level
            (0, 256, False),
            (256, 512, True),
            ...
        ]
    },
    ...
}
```

---

## Step 2: Extract and Store Patches

### Purpose

Reads the slide index, extracts actual image patches from the whole-slide images at the specified coordinates, and saves them to disk in a format optimized for deep learning.

### Script: `save_patches.py`

### Inputs

| Input | Description |
|-------|-------------|
| `index_level_N.pkl` | Slide index from Step 1 |
| Original `.tif` files | Referenced by paths in the index |

### Storage Backends

| Backend | Flag | Output | Best For |
|---------|------|--------|----------|
| Filesystem | `--backend fs` | PNG files in `normal/` and `tumor/` folders | Debugging, small datasets, `ImageFolder` |
| LMDB | `--backend lmdb` | Single LMDB database | Large datasets, fast random access |
| HDF5 | `--backend hdf5` | Single `.h5` file | Moderate datasets, easy sharing |

### Usage

**Filesystem (PNG files):**
```bash
python save_patches.py \
    --index_file index_level_4.pkl \
    --patch_size 256 \
    --backend fs \
    --out_dir image_patches
```

**LMDB (recommended for training):**
```bash
python save_patches.py \
    --index_file index_level_4.pkl \
    --patch_size 256 \
    --backend lmdb \
    --lmdb_path patches_lmdb
```

**HDF5:**
```bash
python save_patches.py \
    --index_file index_level_4.pkl \
    --patch_size 256 \
    --backend hdf5 \
    --hdf5_path patches.h5
```

### Output

| Backend | Output |
|---------|--------|
| `fs` | `image_patches/normal/*.png` and `image_patches/tumor/*.png` |
| `lmdb` | `patches_lmdb/` directory (LMDB environment) |
| `hdf5` | `patches.h5` with `images` and `labels` datasets |

### Patch Naming Convention

Patches are named as `{slide_id}_{count:06d}.png` (or used as keys in LMDB/HDF5), where the count is a global incrementing index across all slides.

---

## Step 3: Train the Model

### Purpose

Trains a ResNet-50 binary classifier to distinguish tumor patches from normal tissue, using multiple techniques to handle class imbalance.

### Script: `train.py`

### Inputs

| Input | Description |
|-------|-------------|
| `index_level_N.pkl` | Slide index (for key/label reconstruction) |
| `patches_lmdb/` | LMDB database from Step 2 |

### Class Imbalance Handling

The training script uses four complementary strategies:

| Strategy | Description |
|----------|-------------|
| **Positive Augmentation** | Tumor patches duplicated `aug_factor` times with random transforms |
| **Class Weights** | Loss weighted inversely by class frequency |
| **Focal Loss** | Down-weights easy examples (γ=2.0) |
| **OHEM** | Online Hard Example Mining — backprop only on hardest negatives |

### Usage

```bash
python train.py \
    --index_path index_level_4.pkl \
    --lmdb_path patches_lmdb \
    --batch_size 64 \
    --num_workers 4 \
    --max_epochs 50 \
    --patience 2 \
    --lr 1e-3 \
    --aug_factor 5 \
    --ohem_ratio 3.0 \
    --val_size 0.1 \
    --checkpoint_dir checkpoints \
    --save_interval 5 \
    --loss_csv epoch_losses.csv
```

### Command Line Arguments

| Argument | Default | Description |
|----------|---------|-------------|
| `--index_path` | required | Path to slide index pickle |
| `--lmdb_path` | required | Path to LMDB database |
| `--batch_size` | 64 | Training batch size |
| `--num_workers` | 4 | DataLoader workers |
| `--max_epochs` | 50 | Maximum training epochs |
| `--patience` | 2 | Early stopping patience |
| `--lr` | 1e-3 | Initial learning rate |
| `--aug_factor` | 5 | Augmented copies per tumor patch |
| `--ohem_ratio` | 3.0 | Hard negatives per positive |
| `--val_size` | 0.1 | Validation set fraction |
| `--checkpoint_dir` | checkpoints | Directory for model checkpoints |
| `--save_interval` | 5 | Save checkpoint every N epochs |
| `--loss_csv` | epoch_losses.csv | Training metrics output |
| `--seed` | 42 | Random seed |

### Data Splitting

- **Test set**: Slides with `'test'` in the slide ID (e.g., `test001_node_1`)
- **Train/Val**: Remaining slides, split by `--val_size` with stratification

### Output

| Output | Description |
|--------|-------------|
| `checkpoints/best.pt` | Best model weights (lowest validation loss) |
| `checkpoints/ckpt_epoch_N.pt` | Periodic checkpoints |
| `epoch_losses.csv` | Per-epoch training and validation metrics |
| `slide_index_test.pkl` | Test slides with per-patch predictions added |

### Output CSV Format

```csv
epoch,train_loss,train_acc,val_loss,val_acc
1,0.4523,0.7821,0.3912,0.8234
2,0.3102,0.8567,0.2891,0.8612
...
```

### Test Results Structure

`slide_index_test.pkl` contains the original slide info plus:

```python
{
    'test_slide_001': {
        # ... original fields ...
        'probs': [0.12, 0.87, 0.93, ...],  # tumor probability per patch
        'preds': [0, 1, 1, ...],            # binary predictions (threshold=0.5)
    }
}
```

---

## Complete Example Workflow

```bash
# Step 1: Build index for level 4 (10x magnification at 40x base)
python build_index.py
# Produces: index_level_4.pkl

# Step 2: Extract patches to LMDB
python save_patches.py \
    --index_file index_level_4.pkl \
    --backend lmdb \
    --lmdb_path patches_lmdb
# Produces: patches_lmdb/

# Step 3: Train model
python train.py \
    --index_path index_level_4.pkl \
    --lmdb_path patches_lmdb \
    --max_epochs 50 \
    --batch_size 64
# Produces: checkpoints/best.pt, epoch_losses.csv, slide_index_test.pkl
```

---

## Tips and Notes

### Memory Considerations

- LMDB is memory-mapped, so large datasets work well
- HDF5 can be loaded entirely into RAM for small datasets
- Filesystem backend is slowest but most debuggable

### Naming Convention for Test Slides

Include `'test'` in the slide ID to automatically route to test set:
- `test001_node_1.tif` → test set
- `patient001_node_1.tif` → train/val set

---


## File Summary

| File | Purpose | Inputs | Outputs |
|------|---------|--------|---------|
| `build_index.py` | Index slides and annotations | `.tif` images, `.xml` annotations | `index_level_N.pkl` |
| `save_patches.py` | Extract and store patches | `index_level_N.pkl`, `.tif` images | LMDB / HDF5 / PNGs |
| `train.py` | Train classifier | `index_level_N.pkl`, LMDB | `best.pt`, metrics |