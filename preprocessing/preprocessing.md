# Preprocessing Pipeline

This module prepares the dataset for the image colorization model.

## Overview

RGB images are converted to the CIELab color space:

- `L` → luminance (model input)
- `a,b` → color channels

The project uses K-Means clustering on `(a,b)` pixels to convert color prediction into a classification problem.

The model learns:

```text
L → color class
```

instead of direct RGB regression.

---

## Pipeline

### 1. Image preprocessing

Each image is:
- loaded
- resized
- converted from RGB to Lab

The normalized luminance channel is saved as input data.

---

### 2. K-Means color quantization

Random `(a,b)` pixels from the dataset are collected and clustered using MiniBatchKMeans.

Cluster centroids become discrete color classes.

---

### 3. Dataset generation

For every image:

#### Input (`X`)
Luminance channel:

```python
shape = (H, W)
dtype = float32
```

#### Target (`Y`)
Per-pixel class labels:

```python
shape = (H, W)
dtype = uint16
```

Each pixel contains the index of the nearest color centroid.

---

## Directory Structure

```text
dataset_processed/
│
├── L_luminance/
├── AB_labels/
└── kmeans.pkl
```

---

## Current Features

- RGB → Lab conversion
- image resizing
- K-Means color quantization
- dataset generation
- preprocessing validation