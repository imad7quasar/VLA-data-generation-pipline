# Testing Dataset: Dirty vs Clean Image Comparison

## Overview

This folder contains training/testing data for radio interferometric image reconstruction, showing the relationship between **dirty images (input)** and **clean images (output)**.

---

## Files

### 1. `dirty_vs_clean_comparison.png` (1.3 MB)
**Comprehensive 9-panel comparison visualization**

#### Layout:
```
ROW 1 (Amplitude):
  - Dirty Image (Input) | Clean Image (Output) | True Sky (Ground Truth)

ROW 2 (Phase):
  - Dirty Image Phase | Clean Image Phase | True Sky Phase

ROW 3 (Error & PSF):
  - Error Map: Dirty - True | Error Map: Clean - True | PSF
```

#### What Each Panel Shows:
- **DIRTY IMAGE (Input)**: Raw reconstruction from interferometric visibility measurements
  - Contains PSF convolution artifacts
  - Shows sidelobes and blurring
  - Peak: normalized to 1.0
  
- **CLEAN IMAGE (Output)**: Restored/deconvolved image
  - Wiener filter restoration applied
  - Significantly reduced artifacts
  - Better match to true sky
  
- **TRUE SKY (Ground Truth)**: Original simulated sky model
  - Contains 4 sources (point + Gaussian)
  - Reference for error calculation

### 2. `dirty_vs_clean_analysis.png` (301 KB)
**Detailed analysis with 6 diagnostic plots**

#### Subplots:
1. **Horizontal Cross-Section (Y=0)** - Brightness profile across center
   - Shows blurring in dirty image
   - Cleaner profile in restored image
   
2. **Vertical Cross-Section (X=0)** - Vertical brightness profile
   - Demonstrates deconvolution effectiveness
   
3. **Radial Profile** - Brightness vs distance from center
   - Shows how restoration improves with radius
   
4. **Error Metrics** - Quantitative comparison
   - MSE (Mean Squared Error)
   - MAE (Mean Absolute Error)
   - Improvement percentages
   
5. **Peak Value Comparison** - Bar chart of brightest pixels
   - Dirty: 1.0000
   - Clean: 1.0000
   - Truth: 1.0000
   
6. **Fourier Spectrum** - Log-log frequency comparison
   - Shows how deconvolution affects frequency content

### 3. `dirty_clean_dataset.npz` (3.1 MB)
**Raw data file for machine learning training**

This NumPy compressed archive contains:
```python
{
    'dirty_image': (256, 256) array      # Input to restoration algorithm
    'clean_image': (256, 256) array      # Ground truth output/label
    'sky_truth': (256, 256) array        # Original sky model
    'psf': (256, 256) array              # Point spread function
    'dirty_phase': (256, 256) array      # Phase of dirty image
    'clean_phase': (256, 256) array      # Phase of clean image
}
```

#### Load with Python:
```python
import numpy as np

# Load the dataset
data = np.load('dirty_clean_dataset.npz')

# Access individual arrays
dirty = data['dirty_image']      # Shape: (256, 256)
clean = data['clean_image']      # Shape: (256, 256)
truth = data['sky_truth']        # Shape: (256, 256)
psf = data['psf']                # Shape: (256, 256)

# Use for ML training
X_train = dirty.reshape(1, 256, 256, 1)  # For neural networks
y_train = clean.reshape(1, 256, 256, 1)
```

---

## Dataset Statistics

### Image Properties
- **Size**: 256 × 256 pixels
- **Pixel Scale**: 2.0 arcsec/pixel
- **Total Field of View**: 512 × 512 arcsec (~8.5 arcmin)
- **Wavelength**: 0.214 m (1.4 GHz, L-band)

### Source Composition
- **4 sources total**:
  - 2 point sources (bright and faint)
  - 2 Gaussian extended sources
- **Total Flux**: 445.64 Jy

### Quality Metrics

| Metric | Dirty Image | Clean Image | Unit |
|--------|-------------|------------|------|
| Peak Brightness | 1.0000 | 1.0000 | normalized |
| MSE vs Truth | 0.1678 | 0.0030 | - |
| MAE vs Truth | 0.1024 | 0.0176 | - |
| **Improvement** | - | **98.2%** (MSE) | - |

### Noise Level
- **SNR (Signal-to-Noise Ratio)**: 40 dB
- **Type**: Gaussian thermal noise in UV domain
- **Effect**: Visible as small-scale structure in dirty image

---

## Physical Parameters

### VLA Configuration
- **Array Mode**: A-array (extended configuration)
- **Frequency**: 1.4 GHz (L-band)
- **Wavelength**: 0.214 m
- **Number of Antennas**: 9
- **Number of Baselines**: 36

### Source Parameters
```
Source 1: Point source
  Position: (-30, 20) pixels
  Flux: 3.0 Jy
  Phase: 0.5 rad

Source 2: Gaussian
  Position: (20, 0) pixels
  Flux: 2.5 Jy
  FWHM: 12 pixels
  Phase: -0.3 rad

Source 3: Point source
  Position: (30, 25) pixels
  Flux: 0.8 Jy
  Phase: 0.1 rad

Source 4: Small Gaussian
  Position: (-20, -30) pixels
  Flux: 1.2 Jy
  FWHM: 5 pixels
  Phase: 0.0 rad
```

---

## Using the Data

### Machine Learning Applications

#### PyTorch Example:
```python
import torch
import numpy as np

# Load data
data = np.load('dirty_clean_dataset.npz')
X = torch.tensor(data['dirty_image'], dtype=torch.float32)
y = torch.tensor(data['clean_image'], dtype=torch.float32)

# Reshape for CNN
X = X.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
y = y.unsqueeze(0).unsqueeze(0)

# Normalize
X = (X - X.mean()) / X.std()
y = (y - y.mean()) / y.std()

# Train U-Net or similar architecture
model = UNet()  # Your restoration network
```

#### TensorFlow Example:
```python
import tensorflow as tf
import numpy as np

# Load data
data = np.load('dirty_clean_dataset.npz')
X = data['dirty_image'][np.newaxis, :, :, np.newaxis]
y = data['clean_image'][np.newaxis, :, :, np.newaxis]

# Create dataset
dataset = tf.data.Dataset.from_tensor_slices((X, y))
dataset = dataset.shuffle(buffer_size=1).batch(1)

# Build restoration model
model = tf.keras.Sequential([
    tf.keras.layers.Input((256, 256, 1)),
    tf.keras.layers.Conv2D(32, 3, padding='same'),
    # ... more layers
])
```

### Analysis Applications

```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('dirty_clean_dataset.npz')

# Compute metrics
dirty = data['dirty_image']
clean = data['clean_image']
truth = data['sky_truth']

mse_dirty = np.mean((dirty - truth)**2)
mse_clean = np.mean((clean - truth)**2)

print(f"Dirty MSE: {mse_dirty:.6f}")
print(f"Clean MSE: {mse_clean:.6f}")
print(f"Improvement: {(1-mse_clean/mse_dirty)*100:.1f}%")

# Plot
fig, axes = plt.subplots(1, 3)
axes[0].imshow(dirty, cmap='viridis')
axes[0].set_title('Dirty')
axes[1].imshow(clean, cmap='viridis')
axes[1].set_title('Clean')
axes[2].imshow(truth, cmap='viridis')
axes[2].set_title('Truth')
plt.show()
```

---

## Deconvolution Method

The **Clean image** was generated using:
- **Algorithm**: Wiener filter restoration
- **Noise Level**: 0.01 (estimated from SNR)
- **Post-processing**: Gaussian smoothing (σ=1.0 pixel)

### Mathematical Formula:
```
Clean_image = IFFT[Dirty_FFT × Wiener_Filter]
Wiener_Filter = conj(PSF_FFT) / (|PSF_FFT|² + noise_level)
```

This approach provides:
- 98.2% MSE improvement
- Realistic noise handling
- Frequency-domain regularization

---

## Key Observations

1. **PSF Effects**: The dirty image is convolved with the PSF, causing:
   - Blurring of source structure
   - Sidelobe ringing around bright sources
   - Loss of small-scale detail

2. **Phase Information**: Phase is preserved through the pipeline
   - Dirty phase shows source structure
   - Clean phase better matches truth
   - Phase errors reduced by 95%+

3. **Spatial Artifacts**: 
   - Dirty image: broad sidelobe patterns
   - Clean image: localized sources
   - PSF shapes visible artifacts

4. **Frequency Content**:
   - Dirty: low-frequency dominated
   - Clean: more uniform frequency distribution
   - Closer match to true sky spectrum

---

## Extending the Dataset

To generate additional samples with different:
- Source configurations
- Noise levels
- Array configurations
- Frequencies

```python
from clean_comparison import generate_comparison_dataset

# Generate multiple datasets
for snr in [20, 40, 60, 100]:
    for mode in ['A', 'B', 'C', 'D']:
        generate_comparison_dataset(
            output_dir=f'testing/SNR{snr}_mode{mode}'
        )
```

---

## Related Files

- `../clean_comparison.py` - Script to generate datasets
- `../pipeline.py` - Main VLA imaging pipeline
- `../imaging.py` - PSF and dirty image computation
- `../visualization.py` - Plotting utilities

---

## Citation

If using this dataset, please cite:

```
VLA Imaging Pipeline v1.0 - Testing Dataset
December 2025
Dirty vs Clean Image Comparison for Radio Interferometric Restoration
```

---

## Contact & Support

For questions about the dataset:
1. Check `../USAGE_GUIDE.md` for pipeline details
2. Review `../README.md` for physics background
3. Examine `../clean_comparison.py` for generation method

---

**Generated**: December 2025
**Status**: Ready for machine learning applications
**Format**: PNG images + NumPy archive
