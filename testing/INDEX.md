# Testing Dataset Index

Complete collection of dirty vs clean image comparisons for radio interferometric image restoration.

## Contents Overview

### 📊 Dataset 1: Multiple Sources Scenario
**File Prefix:** `dirty_vs_clean_*`

Complex sky with 4 sources (2 point sources + 2 Gaussians)

**Files:**
- `dirty_vs_clean_comparison.png` (1.3 MB)
  - 9-panel comprehensive visualization
  - Side-by-side dirty, clean, and truth images
  - Phase information and error maps
  - PSF visualization (log scale)
  
- `dirty_vs_clean_analysis.png` (301 KB)
  - Detailed diagnostic plots (6 subplots)
  - Cross-sections and radial profiles
  - Error metrics and peak comparisons
  - Fourier spectrum analysis

- `dirty_clean_dataset.npz` (3.1 MB)
  - Raw numpy arrays for machine learning
  - Dirty image, clean image, sky truth
  - Phase information

**Key Metrics:**
- Total Flux: 445.64 Jy
- Image Size: 256×256 pixels
- Dirty MSE: 0.1678
- Clean MSE: 0.0030
- Improvement: **98.2%**

---

### ⭐ Dataset 2: Single Bright Star
**File Prefix:** `single_star_*`

Simple scenario with one point source (bright star) at center

**Files:**
- `single_star_dirty_vs_clean.png` (1.4 MB)
  - 9-panel comparison for single source
  - Amplitude, log-scale, and error visualization
  - Linear and logarithmic PSF views
  
- `single_star_analysis.png` (350 KB)
  - Horizontal and vertical cross-sections
  - Radial brightness profile
  - Detailed error metrics
  - Peak brightness bar chart
  - Fourier spectrum comparison

- `single_star_dataset.npz` (2.8 MB)
  - Single star training data
  - Dirty image (input)
  - Clean image (output/label)
  - Sky truth (ground truth)
  - Phase information

**Key Metrics:**
- Single Star Flux: 5.0 Jy
- Image Size: 256×256 pixels
- Dirty RMS Error: 0.4112
- Clean RMS Error: 0.0460
- Improvement: **98.7%**

---

## Quick Start

### Viewing the Comparisons

All PNG files can be opened directly in any image viewer:

```bash
# Windows
start testing/single_star_dirty_vs_clean.png
start testing/dirty_vs_clean_comparison.png

# Linux
display testing/single_star_dirty_vs_clean.png

# macOS
open testing/single_star_dirty_vs_clean.png
```

### Loading Data for Machine Learning

**Python (NumPy/TensorFlow):**
```python
import numpy as np

# Load single star dataset
data = np.load('testing/single_star_dataset.npz')
dirty = data['dirty_image']      # Input: 256×256
clean = data['clean_image']      # Output/Label: 256×256
truth = data['sky_truth']        # Ground truth: 256×256

# Prepare for neural network training
X_train = dirty.reshape(1, 256, 256, 1)
y_train = clean.reshape(1, 256, 256, 1)
```

**PyTorch:**
```python
import torch
import numpy as np

data = np.load('testing/single_star_dataset.npz')
dirty = torch.tensor(data['dirty_image'], dtype=torch.float32)
clean = torch.tensor(data['clean_image'], dtype=torch.float32)

# Reshape for CNN
X = dirty.unsqueeze(0).unsqueeze(0)  # (1, 1, 256, 256)
y = clean.unsqueeze(0).unsqueeze(0)
```

---

## Comparison: Multiple Sources vs Single Star

| Aspect | Multiple Sources | Single Star |
|--------|------------------|-------------|
| **Complexity** | 4 sources (point + Gaussian) | 1 point source |
| **Total Flux** | 445.64 Jy | 5.0 Jy |
| **Use Case** | Complex scenes, multi-object detection | Simple cases, point source restoration |
| **Dirty MSE** | 0.1678 | 0.4112 |
| **Clean MSE** | 0.0030 | 0.0460 |
| **Improvement** | 98.2% | 98.7% |
| **Difficulty** | High (extended + point sources) | Low (single point source) |
| **Training Data** | Better for general restoration | Better for point source networks |

---

## Physical Parameters

### Common Settings (Both Datasets)
- **VLA Array**: A-array (9 antennas, 36 baselines)
- **Frequency**: 1.4 GHz (L-band)
- **Wavelength**: 0.214 m
- **Image Size**: 256×256 pixels
- **Pixel Scale**: 2.0 arcsec/pixel
- **Field of View**: 512×512 arcsec (~8.5 arcmin)
- **Noise Level**: SNR = 40 dB

### Restoration Method
- **Algorithm**: Wiener Filter
- **Post-processing**: Gaussian smoothing (σ=1.0 pixel)
- **Noise Parameter**: 0.01

---

## Visual Interpretation

### What to Look for in the Images

1. **Dirty Image (Input)**
   - Shows raw interferometric reconstruction
   - Contains PSF convolution artifacts
   - Sidelobe patterns around bright sources
   - Noisy appearance (thermal noise)

2. **Clean Image (Output)**
   - Deconvolved/restored image
   - PSF effects removed
   - Cleaner source structure
   - Noise reduced

3. **True Sky (Ground Truth)**
   - Original simulated sky model
   - Clean point/Gaussian sources
   - No noise or PSF effects
   - Reference for error calculation

4. **Error Map**
   - Shows Dirty - True (larger errors)
   - Shows Clean - True (smaller errors)
   - Demonstrates deconvolution improvement
   - Structured patterns = PSF effects

5. **PSF (Point Spread Function)**
   - Response of system to point source
   - Central bright peak
   - Sidelobe patterns
   - Asymmetric due to incomplete UV coverage
   - Log scale reveals sidelobe structure

---

## Analysis Plots Explained

### Cross-Sections
- Show brightness profile through source center
- Reveal blurring in dirty image
- Show sharpening after deconvolution
- Compare to true source profile

### Radial Profile
- Brightness as function of distance from center
- Shows how restoration works at different scales
- Reveals sidelobe structure in dirty image
- Clean image profile approaches truth

### Error Metrics
- **RMS Error**: Root mean square difference vs truth
- **MAE**: Mean absolute error
- **Peak**: Maximum brightness value
- **Improvement**: Percentage reduction in error

### Fourier Spectrum
- Log-log plot of frequency components
- Shows how deconvolution affects different frequencies
- Dirty: concentrated at low frequencies (blurred)
- Clean: more uniform frequency distribution

---

## Dataset Files - Detailed Breakdown

### dirty_clean_dataset.npz Structure
```
{
  'dirty_image': ndarray(256, 256)     # Normalized dirty image (0-1)
  'clean_image': ndarray(256, 256)     # Normalized clean image (0-1)
  'sky_truth': ndarray(256, 256)       # Normalized true sky (0-1)
  'psf': ndarray(256, 256)             # Normalized PSF (0-1)
  'dirty_phase': ndarray(256, 256)     # Phase of dirty image (radians)
  'clean_phase': ndarray(256, 256)     # Phase of clean image (radians)
}
```

### single_star_dataset.npz Structure
```
{
  'dirty_image': ndarray(256, 256)     # Dirty image (single star)
  'clean_image': ndarray(256, 256)     # Clean image (restored)
  'sky_truth': ndarray(256, 256)       # True single star
  'psf': ndarray(256, 256)             # PSF
  'dirty_phase': ndarray(256, 256)     # Phase information
  'clean_phase': ndarray(256, 256)     # Phase information
}
```

Load in Python:
```python
import numpy as np
data = np.load('testing/single_star_dataset.npz')
print(data.files)  # List available arrays
dirty = data['dirty_image']
```

---

## Usage Examples

### Example 1: Load and Visualize
```python
import numpy as np
import matplotlib.pyplot as plt

data = np.load('testing/single_star_dataset.npz')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

axes[0].imshow(data['dirty_image'], cmap='hot')
axes[0].set_title('Dirty (Input)')

axes[1].imshow(data['clean_image'], cmap='hot')
axes[1].set_title('Clean (Output)')

axes[2].imshow(data['sky_truth'], cmap='hot')
axes[2].set_title('Truth')

plt.show()
```

### Example 2: Calculate Metrics
```python
import numpy as np

data = np.load('testing/single_star_dataset.npz')
dirty = data['dirty_image']
clean = data['clean_image']
truth = data['sky_truth']

# Calculate metrics
mse_dirty = np.mean((dirty - truth)**2)
mse_clean = np.mean((clean - truth)**2)

print(f"Dirty MSE: {mse_dirty:.6f}")
print(f"Clean MSE: {mse_clean:.6f}")
print(f"Improvement: {(1 - mse_clean/mse_dirty)*100:.1f}%")
```

### Example 3: Train Neural Network
```python
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
import numpy as np

# Load data
data = np.load('testing/single_star_dataset.npz')
X = torch.tensor(data['dirty_image'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
y = torch.tensor(data['clean_image'], dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Create dataset and loader
dataset = TensorDataset(X, y)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

# Define simple restoration network
class SimpleRestorer(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 1, 3, padding=1)
    
    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = self.conv3(x)
        return x

# Train
model = SimpleRestorer()
loss_fn = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

for epoch in range(10):
    for X_batch, y_batch in loader:
        pred = model(X_batch)
        loss = loss_fn(pred, y_batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")
```

---

## Image Quality Notes

### Single Star Dataset
- **Pros**: Simple, clean, ideal for learning point source deconvolution
- **Cons**: Too simple for real interferometric scenarios
- **Best for**: Baseline models, educational purposes

### Multiple Sources Dataset
- **Pros**: Realistic complexity, multiple object types
- **Cons**: More challenging training
- **Best for**: Production models, real-world scenarios

---

## Regenerating Datasets

To create new datasets or modify parameters:

**Single Star:**
```bash
python single_star_comparison.py
```

**Multiple Sources:**
```bash
python clean_comparison.py
```

Both scripts accept output directory parameter:
```python
from single_star_comparison import generate_single_star_comparison
generate_single_star_comparison(output_dir='testing')
```

---

## File Sizes Summary

| File | Size | Format |
|------|------|--------|
| dirty_vs_clean_comparison.png | 1.3 MB | PNG (150 dpi) |
| dirty_vs_clean_analysis.png | 301 KB | PNG (150 dpi) |
| dirty_clean_dataset.npz | 3.1 MB | NumPy compressed |
| single_star_dirty_vs_clean.png | 1.4 MB | PNG (150 dpi) |
| single_star_analysis.png | 350 KB | PNG (150 dpi) |
| single_star_dataset.npz | 2.8 MB | NumPy compressed |
| **Total** | **~8.9 MB** | Mixed |

---

## Related Documentation

- `../README.md` - VLA pipeline technical details
- `../USAGE_GUIDE.md` - Complete API reference
- `../clean_comparison.py` - Multiple sources generation script
- `../single_star_comparison.py` - Single star generation script

---

## Key Takeaways

1. **Dirty images** contain PSF convolution artifacts and noise
2. **Clean images** are restored versions using deconvolution
3. **Improvement** is typically 95%+ in error metrics
4. **Both datasets** are valid for training restoration networks
5. **Single star** is simpler, **multiple sources** is more realistic

---

**Generated:** February 2026
**Status:** Ready for machine learning applications
**Format:** PNG visualizations + NumPy data archives
