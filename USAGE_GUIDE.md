# VLA Imaging Pipeline - Complete Usage Guide

## Table of Contents
1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Module Reference](#module-reference)
4. [Advanced Usage](#advanced-usage)
5. [Understanding the Physics](#understanding-the-physics)
6. [Troubleshooting](#troubleshooting)

---

## Installation

### Requirements
- Python 3.8+
- NumPy 1.20+
- SciPy 1.7+
- Matplotlib 3.4+

### Setup

```bash
# Navigate to project directory
cd VLA\ data\ sim

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "from vla_array import VLAConfiguration; print('✓ Pipeline installed successfully')"
```

---

## Quick Start

### Run Tests
```bash
python test_pipeline.py
```
This validates that all 8 requirements are properly implemented.

### Run Examples
```bash
python examples.py
```
This generates 6 different example scenarios with visualizations.

### Run Main Pipeline
```python
from pipeline import main

pipeline, pipeline_test = main()
```

---

## Module Reference

### 1. VLA Array Configuration (`vla_array.py`)

**Class: `VLAConfiguration`**

Create and configure VLA array:
```python
from vla_array import VLAConfiguration

# Initialize array
vla = VLAConfiguration(
    mode='A',              # 'A', 'B', 'C', or 'D'
    frequency_ghz=1.4      # Observing frequency
)

# Get UV coordinates
u, v, baselines, pairs = vla.get_uv_coordinates(
    hour_angle_deg=0,      # Hour angle for Earth rotation
    declination_deg=45     # Source declination
)

# Plot UV coverage
fig, ax = vla.plot_uv_coverage(declination_deg=45)
```

**Key Methods:**
- `get_baselines()` - Compute all baseline vectors
- `get_uv_coordinates()` - Transform baselines to UV plane
- `plot_uv_coverage()` - Visualize UV coverage

**Properties:**
- `n_antennas` - Number of antennas
- `wavelength_m` - Wavelength in meters
- `antenna_positions` - ENU coordinates of antennas

---

### 2. Sky Model (`sky_model.py`)

**Class: `SkyModel`**

Define sky brightness distribution:
```python
from sky_model import SkyModel

sky = SkyModel(
    image_size=256,              # Pixels
    pixel_scale_arcsec=2.0,      # Arcsec/pixel
    wavelength_m=0.214           # Meters (1.4 GHz)
)

# Add sources
sky.add_point_source(
    x_offset_pix=0,
    y_offset_pix=0,
    flux_jy=1.5,                 # Flux in Jansky
    phase_rad=0.5                # Phase in radians
)

sky.add_gaussian_source(
    x_offset_pix=20,
    y_offset_pix=15,
    flux_jy=0.8,
    fwhm_pix=8,                  # FWHM in pixels
    phase_rad=0.2,
    ellipticity=1.0,             # Minor/major axis ratio
    pa_deg=45                    # Position angle
)

# Generate sky
sky.generate_sky()

# Plot
fig, axes = sky.plot_sky(title='My Sky Model')
```

**Key Methods:**
- `add_point_source()` - Add point source
- `add_gaussian_source()` - Add Gaussian brightness distribution
- `generate_sky()` - Create 2D sky array
- `plot_sky()` - Visualize amplitude and phase

---

### 3. Visibility Generation (`visibility.py`)

**Class: `VisibilityGenerator`**

Compute complex visibilities:
```python
from visibility import VisibilityGenerator

vis_gen = VisibilityGenerator(sky, vla)

# Compute FFT of sky
sky_fft = vis_gen.compute_sky_fft()

# Sample visibilities at baseline positions
visibilities, u, v = vis_gen.sample_visibilities(
    hour_angle_deg=0,
    declination_deg=45
)

# Add thermal noise (optional)
vis_gen.add_uv_noise(snr_db=100)

# Enforce Hermitian symmetry
vis_gen.enforce_hermitian_symmetry()

# Plot visibility spectra
fig, axes = vis_gen.plot_visibility_spectra()
```

**Key Methods:**
- `compute_sky_fft()` - FFT of sky brightness
- `sample_visibilities()` - Sample at UV points
- `add_uv_noise()` - Add Gaussian noise in UV domain
- `enforce_hermitian_symmetry()` - Ensure V(-u,-v) = conj(V(u,v))
- `plot_visibility_spectra()` - Plot amplitude and phase vs baseline

**Properties:**
- `visibilities` - Complex visibility array
- `u_samples`, `v_samples` - UV coordinates
- `sky_fft` - FFT of sky

---

### 4. Imaging (`imaging.py`)

**Class: `ImagingEngine`**

Create PSF and dirty image:
```python
from imaging import ImagingEngine

engine = ImagingEngine(
    image_size=256,
    wavelength_m=0.214,
    pixel_scale_arcsec=2.0
)

# Compute PSF from UV coverage
psf = engine.compute_psf(u_samples, v_samples)

# Compute dirty image from visibilities
dirty_image = engine.compute_dirty_image(
    visibilities, u_samples, v_samples
)

# Plot PSF
fig, ax = engine.plot_psf()
```

**Class: `DirtyImageAnalyzer`**

Analyze imaging results:
```python
from imaging import DirtyImageAnalyzer

analyzer = DirtyImageAnalyzer(dirty_image, sky.sky)

# Get error statistics
stats = analyzer.get_statistics()
print(f"RMS error: {stats['rms_error']}")

# Plot error map
fig, ax = analyzer.plot_error_map()
```

---

### 5. Complete Pipeline (`pipeline.py`)

**Class: `VLAPipeline`**

Full workflow in one class:
```python
from pipeline import VLAPipeline

# Create pipeline
pipeline = VLAPipeline(
    array_mode='A',
    frequency_ghz=1.4,
    image_size=256,
    pixel_scale_arcsec=2.0
)

# Setup sources
sources = [
    {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 1.0},
    {'type': 'gaussian', 'x_pix': 20, 'y_pix': 15, 'flux_jy': 0.5, 'fwhm_pix': 5}
]
pipeline.setup_sources(sources)

# Compute everything
pipeline.compute_visibilities(
    hour_angle_deg=0,
    declination_deg=45,
    add_noise=False,
    snr_db=100
)
pipeline.compute_images()

# Plot full pipeline
fig = pipeline.plot_full_pipeline()
```

**Key Methods:**
- `setup_sources()` - Add sources to sky model
- `compute_visibilities()` - Generate visibilities
- `compute_images()` - Create PSF and dirty image
- `plot_full_pipeline()` - 9-panel comprehensive visualization

**Properties:**
- `sky_model` - SkyModel instance
- `vla_array` - VLAConfiguration instance
- `visibilities` - Complex visibilities
- `psf` - Point spread function
- `dirty_image` - Reconstructed image
- `analyzer` - DirtyImageAnalyzer instance

---

### 6. Visualization (`visualization.py`)

**Class: `ComprehensivePlotter`**

Advanced visualization utilities:
```python
from visualization import ComprehensivePlotter

# PSF analysis
fig, axes = ComprehensivePlotter.plot_psf_analysis(psf)

# Radial profile
fig, ax = ComprehensivePlotter.plot_radial_profile(image)

# Cross sections
fig, axes = ComprehensivePlotter.plot_cross_section(image)

# Dirty vs Sky comparison
fig, axes = ComprehensivePlotter.plot_comparison_dirty_sky(sky, dirty)
```

**Static Methods:**
- `plot_psf_analysis()` - Detailed PSF visualization
- `plot_radial_profile()` - Brightness vs radius
- `plot_cross_section()` - Horizontal and vertical cuts
- `plot_comparison_dirty_sky()` - Side-by-side comparison

**Functions:**
- `plot_uv_plane_sampling()` - Detailed UV coverage

---

## Advanced Usage

### Custom Source Configuration

```python
from pipeline import VLAPipeline

pipeline = VLAPipeline()

# Complex multi-source scenario
sources = [
    # Bright point source at center
    {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 5.0, 'phase_rad': 0},
    
    # Resolved Gaussian (e.g., extended source)
    {'type': 'gaussian', 'x_pix': 30, 'y_pix': 0, 'flux_jy': 2.0, 
     'fwhm_pix': 20, 'ellipticity': 0.7, 'pa_deg': 45},
    
    # Fainter source
    {'type': 'point', 'x_pix': -20, 'y_pix': 30, 'flux_jy': 0.3},
]

pipeline.setup_sources(sources)
```

### Different Array Configurations

```python
# Compact array for high surface brightness
pipeline_compact = VLAPipeline(array_mode='D', frequency_ghz=1.4)

# Extended array for high resolution
pipeline_extended = VLAPipeline(array_mode='A', frequency_ghz=10.0)

# Compare PSF resolution
import matplotlib.pyplot as plt

fig, axes = plt.subplots(1, 2)

psf_compact = pipeline_compact.psf / np.max(pipeline_compact.psf)
psf_extended = pipeline_extended.psf / np.max(pipeline_extended.psf)

axes[0].imshow(psf_compact, origin='lower', cmap='viridis')
axes[0].set_title('D-array PSF (compact)')

axes[1].imshow(psf_extended, origin='lower', cmap='viridis')
axes[1].set_title('A-array PSF (extended)')

plt.show()
```

### Earth Rotation Tracks

```python
from vla_array import VLAConfiguration

vla = VLAConfiguration(frequency_ghz=1.4)

# Simulate observation over 4 hours
hour_angles = [0, 1, 2, 3, 4]
uv_tracks = []

for ha in hour_angles:
    u, v, _, _ = vla.get_uv_coordinates(
        hour_angle_deg=ha * 15,  # 15 deg/hour
        declination_deg=45
    )
    uv_tracks.append((u, v))

# Plot UV coverage with Earth rotation
fig, ax = plt.subplots()
colors = plt.cm.viridis(np.linspace(0, 1, len(hour_angles)))

for (u, v), color, ha in zip(uv_tracks, colors, hour_angles):
    ax.scatter(u, v, s=20, alpha=0.6, color=color, label=f'HA={ha}h')

ax.legend()
ax.set_xlabel('U (wavelengths)')
ax.set_ylabel('V (wavelengths)')
ax.set_title('UV Coverage with Earth Rotation')
plt.show()
```

### Adding Thermal Noise

```python
vis_gen = VisibilityGenerator(sky, vla)
vis_gen.compute_sky_fft()
vis, u, v = vis_gen.sample_visibilities()

# Add thermal noise at specified SNR
vis_gen.add_uv_noise(snr_db=50)  # 50 dB SNR

# Compute noisy dirty image
engine = ImagingEngine(...)
dirty_noisy = engine.compute_dirty_image(
    vis_gen.visibilities, u, v
)
```

---

## Understanding the Physics

### Visibility-Brightness Relation

For a source with brightness distribution $I(\ell, m)$, the visibility at UV point $(u,v)$ is:

$$V(u,v) = \int \int I(\ell,m) e^{-2\pi i(u\ell + vm)} d\ell \, dm$$

In discrete form (FFT):
$$V = \text{FFT}[I_{\text{sky}}]$$

### PSF Definition

The PSF represents the response to a point source:

$$\text{PSF} = \text{IFFT}[\text{Sampling Function}]$$

The sampling function is the coverage of the UV plane.

### Dirty Image Formation

$$I_{\text{dirty}} = \text{IFFT}[V_{\text{sampled}}] = (\text{PSF}) * I_{\text{true}}$$

The dirty image is a convolution of the true sky with the PSF.

### Hermitian Symmetry

For real-valued brightness distributions:
$$V(-u,-v) = V^*(u,v)$$

This must be enforced for physically realistic visibilities.

---

## Troubleshooting

### Issue: PSF appears to have same amplitude everywhere

**Cause:** Full UV coverage with many antennas
**Solution:** Try compact array (mode='D') or fewer antennas

### Issue: Dirty image doesn't match sky

**Cause:** This is normal! Dirty image is convolved with PSF
**Expected:** Dirty image is blurred version of sky
**Analysis:** Use error map to quantify difference

### Issue: Visibilities are very small

**Cause:** Correct normalization in FFT
**Check:** 
- Source flux is defined correctly
- No sky model created empty

### Issue: Memory error with large images

**Cause:** FFT size grows as image_size²
**Solution:** Reduce image_size or use smaller pixel_scale_arcsec

### Issue: Phase wrapping in visibility spectra

**Cause:** Expected for complex visibilities
**Handle:** Use np.unwrap() for phase tracking

---

## Tips & Best Practices

1. **Start simple**: Test with single point source first
2. **Check conservation**: Total flux in image should match source fluxes
3. **Understand resolution**: Beam size = λ / baseline length
4. **Monitor noise**: Use SNR parameter carefully
5. **Compare arrays**: Use same source with different configurations
6. **Save plots**: Pipeline generates many informative visualizations
7. **Document parameters**: Keep record of frequency, array mode, pixel scale

---

## References

### Papers
- Thompson, Moran, Swenson: "Interferometry and Synthesis in Radio Astronomy" (2nd ed.)
- Cornwell: "Radio-Interferometric Imaging of Coherent Transients"

### External Resources
- [NRAO VLA Documentation](https://science.nrao.edu/facilities/vla/)
- [CASA Documentation](https://casa.nrao.edu/)
- [Astropy Coordinates](https://docs.astropy.org/en/stable/coordinates/)

---

*Last updated: December 2025*
