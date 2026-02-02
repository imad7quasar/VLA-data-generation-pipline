# VLA Interferometric Imaging Pipeline

A complete, from-first-principles implementation of a VLA interferometric imaging pipeline in Python. This pipeline simulates the entire process from sky model to dirty image, with comprehensive verification and visualization.

## Project Structure

```
vla_pipeline/
├── vla_array.py          # VLA array configuration and UV coverage
├── sky_model.py          # Sky brightness distribution (point/Gaussian sources)
├── visibility.py         # Visibility generation via FFT
├── imaging.py            # PSF and dirty image computation
├── pipeline.py           # Main pipeline orchestration
├── visualization.py      # Advanced plotting and analysis
├── config.py             # Configuration parameters
└── README.md             # This file
```

## Features

### 1️⃣ Sky Model (`sky_model.py`)
- **Support**: Single/multiple point sources and Gaussian brightness distributions
- **Format**: 2D complex image plane with amplitude and phase
- **Flexibility**: Elliptical Gaussians with rotation
- **Pixel units**: Consistent with wavelength and pixel scale

### 2️⃣ UV Sampling / Array Configuration (`vla_array.py`)
- **Realistic VLA**: Four standard configurations (A, B, C, D)
- **Antenna positions**: ENU coordinates based on NRAO specifications
- **UV coverage**: Proper baseline-to-UV transformation including hour angle and declination
- **Hermitian symmetry**: Automatic enforcement of visibility conjugate symmetry
- **Visualization**: UV coverage plots with baseline conjugates

### 3️⃣ Visibility Generation (`visibility.py`)
- **FFT-based**: Complex visibilities computed as FFT of sky sampled at UV points
- **Proper handling**: Bilinear interpolation for accurate UV sampling
- **Noise**: Optional Gaussian noise addition in UV domain (not image space)
- **Verification**: Visibility amplitude and phase spectra

### 4️⃣ PSF / Beam (`imaging.py`)
- **Computation**: PSF as inverse FFT of UV coverage (no sky)
- **Features**: Bright central peak, realistic sidelobes reflecting UV coverage sparseness
- **Verification**: Log-scale visualization showing sidelobe structure
- **Asymmetry**: Correctly reflects incomplete UV plane sampling

### 5️⃣ Dirty Image (`imaging.py`)
- **Computation**: Inverse FFT of sampled visibilities
- **Properties**: Blurred (vs. sky), includes sidelobes, non-zero everywhere source contributes
- **Verification**: Visual comparison with sky model

### 6️⃣ Error Maps (`imaging.py`)
- **Computation**: Difference between dirty image and sky (Dirty - Sky)
- **Structure**: Non-trivial spatial structure; not coincidental perfect match
- **Analysis**: RMS error, max error statistics

### 7️⃣ Sanity Tests (`pipeline.py`)
- **Single point source**: Center position, full UV coverage, no noise
- **Expected result**: Dirty image approximately equals sky (within PSF broadening)
- **PSF verification**: Matches inverse FFT of UV coverage
- **Deviation explanation**: FFT discretization, PSF convolution effects

### 8️⃣ Comprehensive Visualization
All outputs include:
- **Sky**: Amplitude and phase
- **Dirty image**: Amplitude and phase
- **PSF**: Log scale with sidelobe detail
- **Error map**: Structured difference
- **UV coverage**: Complete coverage including conjugates
- **Visibility spectra**: Amplitude and phase vs. baseline length
- **Radial profiles**: Brightness vs. radius
- **Cross sections**: Horizontal and vertical cuts

**Units**: All axes properly labeled (wavelengths for UV, pixels for images, arcseconds for sky)
**Color scales**: Physically correct, with log scales where appropriate

---

## Usage

### Basic Pipeline Run
```python
from pipeline import main

# Run complete pipeline with default settings
pipeline, pipeline_test = main()
```

### Custom Configuration
```python
from pipeline import VLAPipeline

# Create custom pipeline
pipeline = VLAPipeline(
    array_mode='C',              # C-array configuration
    frequency_ghz=5.0,           # C-band
    image_size=512,              # Higher resolution
    pixel_scale_arcsec=1.0       # Finer pixels
)

# Define sky model
sources = [
    {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 2.0},
    {'type': 'gaussian', 'x_pix': 30, 'y_pix': 20, 'flux_jy': 1.0, 'fwhm_pix': 10}
]
pipeline.setup_sources(sources)

# Run imaging pipeline
pipeline.compute_visibilities(hour_angle_deg=0, declination_deg=45)
pipeline.compute_images()

# Visualize results
fig = pipeline.plot_full_pipeline()
```

### Advanced Visualization
```python
from visualization import ComprehensivePlotter, plot_uv_plane_sampling

# PSF analysis
fig, axes = ComprehensivePlotter.plot_psf_analysis(pipeline.psf)

# Radial profile
fig, ax = ComprehensivePlotter.plot_radial_profile(pipeline.dirty_image)

# Cross sections
fig, axes = ComprehensivePlotter.plot_cross_section(pipeline.dirty_image)

# UV plane detail
fig, ax = plot_uv_plane_sampling(pipeline.u_samples, pipeline.v_samples)
```

### Run Sanity Tests
```python
from pipeline import test_sanity_point_source

pipeline = test_sanity_point_source()
```

---

## Physical Parameters

### Array Configurations
| Mode | Description | Max Baseline | Resolution @ 1.4 GHz |
|------|-------------|--------------|---------------------|
| A | Extended | 36.4 km | ~0.5 arcsec |
| B | Intermediate | 10 km | ~1.5 arcsec |
| C | Compact | 3 km | ~5 arcsec |
| D | Most compact | 600 m | ~25 arcsec |

### Common Observing Frequencies
- L-band: 1.4 GHz (λ ≈ 21 cm)
- C-band: 5.0 GHz (λ ≈ 6 cm)
- X-band: 10 GHz (λ ≈ 3 cm)
- Ku-band: 15 GHz (λ ≈ 2 cm)

### Key Relations
- **Wavelength**: λ = c / f
- **Beam size**: θ ≈ λ / B (where B is baseline length)
- **PSF FWHM** at 1.4 GHz, A-array: ~0.5 arcsec
- **Field of view**: ~30' at L-band, scales as λ

---

## Mathematical Framework

### Visibility-Sky Relationship
The complex visibility at UV point (u,v) is:
$$V(u,v) = \int \int I(l,m) e^{-2\pi i(ul + vm)} dl \, dm$$

In discrete form (FFT):
$$V = \text{FFT}[I_{\text{sky}}]$$

### PSF Definition
The PSF is the inverse FFT of the UV sampling function (coverage):
$$\text{PSF} = \text{IFFT}[\text{Coverage}]$$

### Dirty Image
The dirty image is the inverse FFT of the sampled visibilities:
$$I_{\text{dirty}} = \text{IFFT}[V_{\text{sampled}}]$$

### Error Map
$$E = I_{\text{dirty}} - I_{\text{sky}}$$

---

## Verification Checklist

- ✅ **Hermitian symmetry**: V(-u,-v) = conj(V(u,v))
- ✅ **PSF properties**: Central peak, realistic sidelobes
- ✅ **Dirty image**: Blurred compared to sky, includes sidelobes
- ✅ **Error map**: Non-zero, structured, not coincidentally zero
- ✅ **Sanity test**: Point source → dirty peak ≈ sky peak (within PSF)
- ✅ **Unit consistency**: Wavelengths in UV, pixels in image, arcseconds in sky
- ✅ **Phase tracking**: Visibility phases affect dirty image structure
- ✅ **Baseline dependencies**: Visibility amplitude decreases with baseline

---

## Output Files

The pipeline generates:
- `vla_pipeline_full.png` - Comprehensive 9-panel pipeline overview
- `uv_coverage.png` - UV plane sampling visualization
- `sky_model.png` - Sky amplitude and phase
- `visibility_spectra.png` - Amplitude and phase vs. baseline
- `psf.png` - PSF with log-scale sidelobe detail
- `dirty_sky_comparison.png` - Side-by-side comparison

---

## Dependencies

- NumPy: Array operations
- SciPy: FFT, constants, signal processing
- Matplotlib: Visualization

See `requirements.txt` for exact versions.

---

## Implementation Notes

### FFT Normalization
Visibilities are normalized by image size squared to maintain flux conservation:
```python
V_FFT = fft2(sky) / (image_size ** 2)
```

### UV Sampling
UV coordinates are computed from antenna positions via:
```
u = baseline_east * sin(HA) / wavelength
v = baseline_north * sin(Dec) * cos(HA) + baseline_up * cos(Dec) / wavelength
```

### Hermitian Enforcement
Conjugate visibility pairs are averaged to enforce symmetry:
```python
V(-u, -v) = conj(V(u, v))
```

### Interpolation
Visibilities at arbitrary UV points are computed via bilinear interpolation of the FFT.

---

## Future Extensions

- Gridding kernels (Gaussian, Kaiser-Bessel) for better UV sampling
- Self-calibration for phase correction
- Natural weighting schemes
- Multi-frequency synthesis
- 3D data cube support
- Deconvolution (CLEAN algorithm)
- Restore images with beam

---

## References

- Thompson, Moran & Swenson: "Interferometry and Synthesis in Radio Astronomy" (2nd ed.)
- Cornwell: "Radio-Interferometric Imaging of Coherent Transients"
- NRAO VLA Documentation: https://science.nrao.edu/facilities/vla/

---

## Author

VLA Imaging Pipeline v1.0
December 2025

---

## License

Open source for research and educational purposes.
