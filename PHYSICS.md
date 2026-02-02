# VLA Pipeline Physics & Mathematics

Complete mathematical framework for the VLA interferometric imaging pipeline.

---

## 1. Visibility-Brightness Relationship

### Fundamental Equation

The visibility at UV point $(u, v)$ is the 2D Fourier transform of the sky brightness:

$$V(u,v) = \int_{-\infty}^{\infty} \int_{-\infty}^{\infty} I(\ell,m) e^{-2\pi i(u\ell + vm)} d\ell \, dm$$

Where:
- $I(\ell, m)$ is sky brightness (intensity) distribution
- $(u, v)$ are UV coordinates in wavelengths
- $(\ell, m)$ are sky coordinates (direction cosines)

### Discrete Form (FFT)

For discrete image arrays:

$$V_{\text{discrete}} = \text{FFT}[I_{\text{sky}}]$$

The inverse relationship reconstructs the sky from visibilities:

$$I_{\text{reconstructed}} = \text{IFFT}[V_{\text{sampled}}]$$

---

## 2. Baseline-to-UV Transformation

### ENU Coordinates

Antenna positions in Earth-North-Up (ENU) local coordinates:
- **E**: East direction
- **N**: North direction  
- **U**: Up (zenith) direction

### Baseline Vector

The baseline between antenna $i$ and $j$:

$$\vec{B}_{ij} = \vec{r}_j - \vec{r}_i$$

### UV Transformation

For a source at:
- Hour angle: $h$
- Declination: $\delta$

The UV coordinates are:

$$u = B_E \sin(h) - B_N \cos(h)$$

$$v = B_E \cos(h) \sin(\delta) + B_N \sin(h) \sin(\delta) + B_U \cos(\delta)$$

Converting to wavelengths:

$$u_\lambda = \frac{u}{\lambda}, \quad v_\lambda = \frac{v}{\lambda}$$

### Hour Angle (Earth Rotation)

As the source tracks across the sky, hour angle changes:

$$h(t) = h_0 + \omega_E t$$

Where $\omega_E \approx 15°$/hour is Earth's rotation rate.

This causes the UV points to trace elliptical paths (UV tracks).

---

## 3. Hermitian Symmetry

### For Real-Valued Brightness

If the sky brightness is real ($I(\ell,m) \in \mathbb{R}$):

$$V(-u, -v) = V^*(u,v)$$

Where $*$ denotes complex conjugate.

### Implementation

In the pipeline:
1. Compute visibilities for positive $(u,v)$ points only
2. For each visibility $V(u,v)$, create conjugate at $V(-u,-v) = V^*(u,v)$
3. For points without explicit conjugates, average:

$$\tilde{V}(u,v) = \frac{V(u,v) + V^*(-u,-v)}{2}$$

---

## 4. Point Spread Function (PSF)

### Definition

The PSF is the response to a point source (Dirac delta function):

$$\text{PSF}(\ell,m) = \text{IFFT}[\text{Sampling Function}]$$

The sampling function is binary: 1 where baselines are observed, 0 elsewhere.

### Mathematical Form

$$\text{PSF}(\ell,m) = \left| \text{IFFT}[W(\ell,m)] \right|^2$$

Where $W(\ell,m)$ is the coverage function (UV plane mask).

### Properties

1. **Central peak:** Inverse of UV coverage diameter
2. **Sidelobes:** Reflect incompleteness of UV plane sampling
3. **Beam size (FWHM):** $\theta_{FWHM} \approx \lambda / B_{\max}$
4. **Asymmetry:** Non-uniform UV coverage produces asymmetric PSF

---

## 5. Dirty Image Formation

### Convolution Relationship

The dirty image is the true sky **convolved** with the PSF:

$$I_{\text{dirty}}(\ell,m) = (\text{PSF} * I_{\text{true}})(\ell,m)$$

In Fourier domain:

$$V_{\text{dirty}}(u,v) = V_{\text{true}}(u,v) \times W(u,v)$$

Where $W(u,v)$ is the sampling function (coverage).

### Computational Steps

1. **Sample visibilities:** Only observe at baseline-determined UV points
2. **Inverse FFT:** 
   $$I_{\text{dirty}} = \text{IFFT}[V_{\text{sampled}}]$$

### Effects

- Source appears **blurred** (broader in dirty image)
- **Sidelobes** appear around sources
- Noise propagates from UV domain to image domain

---

## 6. Error Map Analysis

### Definition

The error (residual) map is:

$$E(\ell,m) = I_{\text{dirty}}(\ell,m) - I_{\text{true}}(\ell,m)$$

### Decomposition

This error comes from:

1. **PSF convolution:** Blurring effect
2. **Aliasing:** FFT grid discretization
3. **Interpolation error:** Bilinear interpolation of FFT
4. **Sampling error:** Sparse UV plane

### Statistical Properties

- **RMS error:** $\sigma_E = \sqrt{\langle E^2 \rangle}$
- **Max error:** $E_{\max} = \max|E|$
- **Mean error:** $\mu_E = \langle E \rangle$ (should be ~0 by symmetry)

---

## 7. FFT Normalization

### Proper Scaling

To preserve flux, normalization factor is $1/N^2$:

$$V = \frac{1}{N^2} \text{FFT}[I_{\text{sky}}]$$

$$I_{\text{reconstructed}} = \text{IFFT}[V] \times N^2$$

This ensures:
$$\sum I_{\text{true}} \approx \sum I_{\text{reconstructed}}$$

### Pixel Scale Relationship

The relationship between image and UV domains:

**Pixel scale in image plane:** $s_\ell$ (radians/pixel)

**Frequency spacing in UV plane:** $\Delta u = \frac{1}{N \cdot s_\ell / \lambda}$ (wavelengths)

This is the Nyquist-Shannon sampling theorem in 2D.

---

## 8. Thermal Noise

### UV Domain Addition (Correct Method)

Noise is added in UV domain:

$$V_{\text{noisy}} = V_{\text{true}} + n(u,v)$$

Where noise is complex Gaussian:

$$n(u,v) = \sqrt{\frac{N_0}{2}} \left( n_R + i \, n_I \right)$$

With $n_R, n_I \sim \mathcal{N}(0,1)$

### SNR Relationship

For target SNR in dB:

$$\text{SNR}_{\text{dB}} = 10 \log_{10}\left(\frac{P_{\text{signal}}}{P_{\text{noise}}}\right)$$

Noise power:
$$P_{\text{noise}} = \frac{P_{\text{signal}}}{10^{\text{SNR}_{\text{dB}}/10}}$$

### Why UV Domain?

- Correct antenna integration effects
- Proper baseline-dependent weighting
- Physical correspondence to receiver noise

---

## 9. Source Models

### Point Source

Brightness distribution (Dirac delta):

$$I(\ell,m) = S \, \delta(\ell - \ell_0) \delta(m - m_0) e^{i\phi}$$

Visibility:
$$V(u,v) = S \, e^{-2\pi i(u\ell_0 + vm_0)} e^{i\phi}$$

Amplitude independent of baseline (for point source).

### Gaussian Source

Brightness distribution:

$$I(\ell,m) = S \exp\left(-\frac{(\ell-\ell_0)^2}{2\sigma_\ell^2} - \frac{(m-m_0)^2}{2\sigma_m^2}\right) e^{i\phi}$$

Visibility (Gaussian Fourier transform is Gaussian):

$$V(u,v) \propto \exp\left(-2\pi^2 \sigma_\ell^2 u^2 - 2\pi^2 \sigma_m^2 v^2\right)$$

Amplitude decreases with baseline (longer baselines resolve Gaussian).

---

## 10. Beam Size Relationship

### Diffraction Limit

For an antenna of diameter $D$:

$$\theta_{\text{beam}} \approx 1.22 \frac{\lambda}{D}$$

### Synthesized Beam

From interferometer with maximum baseline $B_{\max}$:

$$\theta_{\text{synth}} \approx \frac{\lambda}{B_{\max}}$$

At 1.4 GHz with VLA A-array ($B_{\max} = 36.4$ km):

$$\theta_{\text{synth}} = \frac{0.214 \text{ m}}{36,400 \text{ m}} = 5.87 \times 10^{-6} \text{ rad} = 0.54 \text{ arcsec}$$

---

## 11. Field of View

### Primary Beam

The primary beam of VLA is:

$$\text{PB}(\theta) \approx \left[\frac{J_1(x)}{2x}\right]^2$$

Where $x = \pi D \theta / \lambda$ and $J_1$ is Bessel function.

FWHM of primary beam:
$$\text{FWHM}_{\text{PB}} \approx 1.22 \frac{\lambda}{D}$$

For D = 25 m (VLA dish), at 1.4 GHz:
$$\text{FWHM}_{\text{PB}} \approx 30 \text{ arcminutes}$$

### Field of View Definition

Typically FOV is defined as primary beam FWHM or first null.

---

## 12. Visibility Phase Information

### Phase from Source Position

Source offset creates linear phase:

$$\phi(u,v) = -2\pi(u\Delta\ell + v\Delta m)$$

This allows astrometric localization.

### Phase from Calibration

Complex gain variations:
$$g_i(t) = |g_i| e^{i\phi_i}$$

Combined antenna gains:
$$V_{\text{measured}} = g_i^* g_j \, V_{\text{true}}$$

Self-calibration uses visibility phases to determine $g_i$.

---

## 13. Sanity Test Analysis

### Single Point Source, Full UV, No Noise

**True sky:**
$$I_{\text{true}}(\ell,m) = S \, \delta(\ell) \delta(m)$$

**Visibility:**
$$V_{\text{true}}(u,v) = S \, (\text{constant})$$

**Sampled visibility:**
$$V_{\text{sampled}}(u,v) = S \quad \text{(all baselines)}$$

**Ideal dirty image:**
$$I_{\text{dirty}} = \text{IFFT}[V_{\text{sampled}}] = S \, \text{PSF}(u_0,v_0)$$

Where PSF is evaluated at sampled points only.

### Expected Behavior

- **Peak ratio:** $I_{\text{dirty,peak}} / I_{\text{true,peak}} \approx (\text{fraction of UV plane covered})^{-1}$
- **With typical VLA:** Only discrete baselines, peak attenuated significantly
- **Not unit ratio:** This is expected and correct!

---

## 14. Physical Units Verification

| Quantity | SI Units | Pipeline Units | Conversion |
|----------|----------|-----------------|------------|
| Brightness | W/(m²·Hz·sr) | Jy | 1 Jy = $10^{-26}$ W/(m²·Hz) |
| Baseline | meters | wavelengths | $\lambda = c/f$ |
| Image | radians | pixels | pixel scale × pixel index |
| Phase | radians | radians | $[0, 2\pi]$ |
| Frequency | Hz | GHz | $10^9$ conversion |

---

## 15. Derivation of Error Statistics

### RMS Error

$$\text{RMS} = \sqrt{\frac{1}{N_{\ell} N_m} \sum_{\ell,m} |E(\ell,m)|^2}$$

### Max Absolute Error

$$E_{\max} = \max_{\ell,m} |E(\ell,m)|$$

### Mean Error (should be zero)

$$\mu_E = \frac{1}{N_{\ell} N_m} \sum_{\ell,m} E(\ell,m)$$

By flux conservation (Parseval's theorem), this should be very small.

---

## References

### Textbooks
- Thompson, Moran, Swenson: "Interferometry and Synthesis in Radio Astronomy" (2nd ed.), Chapters 2-4
- Cornwell: "Radio-Interferometric Imaging of Coherent Transients"

### Key Concepts
- **Visibility:** Fourier component of sky brightness
- **UV Coverage:** Sampled points in Fourier domain
- **PSF:** Impulse response of imaging system
- **Dirty Image:** Blurred sky (before deconvolution)
- **Beam Size:** Resolution determined by longest baseline

### Online Resources
- NRAO VLA Documentation
- CASA (Common Astronomy Software Applications)
- Radio astronomy textbooks and papers

---

*Last Updated: December 2025*
