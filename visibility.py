"""
Visibility Generation Module
Compute complex visibilities from sky model
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2, fftfreq, fftshift


class VisibilityGenerator:
    """
    Generate visibilities from sky model using FFT.
    Handles UV sampling and Hermitian symmetry.
    """
    
    def __init__(self, sky_model, vla_array):
        """
        Initialize visibility generator.
        
        Parameters
        ----------
        sky_model : SkyModel
            Sky model instance
        vla_array : VLAConfiguration
            VLA array instance
        """
        self.sky_model = sky_model
        self.vla_array = vla_array
        self.wavelength_m = vla_array.wavelength_m
        self.image_size = sky_model.image_size
        
        # FFT of sky (sampled at all frequencies)
        self.sky_fft = None
        self.visibilities = None
        self.u_samples = None
        self.v_samples = None
        self.baseline_pairs = None
    
    def compute_sky_fft(self):
        """
        Compute FFT of sky brightness distribution.
        
        Returns
        -------
        sky_fft : ndarray
            FFT of sky in Fourier domain
        """
        if np.sum(np.abs(self.sky_model.sky)) == 0:
            self.sky_model.generate_sky()
        
        # FFT with proper normalization
        # Note: FFT of brightness gives visibilities
        self.sky_fft = fft2(self.sky_model.sky) / self.image_size**2
        
        return self.sky_fft
    
    def sample_visibilities(self, hour_angle_deg=0, declination_deg=30):
        """
        Sample visibilities at UV points corresponding to baselines.
        
        Parameters
        ----------
        hour_angle_deg : float
            Hour angle in degrees
        declination_deg : float
            Declination in degrees
        
        Returns
        -------
        visibilities : ndarray
            Complex visibilities, shape (n_baselines,)
        u_samples : ndarray
            U coordinates in wavelengths
        v_samples : ndarray
            V coordinates in wavelengths
        baseline_pairs : ndarray
            Antenna pair indices
        """
        if self.sky_fft is None:
            self.compute_sky_fft()
        
        # Get UV coordinates
        u, v, _, baseline_pairs = self.vla_array.get_uv_coordinates(
            hour_angle_deg, declination_deg
        )
        
        # Create frequency grid for FFT
        # Frequency grid in cycles per pixel
        freq_x = fftfreq(self.image_size)
        freq_y = fftfreq(self.image_size)
        
        # Convert pixel frequencies to wavelengths
        # The pixel scale affects frequency sampling
        pixel_scale_rad = self.sky_model.pixel_scale_rad
        freq_x_wavelengths = freq_x / (self.image_size * pixel_scale_rad / self.wavelength_m)
        freq_y_wavelengths = freq_y / (self.image_size * pixel_scale_rad / self.wavelength_m)
        
        # Sample FFT at UV points
        visibilities = []
        for u_val, v_val in zip(u, v):
            # Find nearest FFT grid point
            idx_u = np.argmin(np.abs(freq_x_wavelengths - u_val))
            idx_v = np.argmin(np.abs(freq_y_wavelengths - v_val))
            
            # Bilinear interpolation for more accurate sampling
            vis = self._interpolate_fft(u_val, v_val, freq_x_wavelengths, 
                                       freq_y_wavelengths, self.sky_fft)
            visibilities.append(vis)
        
        self.visibilities = np.array(visibilities)
        self.u_samples = u
        self.v_samples = v
        self.baseline_pairs = baseline_pairs
        
        return self.visibilities, u, v
    
    def _interpolate_fft(self, u_val, v_val, freq_x, freq_y, fft_grid):
        """
        Bilinear interpolation of FFT at UV point.
        
        Parameters
        ----------
        u_val, v_val : float
            UV coordinates
        freq_x, freq_y : ndarray
            Frequency grids
        fft_grid : ndarray
            FFT of sky
        
        Returns
        -------
        vis : complex
            Interpolated visibility
        """
        # Find indices
        idx_u = np.searchsorted(freq_x, u_val)
        idx_v = np.searchsorted(freq_y, v_val)
        
        # Clamp to valid range
        idx_u = np.clip(idx_u, 0, len(freq_x) - 1)
        idx_v = np.clip(idx_v, 0, len(freq_y) - 1)
        
        # Return nearest neighbor for simplicity
        return fft_grid[idx_v, idx_u]
    
    def enforce_hermitian_symmetry(self):
        """
        Enforce Hermitian symmetry on visibilities.
        V(-u, -v) = conj(V(u, v))
        
        This is done by averaging with conjugates where available.
        """
        if self.visibilities is None:
            raise ValueError("No visibilities to enforce symmetry on")
        
        # For each baseline, find its conjugate
        for i in range(len(self.visibilities)):
            u_i = self.u_samples[i]
            v_i = self.v_samples[i]
            
            # Look for conjugate point
            for j in range(i + 1, len(self.visibilities)):
                u_j = self.u_samples[j]
                v_j = self.v_samples[j]
                
                # Check if (u_j, v_j) ≈ (-u_i, -v_i)
                if np.abs(u_j + u_i) < 1e-6 and np.abs(v_j + v_i) < 1e-6:
                    # Average with conjugate
                    avg = (self.visibilities[i] + np.conj(self.visibilities[j])) / 2
                    self.visibilities[i] = avg
                    self.visibilities[j] = np.conj(avg)
    
    def add_uv_noise(self, snr_db=100):
        """
        Add Gaussian noise in UV domain.
        
        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in dB
        """
        if self.visibilities is None:
            raise ValueError("No visibilities to add noise to")
        
        # Compute signal power
        signal_power = np.mean(np.abs(self.visibilities)**2)
        
        # Convert dB to linear
        snr_linear = 10**(snr_db / 10)
        
        # Noise power
        noise_power = signal_power / snr_linear
        
        # Add complex Gaussian noise
        noise = np.sqrt(noise_power / 2) * (
            np.random.randn(len(self.visibilities)) + 
            1j * np.random.randn(len(self.visibilities))
        )
        
        self.visibilities += noise
    
    def plot_visibility_spectra(self, figsize=(14, 5)):
        """
        Plot visibility amplitude and phase vs baseline length.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, axes : matplotlib objects
        """
        if self.visibilities is None:
            raise ValueError("No visibilities to plot")
        
        # Baseline lengths in wavelengths
        baseline_lengths = np.sqrt(self.u_samples**2 + self.v_samples**2)
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Amplitude vs baseline
        vis_amp = np.abs(self.visibilities)
        axes[0].scatter(baseline_lengths, vis_amp, s=30, alpha=0.6)
        axes[0].set_xlabel('Baseline Length (wavelengths)', fontsize=11)
        axes[0].set_ylabel('Visibility Amplitude (Jy)', fontsize=11)
        axes[0].set_title('Visibility Amplitude Spectrum', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Phase vs baseline
        vis_phase = np.angle(self.visibilities)
        axes[1].scatter(baseline_lengths, vis_phase, s=30, alpha=0.6)
        axes[1].set_xlabel('Baseline Length (wavelengths)', fontsize=11)
        axes[1].set_ylabel('Visibility Phase (radians)', fontsize=11)
        axes[1].set_title('Visibility Phase Spectrum', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_ylim([-np.pi, np.pi])
        
        plt.tight_layout()
        return fig, axes


if __name__ == '__main__':
    from vla_array import VLAConfiguration
    from sky_model import SkyModel
    
    # Setup
    sky = SkyModel(image_size=128, pixel_scale_arcsec=5.0, wavelength_m=0.214)
    sky.add_point_source(0, 0, flux_jy=1.0)
    sky.generate_sky()
    
    vla = VLAConfiguration(mode='A', frequency_ghz=1.4)
    
    # Generate visibilities
    vis_gen = VisibilityGenerator(sky, vla)
    vis, u, v = vis_gen.sample_visibilities()
    
    print(f"Number of visibilities: {len(vis)}")
    print(f"Max visibility: {np.max(np.abs(vis)):.6f} Jy")
    
    fig, axes = vis_gen.plot_visibility_spectra()
    plt.savefig('visibility_spectra.png', dpi=150, bbox_inches='tight')
    plt.show()
