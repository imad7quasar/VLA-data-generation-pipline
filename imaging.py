"""
PSF and Dirty Image Module
Compute point spread function and dirty image from sampled visibilities
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import ifft2, fftshift, fftfreq


class ImagingEngine:
    """
    Compute PSF and dirty image from visibilities using inverse FFT.
    """
    
    def __init__(self, image_size, wavelength_m, pixel_scale_arcsec=1.0):
        """
        Initialize imaging engine.
        
        Parameters
        ----------
        image_size : int
            Output image size (image_size x image_size)
        wavelength_m : float
            Wavelength in meters
        pixel_scale_arcsec : float
            Pixel scale in arcseconds
        """
        self.image_size = image_size
        self.wavelength_m = wavelength_m
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.pixel_scale_rad = pixel_scale_arcsec / 3600.0 * np.pi / 180.0
        
        # Frequency grid (in cycles per pixel)
        self.freq_x = fftfreq(image_size)
        self.freq_y = fftfreq(image_size)
        
        # Convert to UV wavelengths
        uv_scale = 1.0 / (image_size * self.pixel_scale_rad / wavelength_m)
        self.u_grid = self.freq_x / uv_scale
        self.v_grid = self.freq_y / uv_scale
        
        self.psf = None
        self.dirty_image = None
        self.coverage_map = None
    
    def compute_psf(self, u_samples, v_samples):
        """
        Compute PSF as inverse FFT of UV coverage.
        
        Parameters
        ----------
        u_samples : ndarray
            U coordinates of sampled points (wavelengths)
        v_samples : ndarray
            V coordinates of sampled points (wavelengths)
        
        Returns
        -------
        psf : ndarray
            Point spread function
        """
        # Create UV coverage grid (binary mask)
        uv_coverage = np.zeros((self.image_size, self.image_size), dtype=complex)
        
        # Mark sampled points in coverage
        for u, v in zip(u_samples, v_samples):
            # Find nearest grid point
            idx_u = np.argmin(np.abs(self.u_grid - u))
            idx_v = np.argmin(np.abs(self.v_grid - v))
            
            # Add point and conjugate
            if 0 <= idx_u < self.image_size and 0 <= idx_v < self.image_size:
                uv_coverage[idx_v, idx_u] += 1.0
            
            # Add conjugate point
            idx_u_conj = np.argmin(np.abs(self.u_grid + u))
            idx_v_conj = np.argmin(np.abs(self.v_grid + v))
            
            if 0 <= idx_u_conj < self.image_size and 0 <= idx_v_conj < self.image_size:
                uv_coverage[idx_v_conj, idx_u_conj] += 1.0
        
        self.coverage_map = uv_coverage
        
        # Inverse FFT to get PSF
        self.psf = np.real(ifft2(uv_coverage) * self.image_size**2)
        
        # Shift zero to center
        self.psf = fftshift(self.psf)
        
        return self.psf
    
    def compute_dirty_image(self, visibilities, u_samples, v_samples):
        """
        Compute dirty image as inverse FFT of sampled visibilities.
        
        Parameters
        ----------
        visibilities : ndarray
            Complex visibilities
        u_samples : ndarray
            U coordinates (wavelengths)
        v_samples : ndarray
            V coordinates (wavelengths)
        
        Returns
        -------
        dirty_image : ndarray
            Dirty image
        """
        # Create visibility grid
        vis_grid = np.zeros((self.image_size, self.image_size), dtype=complex)
        
        # Place visibilities on grid
        for i, (u, v) in enumerate(zip(u_samples, v_samples)):
            # Find nearest grid point
            idx_u = np.argmin(np.abs(self.u_grid - u))
            idx_v = np.argmin(np.abs(self.v_grid - v))
            
            if 0 <= idx_u < self.image_size and 0 <= idx_v < self.image_size:
                vis_grid[idx_v, idx_u] += visibilities[i]
            
            # Add conjugate
            idx_u_conj = np.argmin(np.abs(self.u_grid + u))
            idx_v_conj = np.argmin(np.abs(self.v_grid + v))
            
            if 0 <= idx_u_conj < self.image_size and 0 <= idx_v_conj < self.image_size:
                vis_grid[idx_v_conj, idx_u_conj] += np.conj(visibilities[i])
        
        # Inverse FFT to get dirty image
        self.dirty_image = np.real(ifft2(vis_grid) * self.image_size**2)
        
        # Shift zero to center
        self.dirty_image = fftshift(self.dirty_image)
        
        return self.dirty_image
    
    def _create_psf_analytical(self):
        """
        Create PSF analytically for testing (simple sinc pattern).
        For a point source with limited UV coverage.
        """
        center = self.image_size // 2
        x = np.arange(self.image_size) - center
        y = np.arange(self.image_size) - center
        xx, yy = np.meshgrid(x, y)
        
        # Sinc pattern for limited UV coverage
        r = np.sqrt(xx**2 + yy**2)
        psf = np.sinc(r / 10.0)
        
        return psf
    
    def plot_psf(self, title='PSF', figsize=(8, 8)):
        """
        Plot PSF with log scale to show sidelobes.
        
        Parameters
        ----------
        title : str
            Plot title
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib objects
        """
        if self.psf is None:
            raise ValueError("PSF not computed")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use log scale to see sidelobes
        psf_to_plot = self.psf / np.max(self.psf)  # Normalize
        
        # Add small offset to avoid log(0)
        psf_to_plot = np.abs(psf_to_plot) + 1e-6
        
        im = ax.imshow(np.log10(psf_to_plot), origin='lower', cmap='viridis',
                       extent=[-self.image_size//2, self.image_size//2,
                               -self.image_size//2, self.image_size//2])
        
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title(f'{title} (log scale)', fontsize=14)
        plt.colorbar(im, ax=ax, label='log10(PSF)')
        
        return fig, ax


class DirtyImageAnalyzer:
    """
    Analyze dirty image and compute error maps.
    """
    
    def __init__(self, dirty_image, sky_image):
        """
        Initialize analyzer.
        
        Parameters
        ----------
        dirty_image : ndarray
            Dirty image from inverse FFT
        sky_image : ndarray
            True sky brightness (complex)
        """
        self.dirty_image = dirty_image
        self.sky_image = np.abs(sky_image)  # Use amplitude for comparison
        self.error_map = None
        self.compute_error()
    
    def compute_error(self):
        """
        Compute error as difference between dirty image and sky.
        """
        # Normalize both to same scale for comparison
        dirty_norm = self.dirty_image / (np.max(np.abs(self.dirty_image)) + 1e-10)
        sky_norm = self.sky_image / (np.max(self.sky_image) + 1e-10)
        
        self.error_map = dirty_norm - sky_norm
    
    def get_statistics(self):
        """
        Get error statistics.
        
        Returns
        -------
        stats : dict
            RMS error, max error, mean error
        """
        return {
            'rms_error': np.sqrt(np.mean(self.error_map**2)),
            'max_error': np.max(np.abs(self.error_map)),
            'mean_error': np.mean(self.error_map),
        }
    
    def plot_error_map(self, title='Error Map', figsize=(8, 8)):
        """
        Plot error map.
        
        Parameters
        ----------
        title : str
            Plot title
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib objects
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Use diverging colormap for error
        from matplotlib.colors import SymLogNorm
        
        norm = SymLogNorm(linthresh=1e-3, vmin=-np.max(np.abs(self.error_map)),
                         vmax=np.max(np.abs(self.error_map)))
        
        im = ax.imshow(self.error_map, origin='lower', cmap='RdBu_r', norm=norm,
                       extent=[-self.dirty_image.shape[0]//2, self.dirty_image.shape[0]//2,
                               -self.dirty_image.shape[1]//2, self.dirty_image.shape[1]//2])
        
        ax.set_xlabel('X (pixels)', fontsize=12)
        ax.set_ylabel('Y (pixels)', fontsize=12)
        ax.set_title(title, fontsize=14)
        plt.colorbar(im, ax=ax, label='Error')
        
        return fig, ax


if __name__ == '__main__':
    from vla_array import VLAConfiguration
    from sky_model import SkyModel
    from visibility import VisibilityGenerator
    
    # Setup
    sky = SkyModel(image_size=128, pixel_scale_arcsec=5.0, wavelength_m=0.214)
    sky.add_point_source(0, 0, flux_jy=1.0)
    sky.generate_sky()
    
    vla = VLAConfiguration(mode='A', frequency_ghz=1.4)
    
    # Generate visibilities
    vis_gen = VisibilityGenerator(sky, vla)
    vis, u, v = vis_gen.sample_visibilities()
    
    # Create images
    engine = ImagingEngine(image_size=128, wavelength_m=0.214, pixel_scale_arcsec=5.0)
    psf = engine.compute_psf(u, v)
    dirty = engine.compute_dirty_image(vis, u, v)
    
    print(f"PSF max: {np.max(psf):.6f}")
    print(f"Dirty image max: {np.max(dirty):.6f}")
    
    fig, ax = engine.plot_psf()
    plt.savefig('psf.png', dpi=150, bbox_inches='tight')
    plt.show()
