"""
Sky Model Module
Define source positions and brightness distributions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm


class SkyModel:
    """
    2D sky model with Gaussian and point sources.
    Image plane in pixel units consistent with wavelength.
    """
    
    def __init__(self, image_size, pixel_scale_arcsec=1.0, wavelength_m=0.214):
        """
        Initialize sky model.
        
        Parameters
        ----------
        image_size : int
            Image dimensions (image_size x image_size pixels)
        pixel_scale_arcsec : float
            Pixel scale in arcseconds
        wavelength_m : float
            Wavelength in meters
        """
        self.image_size = image_size
        self.pixel_scale_arcsec = pixel_scale_arcsec
        self.wavelength_m = wavelength_m
        
        # Pixel scale in radians
        self.pixel_scale_rad = pixel_scale_arcsec / 3600.0 * np.pi / 180.0
        
        # Create coordinate grids (centered at origin)
        center = image_size // 2
        x = np.arange(image_size) - center
        y = np.arange(image_size) - center
        self.xx, self.yy = np.meshgrid(x, y)
        
        # Distance from center in radians
        self.rr = np.sqrt(self.xx**2 + self.yy**2) * self.pixel_scale_rad
        
        self.sources = []
        self.sky = np.zeros((image_size, image_size), dtype=complex)
    
    def add_point_source(self, x_offset_pix, y_offset_pix, flux_jy, phase_rad=0):
        """
        Add a point source.
        
        Parameters
        ----------
        x_offset_pix : float
            X offset in pixels from center
        y_offset_pix : float
            Y offset in pixels from center
        flux_jy : float
            Flux in Jy
        phase_rad : float
            Phase in radians
        """
        self.sources.append({
            'type': 'point',
            'x_pix': x_offset_pix,
            'y_pix': y_offset_pix,
            'flux': flux_jy,
            'phase': phase_rad,
        })
    
    def add_gaussian_source(self, x_offset_pix, y_offset_pix, flux_jy, 
                           fwhm_pix, phase_rad=0, ellipticity=1.0, pa_deg=0):
        """
        Add a Gaussian source.
        
        Parameters
        ----------
        x_offset_pix : float
            X offset in pixels from center
        y_offset_pix : float
            Y offset in pixels from center
        flux_jy : float
            Peak flux in Jy
        fwhm_pix : float
            FWHM in pixels
        phase_rad : float
            Phase in radians
        ellipticity : float
            Ratio of minor to major axis
        pa_deg : float
            Position angle in degrees
        """
        self.sources.append({
            'type': 'gaussian',
            'x_pix': x_offset_pix,
            'y_pix': y_offset_pix,
            'flux': flux_jy,
            'fwhm': fwhm_pix,
            'phase': phase_rad,
            'ellipticity': ellipticity,
            'pa': pa_deg,
        })
    
    def generate_sky(self):
        """
        Generate the 2D sky image.
        
        Returns
        -------
        sky : ndarray
            Complex sky brightness, shape (image_size, image_size)
        """
        self.sky = np.zeros((self.image_size, self.image_size), dtype=complex)
        
        for source in self.sources:
            if source['type'] == 'point':
                # Point source at specified location
                x_pix = source['x_pix']
                y_pix = source['y_pix']
                
                # Find nearest pixel
                x_idx = int(np.round(x_pix))
                y_idx = int(np.round(y_pix))
                
                center = self.image_size // 2
                x_idx += center
                y_idx += center
                
                if 0 <= x_idx < self.image_size and 0 <= y_idx < self.image_size:
                    amplitude = source['flux']
                    phase = source['phase']
                    self.sky[y_idx, x_idx] += amplitude * np.exp(1j * phase)
            
            elif source['type'] == 'gaussian':
                # Gaussian source
                x_pix = source['x_pix']
                y_pix = source['y_pix']
                center = self.image_size // 2
                
                # Relative coordinates
                dx = (self.xx - x_pix) * self.pixel_scale_rad
                dy = (self.yy - y_pix) * self.pixel_scale_rad
                
                # FWHM to sigma conversion
                sigma = source['fwhm'] * self.pixel_scale_rad / (2.355)
                
                # Elliptical Gaussian with rotation
                pa = np.radians(source['pa'])
                cos_pa = np.cos(pa)
                sin_pa = np.sin(pa)
                e = source['ellipticity']
                
                # Rotate coordinates
                dx_rot = dx * cos_pa + dy * sin_pa
                dy_rot = -dx * sin_pa + dy * cos_pa
                
                # Gaussian profile
                gauss = np.exp(-(dx_rot**2 / (2 * sigma**2) + 
                                dy_rot**2 / (2 * (e * sigma)**2)))
                
                amplitude = source['flux']
                phase = source['phase']
                self.sky += amplitude * gauss * np.exp(1j * phase)
        
        return self.sky
    
    def plot_sky(self, title='Sky Model', figsize=(12, 5)):
        """
        Plot sky amplitude and phase.
        
        Parameters
        ----------
        title : str
            Plot title
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, axes : matplotlib objects
        """
        if np.sum(np.abs(self.sky)) == 0:
            self.generate_sky()
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Amplitude
        sky_amp = np.abs(self.sky)
        im0 = axes[0].imshow(sky_amp, origin='lower', cmap='viridis', 
                             extent=[-self.image_size//2, self.image_size//2,
                                     -self.image_size//2, self.image_size//2])
        axes[0].set_xlabel('X (pixels)', fontsize=11)
        axes[0].set_ylabel('Y (pixels)', fontsize=11)
        axes[0].set_title(f'{title} - Amplitude (Jy)', fontsize=12)
        plt.colorbar(im0, ax=axes[0], label='Jy')
        
        # Phase
        sky_phase = np.angle(self.sky)
        im1 = axes[1].imshow(sky_phase, origin='lower', cmap='hsv',
                             extent=[-self.image_size//2, self.image_size//2,
                                     -self.image_size//2, self.image_size//2])
        axes[1].set_xlabel('X (pixels)', fontsize=11)
        axes[1].set_ylabel('Y (pixels)', fontsize=11)
        axes[1].set_title(f'{title} - Phase (radians)', fontsize=12)
        plt.colorbar(im1, ax=axes[1], label='radians')
        
        plt.tight_layout()
        return fig, axes


if __name__ == '__main__':
    # Test sky model
    sky = SkyModel(image_size=128, pixel_scale_arcsec=5.0, wavelength_m=0.214)
    
    # Add sources
    sky.add_point_source(0, 0, flux_jy=1.0)
    sky.add_gaussian_source(10, 0, flux_jy=0.5, fwhm_pix=5)
    
    sky.generate_sky()
    fig, axes = sky.plot_sky()
    
    plt.savefig('sky_model.png', dpi=150, bbox_inches='tight')
    plt.show()
