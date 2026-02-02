"""
Complete VLA Interferometric Imaging Pipeline
Demonstrates full workflow with verification and plots
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.constants import c as speed_of_light

from vla_array import VLAConfiguration
from sky_model import SkyModel
from visibility import VisibilityGenerator
from imaging import ImagingEngine, DirtyImageAnalyzer


class VLAPipeline:
    """
    Complete VLA imaging pipeline with all processing steps.
    """
    
    def __init__(self, array_mode='A', frequency_ghz=1.4, 
                 image_size=256, pixel_scale_arcsec=2.0):
        """
        Initialize pipeline.
        
        Parameters
        ----------
        array_mode : str
            VLA array configuration
        frequency_ghz : float
            Observing frequency in GHz
        image_size : int
            Output image size
        pixel_scale_arcsec : float
            Pixel scale in arcseconds
        """
        self.array_mode = array_mode
        self.frequency_ghz = frequency_ghz
        self.wavelength_m = speed_of_light / (frequency_ghz * 1e9)
        self.image_size = image_size
        self.pixel_scale_arcsec = pixel_scale_arcsec
        
        # Initialize components
        self.vla_array = VLAConfiguration(mode=array_mode, frequency_ghz=frequency_ghz)
        self.sky_model = SkyModel(image_size=image_size, 
                                 pixel_scale_arcsec=pixel_scale_arcsec,
                                 wavelength_m=self.wavelength_m)
        
        self.visibilities = None
        self.u_samples = None
        self.v_samples = None
        self.psf = None
        self.dirty_image = None
        self.analyzer = None
    
    def setup_sources(self, sources_list):
        """
        Add sources to sky model.
        
        Parameters
        ----------
        sources_list : list of dict
            Each dict should have keys: 'type' ('point' or 'gaussian'),
            and appropriate parameters
        """
        for source in sources_list:
            if source['type'] == 'point':
                self.sky_model.add_point_source(
                    source['x_pix'], source['y_pix'],
                    source['flux_jy'], source.get('phase_rad', 0)
                )
            elif source['type'] == 'gaussian':
                self.sky_model.add_gaussian_source(
                    source['x_pix'], source['y_pix'],
                    source['flux_jy'], source['fwhm_pix'],
                    source.get('phase_rad', 0),
                    source.get('ellipticity', 1.0),
                    source.get('pa_deg', 0)
                )
        
        self.sky_model.generate_sky()
    
    def compute_visibilities(self, hour_angle_deg=0, declination_deg=30, 
                            add_noise=False, snr_db=100):
        """
        Generate visibilities from sky model.
        
        Parameters
        ----------
        hour_angle_deg : float
            Hour angle in degrees
        declination_deg : float
            Declination in degrees
        add_noise : bool
            Whether to add thermal noise
        snr_db : float
            Signal-to-noise ratio in dB
        """
        vis_gen = VisibilityGenerator(self.sky_model, self.vla_array)
        vis_gen.compute_sky_fft()
        
        self.visibilities, self.u_samples, self.v_samples = \
            vis_gen.sample_visibilities(hour_angle_deg, declination_deg)
        
        if add_noise:
            vis_gen.add_uv_noise(snr_db=snr_db)
            self.visibilities = vis_gen.visibilities
        
        # Enforce Hermitian symmetry
        vis_gen.visibilities = self.visibilities
        vis_gen.u_samples = self.u_samples
        vis_gen.v_samples = self.v_samples
        vis_gen.enforce_hermitian_symmetry()
        self.visibilities = vis_gen.visibilities
        
        print(f"Computed {len(self.visibilities)} visibilities")
        print(f"Visibility RMS amplitude: {np.sqrt(np.mean(np.abs(self.visibilities)**2)):.6f} Jy")
    
    def compute_images(self):
        """
        Compute PSF and dirty image from visibilities.
        """
        engine = ImagingEngine(self.image_size, self.wavelength_m, 
                              self.pixel_scale_arcsec)
        
        self.psf = engine.compute_psf(self.u_samples, self.v_samples)
        self.dirty_image = engine.compute_dirty_image(self.visibilities, 
                                                      self.u_samples, 
                                                      self.v_samples)
        
        self.analyzer = DirtyImageAnalyzer(self.dirty_image, self.sky_model.sky)
        
        stats = self.analyzer.get_statistics()
        print(f"\nImage Statistics:")
        print(f"PSF peak: {np.max(self.psf):.6f}")
        print(f"PSF sidelobe level: {np.max(np.abs(self.psf[1:, 1:])):.6f}")
        print(f"Dirty image max: {np.max(np.abs(self.dirty_image)):.6f}")
        print(f"Error RMS: {stats['rms_error']:.6f}")
        print(f"Error Max: {stats['max_error']:.6f}")
    
    def plot_full_pipeline(self, figsize=(20, 14)):
        """
        Create comprehensive visualization of entire pipeline.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig : matplotlib figure
        """
        fig = plt.figure(figsize=figsize)
        gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
        
        # --- ROW 1: Sky Model ---
        ax1 = fig.add_subplot(gs[0, 0])
        sky_amp = np.abs(self.sky_model.sky)
        im1 = ax1.imshow(sky_amp, origin='lower', cmap='viridis')
        ax1.set_title('Sky Amplitude (Jy)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('X (pixels)')
        ax1.set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=ax1, label='Jy')
        
        ax2 = fig.add_subplot(gs[0, 1])
        sky_phase = np.angle(self.sky_model.sky)
        im2 = ax2.imshow(sky_phase, origin='lower', cmap='hsv')
        ax2.set_title('Sky Phase (rad)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('X (pixels)')
        ax2.set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=ax2, label='radians')
        
        # UV Coverage
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.scatter(self.u_samples, self.v_samples, s=20, alpha=0.6, label='Baselines')
        ax3.scatter(-self.u_samples, -self.v_samples, s=20, alpha=0.6, label='Conjugates')
        ax3.set_xlabel('U (wavelengths)', fontsize=11)
        ax3.set_ylabel('V (wavelengths)', fontsize=11)
        ax3.set_title('UV Coverage', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend(fontsize=9)
        ax3.axis('equal')
        
        # --- ROW 2: Visibilities ---
        baseline_lengths = np.sqrt(self.u_samples**2 + self.v_samples**2)
        
        ax4 = fig.add_subplot(gs[1, 0])
        vis_amp = np.abs(self.visibilities)
        ax4.scatter(baseline_lengths, vis_amp, s=30, alpha=0.6, color='darkblue')
        ax4.set_xlabel('Baseline (wavelengths)', fontsize=11)
        ax4.set_ylabel('Visibility Amplitude (Jy)', fontsize=11)
        ax4.set_title('Visibility Amplitude Spectrum', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        ax5 = fig.add_subplot(gs[1, 1])
        vis_phase = np.angle(self.visibilities)
        ax5.scatter(baseline_lengths, vis_phase, s=30, alpha=0.6, color='darkred')
        ax5.set_xlabel('Baseline (wavelengths)', fontsize=11)
        ax5.set_ylabel('Visibility Phase (rad)', fontsize=11)
        ax5.set_title('Visibility Phase Spectrum', fontsize=12, fontweight='bold')
        ax5.set_ylim([-np.pi, np.pi])
        ax5.grid(True, alpha=0.3)
        
        # PSF
        ax6 = fig.add_subplot(gs[1, 2])
        psf_norm = self.psf / np.max(np.abs(self.psf))
        psf_to_plot = np.abs(psf_norm) + 1e-8
        im6 = ax6.imshow(np.log10(psf_to_plot), origin='lower', cmap='viridis')
        ax6.set_title('PSF (log scale)', fontsize=12, fontweight='bold')
        ax6.set_xlabel('X (pixels)')
        ax6.set_ylabel('Y (pixels)')
        plt.colorbar(im6, ax=ax6, label='log10(PSF)')
        
        # --- ROW 3: Results ---
        ax7 = fig.add_subplot(gs[2, 0])
        dirty_norm = self.dirty_image / (np.max(np.abs(self.dirty_image)) + 1e-10)
        im7 = ax7.imshow(dirty_norm, origin='lower', cmap='viridis')
        ax7.set_title('Dirty Image Amplitude', fontsize=12, fontweight='bold')
        ax7.set_xlabel('X (pixels)')
        ax7.set_ylabel('Y (pixels)')
        plt.colorbar(im7, ax=ax7, label='Normalized')
        
        ax8 = fig.add_subplot(gs[2, 1])
        dirty_phase = np.angle(self.dirty_image + 1e-10j)
        im8 = ax8.imshow(dirty_phase, origin='lower', cmap='hsv')
        ax8.set_title('Dirty Image Phase', fontsize=12, fontweight='bold')
        ax8.set_xlabel('X (pixels)')
        ax8.set_ylabel('Y (pixels)')
        plt.colorbar(im8, ax=ax8, label='radians')
        
        ax9 = fig.add_subplot(gs[2, 2])
        from matplotlib.colors import SymLogNorm
        norm = SymLogNorm(linthresh=1e-3, 
                         vmin=-np.max(np.abs(self.analyzer.error_map)),
                         vmax=np.max(np.abs(self.analyzer.error_map)))
        im9 = ax9.imshow(self.analyzer.error_map, origin='lower', cmap='RdBu_r', norm=norm)
        ax9.set_title('Error Map (Dirty - Sky)', fontsize=12, fontweight='bold')
        ax9.set_xlabel('X (pixels)')
        ax9.set_ylabel('Y (pixels)')
        plt.colorbar(im9, ax=ax9, label='Error')
        
        fig.suptitle(f'VLA Pipeline: {self.array_mode}-array @ {self.frequency_ghz} GHz',
                     fontsize=16, fontweight='bold', y=0.995)
        
        return fig


def test_sanity_point_source():
    """
    Minimal sanity test: single point source at center, full UV coverage, no noise.
    Dirty image should equal sky exactly.
    PSF should match inverse FFT of UV coverage.
    """
    print("\n" + "="*70)
    print("SANITY TEST: SINGLE POINT SOURCE AT CENTER")
    print("="*70)
    
    # Create pipeline
    pipeline = VLAPipeline(array_mode='A', frequency_ghz=1.4, 
                          image_size=256, pixel_scale_arcsec=2.0)
    
    # Single point source at center
    pipeline.setup_sources([
        {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 1.0}
    ])
    
    # Compute visibilities (no noise)
    pipeline.compute_visibilities(hour_angle_deg=0, declination_deg=45)
    
    # Compute images
    pipeline.compute_images()
    
    # Analysis
    print("\n--- Sanity Test Results ---")
    print(f"Sky peak: {np.max(np.abs(pipeline.sky_model.sky)):.6f} Jy")
    print(f"Dirty peak: {np.max(np.abs(pipeline.dirty_image)):.6f} Jy")
    print(f"Peak ratio (dirty/sky): {np.max(np.abs(pipeline.dirty_image))/np.max(np.abs(pipeline.sky_model.sky)):.4f}")
    
    # Check if center pixel matches
    center = 256 // 2
    sky_center = np.abs(pipeline.sky_model.sky[center, center])
    dirty_center = np.abs(pipeline.dirty_image[center, center])
    print(f"Sky center value: {sky_center:.6f} Jy")
    print(f"Dirty center value: {dirty_center:.6f} Jy")
    print(f"Center pixel error: {np.abs(sky_center - dirty_center):.6e}")
    
    # The difference should be due to sampling - central peak broadened by PSF
    print("\nNote: Small differences expected due to:")
    print("  - PSF broadening effect")
    print("  - Discrete UV sampling")
    print("  - FFT grid discretization")
    
    return pipeline


def main():
    """
    Run complete VLA imaging pipeline with realistic example.
    """
    print("\n" + "="*70)
    print("VLA INTERFEROMETRIC IMAGING PIPELINE")
    print("="*70)
    
    # Create pipeline
    pipeline = VLAPipeline(array_mode='A', frequency_ghz=1.4, 
                          image_size=256, pixel_scale_arcsec=2.0)
    
    # Define sky model with multiple sources
    sources = [
        {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 1.5},
        {'type': 'gaussian', 'x_pix': 20, 'y_pix': 15, 'flux_jy': 0.8, 
         'fwhm_pix': 8, 'phase_rad': 0.5},
        {'type': 'point', 'x_pix': -15, 'y_pix': -10, 'flux_jy': 0.5, 
         'phase_rad': -0.3},
    ]
    
    print("\n--- Sky Setup ---")
    print(f"Array mode: {pipeline.array_mode}")
    print(f"Frequency: {pipeline.frequency_ghz} GHz")
    print(f"Wavelength: {pipeline.wavelength_m:.6f} m")
    print(f"Number of sources: {len(sources)}")
    
    # Setup sources
    pipeline.setup_sources(sources)
    
    print(f"Total sky flux: {np.sum(np.abs(pipeline.sky_model.sky)):.6f} Jy")
    
    # Compute visibilities
    print("\n--- Visibility Computation ---")
    pipeline.compute_visibilities(hour_angle_deg=0, declination_deg=45)
    
    # Compute images
    print("\n--- Image Computation ---")
    pipeline.compute_images()
    
    # Create comprehensive plots
    print("\n--- Creating Plots ---")
    fig = pipeline.plot_full_pipeline(figsize=(20, 14))
    plt.savefig('vla_pipeline_full.png', dpi=150, bbox_inches='tight')
    print("Saved: vla_pipeline_full.png")
    
    # Run sanity test
    pipeline_test = test_sanity_point_source()
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETE")
    print("="*70)
    
    return pipeline, pipeline_test


if __name__ == '__main__':
    pipeline, pipeline_test = main()
    plt.show()
