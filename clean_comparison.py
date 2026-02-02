"""
Dirty Image vs Clean Image Comparison
Generates comparison visualization for machine learning training data
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy.ndimage import gaussian_filter

from vla_array import VLAConfiguration
from sky_model import SkyModel
from visibility import VisibilityGenerator
from imaging import ImagingEngine


def create_clean_image_wiener(dirty_image, psf, noise_level=0.01):
    """
    Simple Wiener filter restoration.
    Clean_image ≈ Dirty_image - PSF * source
    
    Parameters
    ----------
    dirty_image : ndarray
        Dirty image (observation)
    psf : ndarray
        Point spread function
    noise_level : float
        Noise level estimate
    
    Returns
    -------
    clean : ndarray
        Restored image
    """
    from scipy.fft import fft2, ifft2
    
    # FFT domain filtering
    dirty_fft = fft2(dirty_image)
    psf_fft = fft2(psf)
    
    # Normalize PSF
    psf_fft_norm = psf_fft / (np.max(np.abs(psf_fft)) + 1e-10)
    
    # Wiener filter
    wiener = np.conj(psf_fft_norm) / (np.abs(psf_fft_norm)**2 + noise_level)
    
    # Apply filter
    clean_fft = dirty_fft * wiener
    clean = np.real(ifft2(clean_fft))
    
    return clean


def generate_comparison_dataset(output_dir='testing'):
    """
    Generate dirty vs clean image pairs for testing/training.
    
    Parameters
    ----------
    output_dir : str
        Output directory for images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("GENERATING DIRTY vs CLEAN IMAGE COMPARISON")
    print("="*70)
    
    # Create pipeline
    pipeline = VLAConfiguration(mode='A', frequency_ghz=1.4)
    
    # Sky model - this will be our "truth"
    sky = SkyModel(image_size=256, pixel_scale_arcsec=2.0, wavelength_m=0.214)
    
    # Add realistic sources
    sources = [
        # Bright point source
        {'type': 'point', 'x_pix': -30, 'y_pix': 20, 'flux_jy': 3.0, 'phase_rad': 0.5},
        # Extended Gaussian
        {'type': 'gaussian', 'x_pix': 20, 'y_pix': 0, 'flux_jy': 2.5, 
         'fwhm_pix': 12, 'phase_rad': -0.3},
        # Weaker point source
        {'type': 'point', 'x_pix': 30, 'y_pix': 25, 'flux_jy': 0.8, 'phase_rad': 0.1},
        # Small Gaussian
        {'type': 'gaussian', 'x_pix': -20, 'y_pix': -30, 'flux_jy': 1.2, 
         'fwhm_pix': 5, 'phase_rad': 0.0},
    ]
    
    sky.generate_sky()
    for source in sources:
        if source['type'] == 'point':
            sky.add_point_source(
                source['x_pix'], source['y_pix'],
                source['flux_jy'], source['phase_rad']
            )
        else:
            sky.add_gaussian_source(
                source['x_pix'], source['y_pix'],
                source['flux_jy'], source['fwhm_pix'],
                source['phase_rad']
            )
    
    sky.generate_sky()
    print(f"✓ Sky model created with {len(sources)} sources")
    print(f"  Total flux: {np.sum(np.abs(sky.sky)):.2f} Jy")
    
    # Generate visibilities
    vis_gen = VisibilityGenerator(sky, pipeline)
    vis_gen.compute_sky_fft()
    vis, u, v = vis_gen.sample_visibilities(declination_deg=45)
    
    # Add realistic noise
    vis_gen.add_uv_noise(snr_db=40)  # Moderate noise
    
    print(f"✓ Visibilities computed with SNR=40dB")
    
    # Create dirty image
    engine = ImagingEngine(image_size=256, wavelength_m=0.214, pixel_scale_arcsec=2.0)
    psf = engine.compute_psf(u, v)
    dirty_image = engine.compute_dirty_image(vis_gen.visibilities, u, v)
    
    print(f"✓ Dirty image created")
    print(f"  Peak brightness: {np.max(np.abs(dirty_image)):.4f} Jy")
    
    # Create clean image using Wiener filter
    clean_image = create_clean_image_wiener(dirty_image, psf, noise_level=0.01)
    
    # Post-processing: Gaussian smoothing for better appearance
    clean_image = gaussian_filter(clean_image, sigma=1.0)
    
    print(f"✓ Clean image created (Wiener restoration)")
    print(f"  Peak brightness: {np.max(clean_image):.4f} Jy")
    
    # Create comprehensive comparison figure
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)
    
    # Normalize for visualization
    dirty_norm = np.abs(dirty_image) / (np.max(np.abs(dirty_image)) + 1e-10)
    clean_norm = np.abs(clean_image) / (np.max(np.abs(clean_image)) + 1e-10)
    sky_norm = np.abs(sky.sky) / (np.max(np.abs(sky.sky)) + 1e-10)
    
    # --- TOP ROW: AMPLITUDE ---
    # Dirty amplitude
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dirty_norm, origin='lower', cmap='viridis',
                     extent=[-128, 128, -128, 128])
    ax1.set_title('DIRTY IMAGE (Input)\nAmplitude', fontsize=13, fontweight='bold', color='darkred')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    plt.colorbar(im1, ax=ax1, label='Normalized')
    
    # Clean amplitude
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(clean_norm, origin='lower', cmap='viridis',
                     extent=[-128, 128, -128, 128])
    ax2.set_title('CLEAN IMAGE (Output)\nAmplitude', fontsize=13, fontweight='bold', color='darkgreen')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    plt.colorbar(im2, ax=ax2, label='Normalized')
    
    # Sky/Truth amplitude
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(sky_norm, origin='lower', cmap='viridis',
                     extent=[-128, 128, -128, 128])
    ax3.set_title('TRUE SKY (Ground Truth)\nAmplitude', fontsize=13, fontweight='bold', color='darkblue')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    plt.colorbar(im3, ax=ax3, label='Normalized')
    
    # --- MIDDLE ROW: PHASE ---
    # Dirty phase
    ax4 = fig.add_subplot(gs[1, 0])
    dirty_phase = np.angle(dirty_image)
    im4 = ax4.imshow(dirty_phase, origin='lower', cmap='hsv',
                     extent=[-128, 128, -128, 128])
    ax4.set_title('DIRTY IMAGE\nPhase', fontsize=13, fontweight='bold', color='darkred')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    plt.colorbar(im4, ax=ax4, label='Radians')
    
    # Clean phase
    ax5 = fig.add_subplot(gs[1, 1])
    clean_phase = np.angle(clean_image)
    im5 = ax5.imshow(clean_phase, origin='lower', cmap='hsv',
                     extent=[-128, 128, -128, 128])
    ax5.set_title('CLEAN IMAGE\nPhase', fontsize=13, fontweight='bold', color='darkgreen')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    plt.colorbar(im5, ax=ax5, label='Radians')
    
    # Sky phase
    ax6 = fig.add_subplot(gs[1, 2])
    sky_phase = np.angle(sky.sky)
    im6 = ax6.imshow(sky_phase, origin='lower', cmap='hsv',
                     extent=[-128, 128, -128, 128])
    ax6.set_title('TRUE SKY\nPhase', fontsize=13, fontweight='bold', color='darkblue')
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    plt.colorbar(im6, ax=ax6, label='Radians')
    
    # --- BOTTOM ROW: ERROR/RESIDUALS ---
    # Dirty error
    ax7 = fig.add_subplot(gs[2, 0])
    dirty_error = dirty_norm - sky_norm
    from matplotlib.colors import SymLogNorm
    norm_error = SymLogNorm(linthresh=0.01, vmin=-np.max(np.abs(dirty_error)),
                           vmax=np.max(np.abs(dirty_error)))
    im7 = ax7.imshow(dirty_error, origin='lower', cmap='RdBu_r', norm=norm_error,
                     extent=[-128, 128, -128, 128])
    ax7.set_title('DIRTY - TRUE\nError', fontsize=13, fontweight='bold', color='darkred')
    ax7.set_xlabel('X (pixels)')
    ax7.set_ylabel('Y (pixels)')
    plt.colorbar(im7, ax=ax7, label='Error')
    
    # Clean error
    ax8 = fig.add_subplot(gs[2, 1])
    clean_error = clean_norm - sky_norm
    norm_error2 = SymLogNorm(linthresh=0.01, vmin=-np.max(np.abs(clean_error)),
                            vmax=np.max(np.abs(clean_error)))
    im8 = ax8.imshow(clean_error, origin='lower', cmap='RdBu_r', norm=norm_error2,
                     extent=[-128, 128, -128, 128])
    ax8.set_title('CLEAN - TRUE\nError', fontsize=13, fontweight='bold', color='darkgreen')
    ax8.set_xlabel('X (pixels)')
    ax8.set_ylabel('Y (pixels)')
    plt.colorbar(im8, ax=ax8, label='Error')
    
    # PSF
    ax9 = fig.add_subplot(gs[2, 2])
    psf_norm = psf / np.max(np.abs(psf))
    psf_to_plot = np.abs(psf_norm) + 1e-8
    im9 = ax9.imshow(np.log10(psf_to_plot), origin='lower', cmap='viridis',
                     extent=[-128, 128, -128, 128])
    ax9.set_title('PSF (log scale)\n', fontsize=13, fontweight='bold')
    ax9.set_xlabel('X (pixels)')
    ax9.set_ylabel('Y (pixels)')
    plt.colorbar(im9, ax=ax9, label='log10(PSF)')
    
    # Overall title
    fig.suptitle('DIRTY vs CLEAN IMAGE COMPARISON\nDirty (Input) → Clean (Output) with Ground Truth',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = f'{output_dir}/dirty_vs_clean_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.close()
    
    # Create additional detailed comparison
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # Cross sections
    center = 256 // 2
    x = np.arange(256) - center
    
    # Horizontal profiles
    dirty_horiz = dirty_norm[center, :]
    clean_horiz = clean_norm[center, :]
    sky_horiz = sky_norm[center, :]
    
    axes[0, 0].plot(x, dirty_horiz, 'r-', linewidth=2, label='Dirty', alpha=0.8)
    axes[0, 0].plot(x, clean_horiz, 'g-', linewidth=2, label='Clean', alpha=0.8)
    axes[0, 0].plot(x, sky_horiz, 'b--', linewidth=2, label='Truth', alpha=0.8)
    axes[0, 0].set_xlabel('X (pixels)')
    axes[0, 0].set_ylabel('Brightness (normalized)')
    axes[0, 0].set_title('Horizontal Cross-Section (Y=0)', fontweight='bold')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Vertical profiles
    dirty_vert = dirty_norm[:, center]
    clean_vert = clean_norm[:, center]
    sky_vert = sky_norm[:, center]
    
    axes[0, 1].plot(x, dirty_vert, 'r-', linewidth=2, label='Dirty', alpha=0.8)
    axes[0, 1].plot(x, clean_vert, 'g-', linewidth=2, label='Clean', alpha=0.8)
    axes[0, 1].plot(x, sky_vert, 'b--', linewidth=2, label='Truth', alpha=0.8)
    axes[0, 1].set_xlabel('Y (pixels)')
    axes[0, 1].set_ylabel('Brightness (normalized)')
    axes[0, 1].set_title('Vertical Cross-Section (X=0)', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Radial profiles
    y_idx, x_idx = np.ogrid[:256, :256]
    r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2)
    
    r_bins = np.arange(0, 150, 2)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    profiles = {'Dirty': dirty_norm, 'Clean': clean_norm, 'Truth': sky_norm}
    colors = {'Dirty': 'red', 'Clean': 'green', 'Truth': 'blue'}
    
    for name, profile in profiles.items():
        r_profile = []
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i+1])
            if np.sum(mask) > 0:
                r_profile.append(np.mean(profile[mask]))
            else:
                r_profile.append(0)
        
        if name == 'Truth':
            axes[0, 2].plot(r_centers, r_profile, color=colors[name], 
                           linewidth=2, label=name, linestyle='--', alpha=0.8)
        else:
            axes[0, 2].plot(r_centers, r_profile, color=colors[name], 
                           linewidth=2, label=name, alpha=0.8)
    
    axes[0, 2].set_xlabel('Radius (pixels)')
    axes[0, 2].set_ylabel('Mean Brightness (normalized)')
    axes[0, 2].set_title('Radial Profile', fontweight='bold')
    axes[0, 2].legend()
    axes[0, 2].grid(True, alpha=0.3)
    
    # Error metrics
    dirty_mse = np.mean((dirty_norm - sky_norm)**2)
    clean_mse = np.mean((clean_norm - sky_norm)**2)
    dirty_mae = np.mean(np.abs(dirty_norm - sky_norm))
    clean_mae = np.mean(np.abs(clean_norm - sky_norm))
    
    metrics_text = f"""
    ERROR METRICS
    ─────────────────────
    Dirty Image:
      MSE:  {dirty_mse:.6f}
      MAE:  {dirty_mae:.6f}
    
    Clean Image:
      MSE:  {clean_mse:.6f}
      MAE:  {clean_mae:.6f}
    
    Improvement:
      MSE:  {(1-clean_mse/dirty_mse)*100:.1f}%
      MAE:  {(1-clean_mae/dirty_mae)*100:.1f}%
    """
    
    axes[1, 0].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                   transform=axes[1, 0].transAxes, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    axes[1, 0].axis('off')
    
    # Peak comparisons
    dirty_peak = np.max(dirty_norm)
    clean_peak = np.max(clean_norm)
    sky_peak = np.max(sky_norm)
    
    peaks = [dirty_peak, clean_peak, sky_peak]
    labels = ['Dirty', 'Clean', 'Truth']
    colors_bar = ['red', 'green', 'blue']
    
    axes[1, 1].bar(labels, peaks, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Peak Brightness (normalized)')
    axes[1, 1].set_title('Peak Value Comparison', fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for i, (label, peak) in enumerate(zip(labels, peaks)):
        axes[1, 1].text(i, peak + 0.02, f'{peak:.4f}', ha='center', fontweight='bold')
    
    # Spectrum comparison
    from scipy.fft import fft2
    dirty_spectrum = np.abs(fft2(dirty_norm))
    clean_spectrum = np.abs(fft2(clean_norm))
    sky_spectrum = np.abs(fft2(sky_norm))
    
    axes[1, 2].loglog(range(1, 129), np.sort(dirty_spectrum.flatten())[-128:][::-1], 
                     'r-', linewidth=2, label='Dirty', alpha=0.8)
    axes[1, 2].loglog(range(1, 129), np.sort(clean_spectrum.flatten())[-128:][::-1], 
                     'g-', linewidth=2, label='Clean', alpha=0.8)
    axes[1, 2].loglog(range(1, 129), np.sort(sky_spectrum.flatten())[-128:][::-1], 
                     'b--', linewidth=2, label='Truth', alpha=0.8)
    axes[1, 2].set_xlabel('Frequency Component (sorted)')
    axes[1, 2].set_ylabel('Magnitude (log scale)')
    axes[1, 2].set_title('Fourier Spectrum Comparison', fontweight='bold')
    axes[1, 2].legend()
    axes[1, 2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_path2 = f'{output_dir}/dirty_vs_clean_analysis.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path2}")
    
    plt.close()
    
    # Save raw data for machine learning
    print("\n✓ Generating ML training data...")
    
    # Prepare data as numpy arrays
    ml_data = {
        'dirty_image': dirty_norm,
        'clean_image': clean_norm,
        'sky_truth': sky_norm,
        'psf': psf_norm,
        'dirty_phase': dirty_phase,
        'clean_phase': clean_phase,
    }
    
    ml_path = f'{output_dir}/dirty_clean_dataset.npz'
    np.savez(ml_path, **ml_data)
    print(f"✓ Saved: {ml_path}")
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files in '{output_dir}/':")
    print("  1. dirty_vs_clean_comparison.png - 9-panel comparison")
    print("  2. dirty_vs_clean_analysis.png - Detailed metrics and profiles")
    print("  3. dirty_clean_dataset.npz - Raw data for ML training")
    print("\nDataset Statistics:")
    print(f"  Images: 256×256 pixels")
    print(f"  Dirty (input) peak: {np.max(dirty_norm):.4f}")
    print(f"  Clean (output) peak: {np.max(clean_norm):.4f}")
    print(f"  Truth peak: {np.max(sky_norm):.4f}")
    print(f"  Dirty MSE: {dirty_mse:.6f}")
    print(f"  Clean MSE: {clean_mse:.6f}")
    print(f"  Improvement: {(1-clean_mse/dirty_mse)*100:.1f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    generate_comparison_dataset(output_dir='testing')
