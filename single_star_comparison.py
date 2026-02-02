"""
Single Point Source (Star) - Dirty vs Clean Comparison
Simple example with one bright object
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
    """Simple Wiener filter restoration."""
    from scipy.fft import fft2, ifft2
    
    dirty_fft = fft2(dirty_image)
    psf_fft = fft2(psf)
    psf_fft_norm = psf_fft / (np.max(np.abs(psf_fft)) + 1e-10)
    wiener = np.conj(psf_fft_norm) / (np.abs(psf_fft_norm)**2 + noise_level)
    clean_fft = dirty_fft * wiener
    clean = np.real(ifft2(clean_fft))
    
    return clean


def generate_single_star_comparison(output_dir='testing'):
    """
    Generate comparison for single point source (star).
    
    Parameters
    ----------
    output_dir : str
        Output directory for images
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    print("="*70)
    print("SINGLE STAR - DIRTY vs CLEAN COMPARISON")
    print("="*70)
    
    # Create pipeline
    pipeline = VLAConfiguration(mode='A', frequency_ghz=1.4)
    
    # Sky model with single bright point source (star)
    sky = SkyModel(image_size=256, pixel_scale_arcsec=2.0, wavelength_m=0.214)
    
    # Single bright point source at center (like a star)
    sky.add_point_source(
        x_offset_pix=0,
        y_offset_pix=0,
        flux_jy=5.0,          # Bright star
        phase_rad=0
    )
    
    sky.generate_sky()
    print(f"✓ Single star created at center")
    print(f"  Flux: 5.0 Jy")
    
    # Generate visibilities
    vis_gen = VisibilityGenerator(sky, pipeline)
    vis_gen.compute_sky_fft()
    vis, u, v = vis_gen.sample_visibilities(declination_deg=45)
    
    # Add realistic noise
    vis_gen.add_uv_noise(snr_db=40)
    print(f"✓ Visibilities computed with SNR=40dB")
    
    # Create dirty image
    engine = ImagingEngine(image_size=256, wavelength_m=0.214, pixel_scale_arcsec=2.0)
    psf = engine.compute_psf(u, v)
    dirty_image = engine.compute_dirty_image(vis_gen.visibilities, u, v)
    
    print(f"✓ Dirty image created")
    print(f"  Peak brightness: {np.max(np.abs(dirty_image)):.4f} Jy")
    
    # Create clean image using Wiener filter
    clean_image = create_clean_image_wiener(dirty_image, psf, noise_level=0.01)
    clean_image = gaussian_filter(clean_image, sigma=1.0)
    
    print(f"✓ Clean image created")
    print(f"  Peak brightness: {np.max(clean_image):.4f} Jy")
    
    # ===== CREATE MAIN COMPARISON FIGURE =====
    fig = plt.figure(figsize=(18, 12))
    gs = GridSpec(3, 3, figure=fig, hspace=0.4, wspace=0.35)
    
    # Normalize for visualization
    dirty_norm = np.abs(dirty_image) / (np.max(np.abs(dirty_image)) + 1e-10)
    clean_norm = np.abs(clean_image) / (np.max(np.abs(clean_image)) + 1e-10)
    sky_norm = np.abs(sky.sky) / (np.max(np.abs(sky.sky)) + 1e-10)
    
    # --- TOP ROW: AMPLITUDE ---
    # Dirty amplitude
    ax1 = fig.add_subplot(gs[0, 0])
    im1 = ax1.imshow(dirty_norm, origin='lower', cmap='hot',
                     extent=[-128, 128, -128, 128])
    ax1.set_title('DIRTY IMAGE (Input)\nAmplitude', fontsize=13, fontweight='bold', color='darkred')
    ax1.set_xlabel('X (pixels)')
    ax1.set_ylabel('Y (pixels)')
    cbar1 = plt.colorbar(im1, ax=ax1, label='Normalized')
    
    # Clean amplitude
    ax2 = fig.add_subplot(gs[0, 1])
    im2 = ax2.imshow(clean_norm, origin='lower', cmap='hot',
                     extent=[-128, 128, -128, 128])
    ax2.set_title('CLEAN IMAGE (Output)\nAmplitude', fontsize=13, fontweight='bold', color='darkgreen')
    ax2.set_xlabel('X (pixels)')
    ax2.set_ylabel('Y (pixels)')
    cbar2 = plt.colorbar(im2, ax=ax2, label='Normalized')
    
    # Sky/Truth amplitude
    ax3 = fig.add_subplot(gs[0, 2])
    im3 = ax3.imshow(sky_norm, origin='lower', cmap='hot',
                     extent=[-128, 128, -128, 128])
    ax3.set_title('TRUE STAR (Ground Truth)\nAmplitude', fontsize=13, fontweight='bold', color='darkblue')
    ax3.set_xlabel('X (pixels)')
    ax3.set_ylabel('Y (pixels)')
    cbar3 = plt.colorbar(im3, ax=ax3, label='Normalized')
    
    # --- MIDDLE ROW: LOG SCALE (Better for PSF visualization) ---
    # Dirty log
    ax4 = fig.add_subplot(gs[1, 0])
    dirty_log = np.abs(dirty_norm) + 1e-8
    im4 = ax4.imshow(np.log10(dirty_log), origin='lower', cmap='viridis',
                     extent=[-128, 128, -128, 128])
    ax4.set_title('DIRTY IMAGE\n(log scale)', fontsize=13, fontweight='bold', color='darkred')
    ax4.set_xlabel('X (pixels)')
    ax4.set_ylabel('Y (pixels)')
    plt.colorbar(im4, ax=ax4, label='log10')
    
    # Clean log
    ax5 = fig.add_subplot(gs[1, 1])
    clean_log = np.abs(clean_norm) + 1e-8
    im5 = ax5.imshow(np.log10(clean_log), origin='lower', cmap='viridis',
                     extent=[-128, 128, -128, 128])
    ax5.set_title('CLEAN IMAGE\n(log scale)', fontsize=13, fontweight='bold', color='darkgreen')
    ax5.set_xlabel('X (pixels)')
    ax5.set_ylabel('Y (pixels)')
    plt.colorbar(im5, ax=ax5, label='log10')
    
    # PSF log
    ax6 = fig.add_subplot(gs[1, 2])
    psf_norm = psf / np.max(np.abs(psf))
    psf_log = np.abs(psf_norm) + 1e-8
    im6 = ax6.imshow(np.log10(psf_log), origin='lower', cmap='viridis',
                     extent=[-128, 128, -128, 128])
    ax6.set_title('POINT SPREAD FUNCTION\n(log scale)', fontsize=13, fontweight='bold')
    ax6.set_xlabel('X (pixels)')
    ax6.set_ylabel('Y (pixels)')
    plt.colorbar(im6, ax=ax6, label='log10')
    
    # --- BOTTOM ROW: ERROR/RESIDUALS ---
    # Dirty error
    ax7 = fig.add_subplot(gs[2, 0])
    dirty_error = dirty_norm - sky_norm
    from matplotlib.colors import SymLogNorm
    norm_error = SymLogNorm(linthresh=0.01, vmin=-np.max(np.abs(dirty_error)),
                           vmax=np.max(np.abs(dirty_error)))
    im7 = ax7.imshow(dirty_error, origin='lower', cmap='RdBu_r', norm=norm_error,
                     extent=[-128, 128, -128, 128])
    ax7.set_title('DIRTY - TRUE\nResiduals', fontsize=13, fontweight='bold', color='darkred')
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
    ax8.set_title('CLEAN - TRUE\nResiduals', fontsize=13, fontweight='bold', color='darkgreen')
    ax8.set_xlabel('X (pixels)')
    ax8.set_ylabel('Y (pixels)')
    plt.colorbar(im8, ax=ax8, label='Error')
    
    # Statistics box
    ax9 = fig.add_subplot(gs[2, 2])
    
    dirty_mse = np.mean((dirty_norm - sky_norm)**2)
    clean_mse = np.mean((clean_norm - sky_norm)**2)
    dirty_mae = np.mean(np.abs(dirty_norm - sky_norm))
    clean_mae = np.mean(np.abs(clean_norm - sky_norm))
    
    stats_text = f"""
    RECONSTRUCTION METRICS
    ───────────────────────
    
    Dirty Image:
      RMS Error:  {np.sqrt(dirty_mse):.4f}
      MAE:        {dirty_mae:.4f}
      Peak:       {np.max(dirty_norm):.4f}
    
    Clean Image:
      RMS Error:  {np.sqrt(clean_mse):.4f}
      MAE:        {clean_mae:.4f}
      Peak:       {np.max(clean_norm):.4f}
    
    Improvement:
      RMS:        {(1-clean_mse/dirty_mse)*100:.1f}%
      MAE:        {(1-clean_mae/dirty_mae)*100:.1f}%
    """
    
    ax9.text(0.05, 0.5, stats_text, fontsize=10, family='monospace',
            transform=ax9.transAxes, verticalalignment='center',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.9, pad=1))
    ax9.axis('off')
    
    # Overall title
    fig.suptitle('SINGLE BRIGHT STAR - DIRTY vs CLEAN RECONSTRUCTION\nIllustrating PSF Deconvolution',
                 fontsize=16, fontweight='bold', y=0.995)
    
    # Save figure
    output_path = f'{output_dir}/single_star_dirty_vs_clean.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Saved: {output_path}")
    
    plt.close()
    
    # ===== CREATE DETAILED ANALYSIS FIGURE =====
    fig2, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    # 1. Horizontal profile
    center = 256 // 2
    x = np.arange(256) - center
    
    dirty_horiz = dirty_norm[center, :]
    clean_horiz = clean_norm[center, :]
    sky_horiz = sky_norm[center, :]
    
    axes[0, 0].plot(x, dirty_horiz, 'r-', linewidth=2.5, label='Dirty (Observed)', alpha=0.8)
    axes[0, 0].plot(x, clean_horiz, 'g-', linewidth=2.5, label='Clean (Restored)', alpha=0.8)
    axes[0, 0].plot(x, sky_horiz, 'b--', linewidth=2.5, label='True Star', alpha=0.9)
    axes[0, 0].set_xlabel('X Position (pixels)', fontsize=11)
    axes[0, 0].set_ylabel('Brightness (normalized)', fontsize=11)
    axes[0, 0].set_title('Horizontal Cross-Section (Y=0)', fontweight='bold', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_xlim([-80, 80])
    
    # 2. Vertical profile
    dirty_vert = dirty_norm[:, center]
    clean_vert = clean_norm[:, center]
    sky_vert = sky_norm[:, center]
    
    axes[0, 1].plot(x, dirty_vert, 'r-', linewidth=2.5, label='Dirty', alpha=0.8)
    axes[0, 1].plot(x, clean_vert, 'g-', linewidth=2.5, label='Clean', alpha=0.8)
    axes[0, 1].plot(x, sky_vert, 'b--', linewidth=2.5, label='True', alpha=0.9)
    axes[0, 1].set_xlabel('Y Position (pixels)', fontsize=11)
    axes[0, 1].set_ylabel('Brightness (normalized)', fontsize=11)
    axes[0, 1].set_title('Vertical Cross-Section (X=0)', fontweight='bold', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].set_xlim([-80, 80])
    
    # 3. Radial profile
    y_idx, x_idx = np.ogrid[:256, :256]
    r = np.sqrt((x_idx - center)**2 + (y_idx - center)**2)
    
    r_bins = np.arange(0, 150, 2)
    r_centers = (r_bins[:-1] + r_bins[1:]) / 2
    
    profiles = {'Dirty': dirty_norm, 'Clean': clean_norm, 'True': sky_norm}
    colors = {'Dirty': 'red', 'Clean': 'green', 'True': 'blue'}
    
    for name, profile in profiles.items():
        r_profile = []
        for i in range(len(r_bins) - 1):
            mask = (r >= r_bins[i]) & (r < r_bins[i+1])
            if np.sum(mask) > 0:
                r_profile.append(np.mean(profile[mask]))
            else:
                r_profile.append(0)
        
        if name == 'True':
            axes[0, 2].plot(r_centers, r_profile, color=colors[name], 
                           linewidth=2.5, label=name, linestyle='--', alpha=0.9)
        else:
            axes[0, 2].plot(r_centers, r_profile, color=colors[name], 
                           linewidth=2.5, label=name, alpha=0.8)
    
    axes[0, 2].set_xlabel('Radius (pixels)', fontsize=11)
    axes[0, 2].set_ylabel('Mean Brightness (normalized)', fontsize=11)
    axes[0, 2].set_title('Radial Profile from Center', fontweight='bold', fontsize=12)
    axes[0, 2].legend(fontsize=10)
    axes[0, 2].grid(True, alpha=0.3)
    axes[0, 2].set_ylim([0, 1.1])
    
    # 4. Error metrics display
    metrics_text = f"""
    ERROR ANALYSIS
    ──────────────────────────
    
    Dirty Image (Observed):
      RMS Error:    {np.sqrt(dirty_mse):.6f}
      MAE:          {dirty_mae:.6f}
      Peak:         {np.max(dirty_norm):.4f}
      Min:          {np.min(dirty_norm):.6f}
    
    Clean Image (Restored):
      RMS Error:    {np.sqrt(clean_mse):.6f}
      MAE:          {clean_mae:.6f}
      Peak:         {np.max(clean_norm):.4f}
      Min:          {np.min(clean_norm):.6f}
    
    Improvement from Cleaning:
      RMS Error ↓:  {(1-clean_mse/dirty_mse)*100:.1f}%
      MAE ↓:        {(1-clean_mae/dirty_mae)*100:.1f}%
    
    PSF Statistics:
      Peak:         {np.max(psf_norm):.4f}
      Sidelobe:     {np.max(np.abs(psf_norm[1:, 1:])):.4f}
      Ratio:        {np.max(psf_norm)/np.max(np.abs(psf_norm[1:, 1:])):.1f}:1
    """
    
    axes[1, 0].text(0.05, 0.5, metrics_text, fontsize=9.5, family='monospace',
                   transform=axes[1, 0].transAxes, verticalalignment='center',
                   bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9, pad=1))
    axes[1, 0].axis('off')
    
    # 5. Peak brightness comparison
    peaks = [np.max(dirty_norm), np.max(clean_norm), np.max(sky_norm)]
    labels = ['Dirty\n(Observed)', 'Clean\n(Restored)', 'True\n(Star)']
    colors_bar = ['red', 'green', 'blue']
    
    bars = axes[1, 1].bar(labels, peaks, color=colors_bar, alpha=0.7, edgecolor='black', linewidth=2)
    axes[1, 1].set_ylabel('Peak Brightness (normalized)', fontsize=11)
    axes[1, 1].set_title('Peak Brightness Comparison', fontweight='bold', fontsize=12)
    axes[1, 1].set_ylim([0, 1.2])
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, peak in zip(bars, peaks):
        height = bar.get_height()
        axes[1, 1].text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{peak:.4f}', ha='center', va='bottom', fontweight='bold', fontsize=10)
    
    # 6. FFT Spectrum comparison
    from scipy.fft import fft2
    dirty_spectrum = np.abs(fft2(dirty_norm))
    clean_spectrum = np.abs(fft2(clean_norm))
    sky_spectrum = np.abs(fft2(sky_norm))
    
    # Get power spectrum sorted
    dirty_power = np.sort(dirty_spectrum.flatten())[-128:][::-1]
    clean_power = np.sort(clean_spectrum.flatten())[-128:][::-1]
    sky_power = np.sort(sky_spectrum.flatten())[-128:][::-1]
    
    axes[1, 2].loglog(range(1, 129), dirty_power, 'r-', linewidth=2.5, label='Dirty', marker='o', markersize=3, alpha=0.7)
    axes[1, 2].loglog(range(1, 129), clean_power, 'g-', linewidth=2.5, label='Clean', marker='s', markersize=3, alpha=0.7)
    axes[1, 2].loglog(range(1, 129), sky_power, 'b--', linewidth=2.5, label='True', marker='^', markersize=3, alpha=0.8)
    axes[1, 2].set_xlabel('Frequency Component Index (sorted)', fontsize=11)
    axes[1, 2].set_ylabel('Power (log scale)', fontsize=11)
    axes[1, 2].set_title('Fourier Spectrum Comparison', fontweight='bold', fontsize=12)
    axes[1, 2].legend(fontsize=10)
    axes[1, 2].grid(True, alpha=0.3, which='both')
    
    plt.tight_layout()
    
    output_path2 = f'{output_dir}/single_star_analysis.png'
    plt.savefig(output_path2, dpi=150, bbox_inches='tight')
    print(f"✓ Saved: {output_path2}")
    
    plt.close()
    
    # ===== SAVE RAW DATA FOR ML =====
    print("\n✓ Generating ML training data...")
    
    ml_data = {
        'dirty_image': dirty_norm,
        'clean_image': clean_norm,
        'sky_truth': sky_norm,
        'psf': psf_norm,
        'dirty_phase': np.angle(dirty_image),
        'clean_phase': np.angle(clean_image),
    }
    
    ml_path = f'{output_dir}/single_star_dataset.npz'
    np.savez(ml_path, **ml_data)
    print(f"✓ Saved: {ml_path}")
    
    print("\n" + "="*70)
    print("SINGLE STAR DATASET GENERATION COMPLETE")
    print("="*70)
    print(f"\nGenerated files in '{output_dir}/':")
    print("  1. single_star_dirty_vs_clean.png - 9-panel comparison")
    print("  2. single_star_analysis.png - Detailed analysis (6 plots)")
    print("  3. single_star_dataset.npz - Raw data for ML")
    print("\nDataset Statistics:")
    print(f"  Image Size: 256×256 pixels")
    print(f"  Single Star Flux: 5.0 Jy")
    print(f"  Dirty Peak (normalized): {np.max(dirty_norm):.4f}")
    print(f"  Clean Peak (normalized): {np.max(clean_norm):.4f}")
    print(f"  True Peak (normalized): {np.max(sky_norm):.4f}")
    print(f"  Dirty RMS Error: {np.sqrt(dirty_mse):.6f}")
    print(f"  Clean RMS Error: {np.sqrt(clean_mse):.6f}")
    print(f"  Improvement: {(1-clean_mse/dirty_mse)*100:.1f}%")
    print("="*70 + "\n")


if __name__ == '__main__':
    generate_single_star_comparison(output_dir='testing')
