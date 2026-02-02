"""
Extended visualization and analysis tools
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm, LogNorm
from scipy.ndimage import gaussian_filter


class ComprehensivePlotter:
    """
    Advanced plotting utilities for pipeline visualization.
    """
    
    @staticmethod
    def plot_comparison_dirty_sky(sky_image, dirty_image, figsize=(15, 5)):
        """
        Side-by-side comparison of sky and dirty image.
        
        Parameters
        ----------
        sky_image : ndarray
            Sky brightness distribution
        dirty_image : ndarray
            Dirty image from inverse FFT
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, axes : matplotlib objects
        """
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        sky_amp = np.abs(sky_image)
        dirty_norm = dirty_image / (np.max(np.abs(dirty_image)) + 1e-10)
        sky_norm = sky_amp / (np.max(sky_amp) + 1e-10)
        
        # Sky
        im0 = axes[0].imshow(sky_norm, origin='lower', cmap='viridis')
        axes[0].set_title('True Sky (Normalized)', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('X (pixels)')
        axes[0].set_ylabel('Y (pixels)')
        plt.colorbar(im0, ax=axes[0])
        
        # Dirty image
        im1 = axes[1].imshow(dirty_norm, origin='lower', cmap='viridis')
        axes[1].set_title('Dirty Image (Normalized)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        plt.colorbar(im1, ax=axes[1])
        
        # Difference
        difference = dirty_norm - sky_norm
        norm = SymLogNorm(linthresh=0.01, vmin=-np.max(np.abs(difference)),
                         vmax=np.max(np.abs(difference)))
        im2 = axes[2].imshow(difference, origin='lower', cmap='RdBu_r', norm=norm)
        axes[2].set_title('Difference (Dirty - Sky)', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('X (pixels)')
        axes[2].set_ylabel('Y (pixels)')
        plt.colorbar(im2, ax=axes[2])
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_radial_profile(image, title='Radial Profile', figsize=(10, 6)):
        """
        Compute and plot radial profile of image.
        
        Parameters
        ----------
        image : ndarray
            2D image
        title : str
            Plot title
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib objects
        """
        center = np.array(image.shape) // 2
        y, x = np.ogrid[:image.shape[0], :image.shape[1]]
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2)
        
        r_flat = r.ravel()
        image_flat = np.abs(image).ravel()
        
        # Bin by radius
        r_bins = np.arange(0, np.max(r) + 1, 1)
        r_centers = (r_bins[:-1] + r_bins[1:]) / 2
        radial_profile = []
        
        for i in range(len(r_bins) - 1):
            mask = (r_flat >= r_bins[i]) & (r_flat < r_bins[i+1])
            if np.sum(mask) > 0:
                radial_profile.append(np.mean(image_flat[mask]))
            else:
                radial_profile.append(0)
        
        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(r_centers, radial_profile, linewidth=2, marker='o', markersize=4)
        ax.set_xlabel('Radius (pixels)', fontsize=12)
        ax.set_ylabel('Brightness (Jy)', fontsize=12)
        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        return fig, ax
    
    @staticmethod
    def plot_cross_section(image, figsize=(12, 5)):
        """
        Plot horizontal and vertical cross sections through image.
        
        Parameters
        ----------
        image : ndarray
            2D image
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, axes : matplotlib objects
        """
        center = image.shape[0] // 2
        horizontal = np.abs(image[center, :])
        vertical = np.abs(image[:, center])
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        x = np.arange(len(horizontal)) - center
        y = np.arange(len(vertical)) - center
        
        axes[0].plot(x, horizontal, linewidth=2)
        axes[0].set_xlabel('X position (pixels)', fontsize=11)
        axes[0].set_ylabel('Brightness (Jy)', fontsize=11)
        axes[0].set_title('Horizontal Cross Section (Y=0)', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].axvline(0, color='red', linestyle='--', alpha=0.5)
        
        axes[1].plot(y, vertical, linewidth=2, color='orange')
        axes[1].set_xlabel('Y position (pixels)', fontsize=11)
        axes[1].set_ylabel('Brightness (Jy)', fontsize=11)
        axes[1].set_title('Vertical Cross Section (X=0)', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].axvline(0, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        return fig, axes
    
    @staticmethod
    def plot_psf_analysis(psf, figsize=(14, 6)):
        """
        Analyze PSF: peak, sidelobes, FWHM.
        
        Parameters
        ----------
        psf : ndarray
            Point spread function
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, axes : matplotlib objects
        """
        center = np.array(psf.shape) // 2
        
        # Normalize PSF
        psf_norm = psf / np.max(psf)
        
        # Cross sections
        horizontal = psf_norm[center[0], :]
        vertical = psf_norm[:, center[1]]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Horizontal
        x = np.arange(len(horizontal)) - center[1]
        axes[0].plot(x, horizontal, linewidth=2, label='Horizontal')
        axes[0].plot(x, vertical, linewidth=2, label='Vertical', alpha=0.7)
        axes[0].set_xlabel('Offset (pixels)', fontsize=11)
        axes[0].set_ylabel('PSF (normalized)', fontsize=11)
        axes[0].set_title('PSF Cross Sections', fontsize=12, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        axes[0].set_xlim([-100, 100])
        
        # Log scale
        psf_to_plot = np.abs(psf_norm) + 1e-6
        im = axes[1].imshow(np.log10(psf_to_plot), origin='lower', cmap='viridis',
                           extent=[-center[1], len(horizontal)-center[1],
                                   -center[0], len(vertical)-center[0]])
        axes[1].set_title('PSF (log scale)', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('X (pixels)')
        axes[1].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[1], label='log10(PSF)')
        
        # Compute statistics
        central_peak = psf_norm[center[0], center[1]]
        sidelobe_max = np.max(np.abs(psf_norm[
            (np.abs(np.arange(psf.shape[0]) - center[0]) > 5) |
            (np.abs(np.arange(psf.shape[1]) - center[1]) > 5)
        ]))
        
        stats_text = f'Peak: {central_peak:.4f}\nMax sidelobe: {sidelobe_max:.4f}\nRatio: {central_peak/sidelobe_max:.1f}:1'
        axes[1].text(0.05, 0.95, stats_text, transform=axes[1].transAxes,
                    fontsize=10, verticalalignment='top',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        return fig, axes


def plot_uv_plane_sampling(u_samples, v_samples, figsize=(10, 10)):
    """
    Detailed plot of UV plane sampling.
    
    Parameters
    ----------
    u_samples, v_samples : ndarray
        UV coordinates in wavelengths
    figsize : tuple
        Figure size
    
    Returns
    -------
    fig, ax : matplotlib objects
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot all points including conjugates
    ax.scatter(u_samples, v_samples, s=50, alpha=0.6, label='Baselines', color='blue')
    ax.scatter(-u_samples, -v_samples, s=50, alpha=0.6, label='Conjugates', color='red')
    
    # Add circle indicating baseline length
    for u, v in zip(u_samples[::5], v_samples[::5]):  # Every 5th point to avoid clutter
        r = np.sqrt(u**2 + v**2)
        circle = plt.Circle((0, 0), r, fill=False, linestyle='--', alpha=0.2)
        ax.add_patch(circle)
    
    ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.axvline(x=0, color='k', linestyle='-', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('U (wavelengths)', fontsize=12)
    ax.set_ylabel('V (wavelengths)', fontsize=12)
    ax.set_title('UV Plane Sampling', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.axis('equal')
    
    return fig, ax


if __name__ == '__main__':
    # Example usage
    print("Visualization tools module loaded successfully")
