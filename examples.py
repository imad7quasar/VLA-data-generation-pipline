"""
Quick Start Guide for VLA Imaging Pipeline
Run this file to see a complete example
"""

import matplotlib.pyplot as plt
from pipeline import VLAPipeline
from visualization import ComprehensivePlotter


def example_1_simple_point_source():
    """
    Example 1: Simple point source imaging
    """
    print("\n" + "="*70)
    print("EXAMPLE 1: SIMPLE POINT SOURCE")
    print("="*70)
    
    # Create pipeline
    pipeline = VLAPipeline(
        array_mode='A',
        frequency_ghz=1.4,
        image_size=256,
        pixel_scale_arcsec=2.0
    )
    
    # Single point source at center
    pipeline.setup_sources([
        {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 2.0}
    ])
    
    # Compute imaging
    pipeline.compute_visibilities(declination_deg=45)
    pipeline.compute_images()
    
    # Plot
    fig = pipeline.plot_full_pipeline()
    plt.savefig('example1_point_source.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example1_point_source.png")
    
    return pipeline


def example_2_multiple_sources():
    """
    Example 2: Multiple point sources and Gaussian
    """
    print("\n" + "="*70)
    print("EXAMPLE 2: MULTIPLE SOURCES")
    print("="*70)
    
    pipeline = VLAPipeline(
        array_mode='A',
        frequency_ghz=1.4,
        image_size=256,
        pixel_scale_arcsec=2.0
    )
    
    # Multiple sources
    sources = [
        {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 1.5},
        {'type': 'point', 'x_pix': 30, 'y_pix': 20, 'flux_jy': 0.8},
        {'type': 'gaussian', 'x_pix': -20, 'y_pix': -15, 'flux_jy': 1.0, 'fwhm_pix': 10},
    ]
    
    pipeline.setup_sources(sources)
    pipeline.compute_visibilities(declination_deg=45)
    pipeline.compute_images()
    
    fig = pipeline.plot_full_pipeline()
    plt.savefig('example2_multiple_sources.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example2_multiple_sources.png")
    
    return pipeline


def example_3_different_array():
    """
    Example 3: Different array configuration (C-array)
    """
    print("\n" + "="*70)
    print("EXAMPLE 3: C-ARRAY CONFIGURATION")
    print("="*70)
    
    pipeline = VLAPipeline(
        array_mode='C',  # Compact
        frequency_ghz=5.0,  # C-band
        image_size=256,
        pixel_scale_arcsec=5.0  # Coarser resolution
    )
    
    sources = [
        {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 2.0},
        {'type': 'gaussian', 'x_pix': 25, 'y_pix': 15, 'flux_jy': 1.5, 'fwhm_pix': 8},
    ]
    
    pipeline.setup_sources(sources)
    pipeline.compute_visibilities(declination_deg=30)
    pipeline.compute_images()
    
    fig = pipeline.plot_full_pipeline()
    plt.savefig('example3_c_array.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example3_c_array.png")
    
    return pipeline


def example_4_detailed_analysis():
    """
    Example 4: Detailed analysis of single source
    """
    print("\n" + "="*70)
    print("EXAMPLE 4: DETAILED ANALYSIS")
    print("="*70)
    
    pipeline = VLAPipeline(
        array_mode='A',
        frequency_ghz=1.4,
        image_size=512,  # High resolution
        pixel_scale_arcsec=1.0
    )
    
    sources = [
        {'type': 'gaussian', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 2.0, 'fwhm_pix': 15}
    ]
    
    pipeline.setup_sources(sources)
    pipeline.compute_visibilities(declination_deg=45)
    pipeline.compute_images()
    
    # Create multiple detailed plots
    fig1 = pipeline.plot_full_pipeline(figsize=(20, 14))
    plt.savefig('example4_detailed_overview.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example4_detailed_overview.png")
    
    # PSF analysis
    fig2, axes = ComprehensivePlotter.plot_psf_analysis(pipeline.psf)
    plt.savefig('example4_psf_analysis.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example4_psf_analysis.png")
    
    # Radial profile
    fig3, ax = ComprehensivePlotter.plot_radial_profile(pipeline.dirty_image)
    plt.savefig('example4_radial_profile.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example4_radial_profile.png")
    
    # Cross sections
    fig4, axes = ComprehensivePlotter.plot_cross_section(pipeline.dirty_image)
    plt.savefig('example4_cross_sections.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example4_cross_sections.png")
    
    # Comparison
    fig5, axes = ComprehensivePlotter.plot_comparison_dirty_sky(
        pipeline.sky_model.sky, pipeline.dirty_image
    )
    plt.savefig('example4_dirty_vs_sky.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example4_dirty_vs_sky.png")
    
    return pipeline


def example_5_uv_coverage_study():
    """
    Example 5: Study UV coverage effect on PSF
    """
    print("\n" + "="*70)
    print("EXAMPLE 5: UV COVERAGE STUDY")
    print("="*70)
    
    from visualization import plot_uv_plane_sampling
    import matplotlib.pyplot as plt
    
    # Compare different array configurations
    modes = ['A', 'B', 'C', 'D']
    fig, axes = plt.subplots(2, 2, figsize=(14, 14))
    
    for idx, mode in enumerate(modes):
        ax = axes[idx // 2, idx % 2]
        
        pipeline = VLAPipeline(
            array_mode=mode,
            frequency_ghz=1.4,
            image_size=128,
            pixel_scale_arcsec=5.0
        )
        
        sources = [
            {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 1.0}
        ]
        
        pipeline.setup_sources(sources)
        pipeline.compute_visibilities()
        
        # Plot UV coverage
        ax.scatter(pipeline.u_samples, pipeline.v_samples, s=30, alpha=0.6)
        ax.scatter(-pipeline.u_samples, -pipeline.v_samples, s=30, alpha=0.6)
        ax.set_xlabel('U (wavelengths)', fontsize=10)
        ax.set_ylabel('V (wavelengths)', fontsize=10)
        ax.set_title(f'{mode}-array UV Coverage', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
    
    plt.tight_layout()
    plt.savefig('example5_uv_coverage_comparison.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example5_uv_coverage_comparison.png")
    
    return None


def example_6_frequency_study():
    """
    Example 6: Study effect of observing frequency
    """
    print("\n" + "="*70)
    print("EXAMPLE 6: FREQUENCY STUDY")
    print("="*70)
    
    frequencies = [1.4, 5.0, 10.0]  # L, C, X band
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for idx, freq in enumerate(frequencies):
        pipeline = VLAPipeline(
            array_mode='A',
            frequency_ghz=freq,
            image_size=256,
            pixel_scale_arcsec=2.0
        )
        
        sources = [
            {'type': 'point', 'x_pix': 0, 'y_pix': 0, 'flux_jy': 1.0}
        ]
        
        pipeline.setup_sources(sources)
        pipeline.compute_visibilities()
        
        # Plot PSF
        psf_norm = pipeline.psf / np.max(np.abs(pipeline.psf))
        psf_to_plot = np.abs(psf_norm) + 1e-8
        
        im = axes[idx].imshow(np.log10(psf_to_plot), origin='lower', cmap='viridis',
                             extent=[-128, 128, -128, 128])
        axes[idx].set_title(f'{freq} GHz PSF', fontsize=12, fontweight='bold')
        axes[idx].set_xlabel('X (pixels)')
        axes[idx].set_ylabel('Y (pixels)')
        plt.colorbar(im, ax=axes[idx], label='log10(PSF)')
    
    plt.tight_layout()
    plt.savefig('example6_frequency_study.png', dpi=150, bbox_inches='tight')
    print("✓ Saved: example6_frequency_study.png")


def run_all_examples():
    """Run all examples"""
    import numpy as np
    
    print("\n" + "="*70)
    print("VLA IMAGING PIPELINE - QUICK START EXAMPLES")
    print("="*70)
    
    # Close any existing plots
    plt.close('all')
    
    try:
        example_1_simple_point_source()
        example_2_multiple_sources()
        example_3_different_array()
        example_4_detailed_analysis()
        example_5_uv_coverage_study()
        example_6_frequency_study()
        
        print("\n" + "="*70)
        print("✅ ALL EXAMPLES COMPLETED")
        print("="*70)
        print("\nGenerated files:")
        print("  - example1_point_source.png")
        print("  - example2_multiple_sources.png")
        print("  - example3_c_array.png")
        print("  - example4_*.png (5 files)")
        print("  - example5_uv_coverage_comparison.png")
        print("  - example6_frequency_study.png")
        print("="*70 + "\n")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    import numpy as np
    
    run_all_examples()
    
    # Show plots (comment out if running in non-interactive environment)
    # plt.show()
