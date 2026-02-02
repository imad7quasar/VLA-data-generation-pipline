"""
Configuration and utility functions for VLA pipeline
"""

import numpy as np
from scipy.constants import c as speed_of_light

# Constants
VLA_LOCATION_LAT = 34.0790  # degrees, Socorro, NM
VLA_LOCATION_LON = -107.6183  # degrees
VLA_LOCATION_ALT = 2124.0  # meters

# Array configuration modes
ARRAY_MODES = {
    'A': {'description': 'Extended configuration', 'max_baseline_m': 36400},
    'B': {'description': 'Intermediate configuration', 'max_baseline_m': 10000},
    'C': {'description': 'Compact configuration', 'max_baseline_m': 3000},
    'D': {'description': 'Most compact configuration', 'max_baseline_m': 600},
}

# Observing frequencies
OBSERVING_FREQUENCIES = {
    'L-band': 1.4,      # GHz
    'C-band': 5.0,      # GHz
    'X-band': 10.0,     # GHz
    'Ku-band': 15.0,    # GHz
    'K-band': 22.0,     # GHz
    'Ka-band': 34.0,    # GHz
    'Q-band': 43.0,     # GHz
}

# Sensitivity parameters
SENSITIVITY = {
    'L-band': 0.10,     # mJy/beam
    'C-band': 0.08,
    'X-band': 0.06,
    'Ku-band': 0.12,
    'K-band': 0.15,
    'Ka-band': 0.25,
    'Q-band': 0.40,
}


class PipelineConfig:
    """
    Configuration holder for VLA pipeline.
    """
    
    def __init__(self):
        self.array_mode = 'A'
        self.frequency_ghz = 1.4
        self.image_size = 256
        self.pixel_scale_arcsec = 2.0
        self.integration_time_s = 1.0
        self.add_noise = False
        self.snr_db = 100
        self.declination_deg = 45
        self.hour_angle_deg = 0
    
    def get_wavelength_m(self):
        """Get wavelength in meters."""
        return speed_of_light / (self.frequency_ghz * 1e9)
    
    def get_beam_size_arcsec(self):
        """
        Estimate synthesized beam size (FWHM).
        For point source at zenith: beam ~ wavelength / baseline
        """
        wavelength = self.get_wavelength_m()
        # Typical max baseline for A-array
        max_baseline = ARRAY_MODES[self.array_mode]['max_baseline_m']
        # Estimate beam size
        beam_rad = wavelength / max_baseline
        beam_arcsec = beam_rad * 206265  # radians to arcseconds
        return beam_arcsec
    
    def get_field_of_view_deg(self):
        """
        Estimate field of view at FWHM of primary beam.
        For VLA: ~30' at L-band, scales as wavelength.
        """
        wavelength = self.get_wavelength_m()
        dish_diameter = 25.0  # meters
        fov_rad = wavelength / dish_diameter
        fov_deg = fov_rad * 180 / np.pi
        return fov_deg


def create_default_config():
    """Create default pipeline configuration."""
    return PipelineConfig()


def print_config_summary(config):
    """Print summary of configuration."""
    print("\n" + "="*60)
    print("PIPELINE CONFIGURATION SUMMARY")
    print("="*60)
    print(f"Array Mode: {config.array_mode}-array")
    print(f"Frequency: {config.frequency_ghz} GHz ({OBSERVING_FREQUENCIES.get(f'L-band', config.frequency_ghz)} GHz)")
    print(f"Wavelength: {config.get_wavelength_m():.6f} m")
    print(f"Image Size: {config.image_size} x {config.image_size} pixels")
    print(f"Pixel Scale: {config.pixel_scale_arcsec} arcsec/pixel")
    print(f"Est. Beam Size: {config.get_beam_size_arcsec():.4f} arcsec")
    print(f"Field of View: {config.get_field_of_view_deg():.2f} degrees")
    print(f"Declination: {config.declination_deg} degrees")
    print(f"Hour Angle: {config.hour_angle_deg} degrees")
    if config.add_noise:
        print(f"SNR: {config.snr_db} dB")
    else:
        print(f"Noise: None (noiseless simulation)")
    print("="*60 + "\n")


if __name__ == '__main__':
    config = create_default_config()
    print_config_summary(config)
