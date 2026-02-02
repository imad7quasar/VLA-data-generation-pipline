"""
VLA Array Configuration Module
Defines antenna positions and baseline calculations
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.constants import c as speed_of_light


class VLAConfiguration:
    """
    VLA array configuration in different modes (A, B, C, D).
    Antenna positions based on NRAO VLA documentation.
    """
    
    def __init__(self, mode='A', frequency_ghz=1.4):
        """
        Initialize VLA array.
        
        Parameters
        ----------
        mode : str
            Array configuration ('A', 'B', 'C', 'D')
        frequency_ghz : float
            Observing frequency in GHz (default 1.4 GHz)
        """
        self.mode = mode
        self.frequency_ghz = frequency_ghz
        self.wavelength_m = speed_of_light / (frequency_ghz * 1e9)
        
        # Define antenna positions for each mode (in meters)
        # Simplified VLA positions in East-North-Up (ENU) coordinates
        self.antenna_positions = self._get_antenna_positions()
        self.n_antennas = len(self.antenna_positions)
        
    def _get_antenna_positions(self):
        """
        Return antenna positions based on array configuration.
        Positions are approximate VLA positions in ENU coordinates (meters).
        """
        # Simplified VLA antenna positions relative to array center
        positions_dict = {
            'A': np.array([
                [0, 0, 0],
                [36400, 0, 0],
                [0, 36400, 0],
                [36400, 36400, 0],
                [-36400, 0, 0],
                [0, -36400, 0],
                [-36400, -36400, 0],
                [25800, 25800, 0],
                [-25800, -25800, 0],
            ]),
            'B': np.array([
                [0, 0, 0],
                [10000, 0, 0],
                [0, 10000, 0],
                [10000, 10000, 0],
                [-10000, 0, 0],
                [0, -10000, 0],
                [-10000, -10000, 0],
                [7000, 7000, 0],
                [-7000, -7000, 0],
            ]),
            'C': np.array([
                [0, 0, 0],
                [3000, 0, 0],
                [0, 3000, 0],
                [3000, 3000, 0],
                [-3000, 0, 0],
                [0, -3000, 0],
                [-3000, -3000, 0],
                [2000, 2000, 0],
                [-2000, -2000, 0],
            ]),
            'D': np.array([
                [0, 0, 0],
                [600, 0, 0],
                [0, 600, 0],
                [600, 600, 0],
                [-600, 0, 0],
                [0, -600, 0],
                [-600, -600, 0],
                [400, 400, 0],
                [-400, -400, 0],
            ]),
        }
        
        return positions_dict.get(self.mode, positions_dict['A'])
    
    def get_baselines(self):
        """
        Compute all baseline vectors (antenna pairs).
        
        Returns
        -------
        baselines : ndarray
            Shape (n_baselines, 3), baseline vectors in meters
        baseline_pairs : ndarray
            Shape (n_baselines, 2), antenna pair indices
        """
        baselines = []
        baseline_pairs = []
        
        for i in range(self.n_antennas):
            for j in range(i + 1, self.n_antennas):
                baseline = self.antenna_positions[j] - self.antenna_positions[i]
                baselines.append(baseline)
                baseline_pairs.append([i, j])
        
        return np.array(baselines), np.array(baseline_pairs)
    
    def get_uv_coordinates(self, hour_angle_deg=0, declination_deg=30):
        """
        Compute UV coordinates for given source position.
        
        Parameters
        ----------
        hour_angle_deg : float
            Hour angle in degrees
        declination_deg : float
            Declination in degrees
        
        Returns
        -------
        u, v : ndarray
            UV coordinates in wavelengths
        baselines : ndarray
            Baseline vectors in meters
        baseline_pairs : ndarray
            Antenna pair indices
        """
        baselines, baseline_pairs = self.get_baselines()
        
        # Convert angles to radians
        ha = np.radians(hour_angle_deg)
        dec = np.radians(declination_deg)
        
        # Rotation matrix: project ENU to UV plane
        # U is East, V is North * sin(dec) + Up * cos(dec)
        sin_ha = np.sin(ha)
        cos_ha = np.cos(ha)
        sin_dec = np.sin(dec)
        cos_dec = np.cos(dec)
        
        # Transform baselines to UV coordinates
        u = baselines[:, 0] * sin_ha - baselines[:, 1] * cos_ha
        v = baselines[:, 0] * cos_ha * sin_dec + baselines[:, 1] * sin_ha * sin_dec + \
            baselines[:, 2] * cos_dec
        
        # Convert to wavelengths
        u = u / self.wavelength_m
        v = v / self.wavelength_m
        
        return u, v, baselines, baseline_pairs
    
    def plot_uv_coverage(self, hour_angle_deg=0, declination_deg=30, figsize=(8, 8)):
        """
        Plot UV coverage.
        
        Parameters
        ----------
        hour_angle_deg : float
            Hour angle in degrees
        declination_deg : float
            Declination in degrees
        figsize : tuple
            Figure size
        """
        u, v, _, _ = self.get_uv_coordinates(hour_angle_deg, declination_deg)
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot all points and their conjugates
        ax.scatter(u, v, s=20, alpha=0.6, label='Baseline points')
        ax.scatter(-u, -v, s=20, alpha=0.6, label='Conjugate points')
        
        ax.set_xlabel('U (wavelengths)', fontsize=12)
        ax.set_ylabel('V (wavelengths)', fontsize=12)
        ax.set_title(f'UV Coverage ({self.mode}-array, {self.frequency_ghz} GHz)', fontsize=14)
        ax.grid(True, alpha=0.3)
        ax.axis('equal')
        ax.legend()
        
        return fig, ax


if __name__ == '__main__':
    # Test array configuration
    vla = VLAConfiguration(mode='A', frequency_ghz=1.4)
    print(f"Number of antennas: {vla.n_antennas}")
    
    u, v, _, _ = vla.get_uv_coordinates()
    print(f"Number of baselines: {len(u)}")
    print(f"Wavelength: {vla.wavelength_m:.4f} m")
    
    fig, ax = vla.plot_uv_coverage()
    plt.savefig('uv_coverage.png', dpi=150, bbox_inches='tight')
    plt.show()
