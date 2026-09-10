"""
Kinetic & Physical Feature Engineering Module.
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Extracts biomechanically interpretable features from 25 Hz 6-channel IMU windows:
- Time Domain: Mean, Variance, Signal Magnitude Area (SMA), Jerk, Zero-Crossing Rate.
- Frequency Domain: Dominant Peak Frequency (Cadence), Spectral Energy, Spectral Entropy.
- Posture/Orientation: Tilt Angles (Pitch, Roll) derived from gravity vector components.
These features are used for both lightweight classification and grounded explanation generation.
"""
import numpy as np
from scipy import signal
from typing import Dict, Any, List

def compute_kinetic_features_for_window(
    window: np.ndarray,
    fs_hz: float = 25.0
) -> Dict[str, float]:
    """
    Computes biomechanical and signal features for a single (window_len, 6) matrix.
    Channels: [acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z]
    """
    # Accelerometer channels
    ax, ay, az = window[:, 0], window[:, 1], window[:, 2]
    # Gyroscope channels
    gx, gy, gz = window[:, 3], window[:, 4], window[:, 5]

    # Magnitudes
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)

    # Accelerometer statistics
    acc_mean_mag = float(np.mean(acc_mag))
    acc_var_mag = float(np.var(acc_mag))
    acc_std_mag = float(np.std(acc_mag))
    acc_max_peak = float(np.max(acc_mag))

    # Signal Magnitude Area (SMA) - normalized by length
    sma_acc = float(np.sum(np.abs(ax) + np.abs(ay) + np.abs(az)) / len(window))
    sma_gyro = float(np.sum(np.abs(gx) + np.abs(gy) + np.abs(gz)) / len(window))

    # Dynamic Jerk (rate of change of acceleration)
    jerk = np.diff(acc_mag) * fs_hz
    jerk_var = float(np.var(jerk)) if len(jerk) > 0 else 0.0

    # Gyroscope statistics
    gyro_mean_mag = float(np.mean(gyro_mag))
    gyro_var_mag = float(np.var(gyro_mag))

    # Posture/Tilt angle estimation (based on static gravity orientation)
    mean_ax, mean_ay, mean_az = float(np.mean(ax)), float(np.mean(ay)), float(np.mean(az))
    total_g = np.sqrt(mean_ax**2 + mean_ay**2 + mean_az**2) + 1e-6
    pitch_deg = float(np.arcsin(np.clip(-mean_ax / total_g, -1.0, 1.0)) * 180.0 / np.pi)
    roll_deg = float(np.arctan2(mean_ay, mean_az) * 180.0 / np.pi)

    # Frequency Domain Analysis on vertical/primary dynamic axis (Ay and Az)
    # Cadence / Dominant step frequency via FFT
    n = len(acc_mag)
    freqs = np.fft.rfftfreq(n, d=1.0/fs_hz)
    fft_vals = np.abs(np.fft.rfft(acc_mag - acc_mean_mag))  # Detrended
    
    # Ignore 0 Hz DC offset, search between 0.3 Hz and 5.0 Hz for gait rhythms
    valid_freq_mask = (freqs >= 0.3) & (freqs <= 5.0)
    if np.any(valid_freq_mask) and np.max(fft_vals[valid_freq_mask]) > 1e-4:
        dom_idx = np.argmax(fft_vals[valid_freq_mask])
        dom_freq_hz = float(freqs[valid_freq_mask][dom_idx])
        dom_spectral_power = float(fft_vals[valid_freq_mask][dom_idx])
    else:
        dom_freq_hz = 0.0
        dom_spectral_power = 0.0

    # Gyroscope periodicity check (pedaling/balance in cycling)
    gyro_fft = np.abs(np.fft.rfft(gyro_mag - gyro_mean_mag))
    if np.any(valid_freq_mask) and np.max(gyro_fft[valid_freq_mask]) > 1e-4:
        gyro_dom_idx = np.argmax(gyro_fft[valid_freq_mask])
        gyro_dom_freq_hz = float(freqs[valid_freq_mask][gyro_dom_idx])
    else:
        gyro_dom_freq_hz = 0.0

    return {
        'acc_mean_mag': acc_mean_mag,
        'acc_var_mag': acc_var_mag,
        'acc_std_mag': acc_std_mag,
        'acc_max_peak': acc_max_peak,
        'sma_acc': sma_acc,
        'sma_gyro': sma_gyro,
        'jerk_var': jerk_var,
        'gyro_mean_mag': gyro_mean_mag,
        'gyro_var_mag': gyro_var_mag,
        'pitch_deg': pitch_deg,
        'roll_deg': roll_deg,
        'dom_freq_hz': dom_freq_hz,
        'dom_spectral_power': dom_spectral_power,
        'gyro_dom_freq_hz': gyro_dom_freq_hz,
        'mean_ax': mean_ax,
        'mean_ay': mean_ay,
        'mean_az': mean_az,
        'var_ax': float(np.var(ax)),
        'var_ay': float(np.var(ay)),
        'var_az': float(np.var(az)),
    }

def extract_features_batch(X: np.ndarray, fs_hz: float = 25.0) -> np.ndarray:
    """
    Extracts numerical feature vectors for an array of windows of shape (num_windows, window_len, 6).
    Returns feature matrix of shape (num_windows, num_features).
    """
    feature_rows = []
    for i in range(len(X)):
        feats = compute_kinetic_features_for_window(X[i], fs_hz=fs_hz)
        feature_rows.append(list(feats.values()))
    return np.array(feature_rows, dtype=np.float32)

def get_feature_names() -> List[str]:
    dummy_win = np.zeros((64, 6), dtype=np.float32)
    return list(compute_kinetic_features_for_window(dummy_win).keys())
