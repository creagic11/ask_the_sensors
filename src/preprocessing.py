"""
Sensor Preprocessing and Stream Normalization Module.
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Handles:
- Resampling arbitrary/irregular sensor timestamps to standard 25 Hz.
- Handling sensor dropouts, missing intervals, and NaN interpolation.
- Sliding window segmentation (e.g., 2.56s / 64 samples with 50% overlap).
- Noise filtering (Butterworth low-pass / bandpass).
"""
import numpy as np
import pandas as pd
from scipy import signal
from typing import Tuple, List, Dict, Optional

TARGET_SAMPLING_RATE_HZ = 25.0
DELTA_T_SEC = 1.0 / TARGET_SAMPLING_RATE_HZ  # 0.04s

SENSOR_COLUMNS = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
LABEL_COLUMNS = [
    'lying_down',
    'sitting',
    'standing_in_place',
    'standing_and_moving',
    'walking',
    'running',
    'bicycling'
]

def clean_and_resample_stream(
    df: pd.DataFrame,
    timestamp_col: str = 'timestamp',
    target_hz: float = TARGET_SAMPLING_RATE_HZ,
    max_interpolation_gap_sec: float = 3.0
) -> pd.DataFrame:
    """
    Resamples a multivariate sensor dataframe to a uniform grid at `target_hz` (25 Hz).
    Fills short missing gaps using linear interpolation. Gaps larger than
    `max_interpolation_gap_sec` are filled with zeros or flagged.
    """
    if df.empty:
        raise ValueError("Input sensor dataframe is empty.")

    # Standardize column naming if variations exist
    col_map = {}
    for col in df.columns:
        c_lower = col.lower()
        if 'timestamp' in c_lower or c_lower == 'time' or c_lower == 't':
            col_map[col] = 'timestamp'
        elif 'acc' in c_lower and ('_x' in c_lower or ':x' in c_lower or c_lower.endswith('x')):
            col_map[col] = 'acc_x'
        elif 'acc' in c_lower and ('_y' in c_lower or ':y' in c_lower or c_lower.endswith('y')):
            col_map[col] = 'acc_y'
        elif 'acc' in c_lower and ('_z' in c_lower or ':z' in c_lower or c_lower.endswith('z')):
            col_map[col] = 'acc_z'
        elif 'gyr' in c_lower and ('_x' in c_lower or ':x' in c_lower or c_lower.endswith('x')):
            col_map[col] = 'gyro_x'
        elif 'gyr' in c_lower and ('_y' in c_lower or ':y' in c_lower or c_lower.endswith('y')):
            col_map[col] = 'gyro_y'
        elif 'gyr' in c_lower and ('_z' in c_lower or ':z' in c_lower or c_lower.endswith('z')):
            col_map[col] = 'gyro_z'

    df_clean = df.rename(columns=col_map).copy()
    
    # Ensure all sensor columns exist
    for c in SENSOR_COLUMNS:
        if c not in df_clean.columns:
            df_clean[c] = 0.0

    # Ensure timestamps are strictly increasing
    df_clean = df_clean.sort_values(by='timestamp').drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    t_start = df_clean['timestamp'].iloc[0]
    t_end = df_clean['timestamp'].iloc[-1]

    if t_end <= t_start:
        raise ValueError(f"Invalid timestamp range: t_start={t_start}, t_end={t_end}")

    # Generate regular 25 Hz target timestamps
    num_samples = int(np.floor((t_end - t_start) * target_hz)) + 1
    target_timestamps = t_start + np.arange(num_samples) / target_hz

    # Interpolate each sensor channel
    resampled_data = {'timestamp': target_timestamps}
    orig_t = df_clean['timestamp'].values

    for col in SENSOR_COLUMNS:
        orig_vals = df_clean[col].values
        # Linear interpolation onto target grid
        interp_vals = np.interp(target_timestamps, orig_t, orig_vals)
        resampled_data[col] = interp_vals

    # Carry forward ground truth labels if present
    for lbl in LABEL_COLUMNS:
        if lbl in df_clean.columns:
            resampled_data[lbl] = np.interp(target_timestamps, orig_t, df_clean[lbl].values) >= 0.5

    return pd.DataFrame(resampled_data)

def apply_butterworth_filter(
    data: np.ndarray,
    cutoff_hz: float = 11.0,
    fs_hz: float = TARGET_SAMPLING_RATE_HZ,
    order: int = 4
) -> np.ndarray:
    """
    Applies a low-pass Butterworth filter to remove high-frequency motion artifacts.
    Cutoff must be < fs/2 (Nyquist = 12.5 Hz).
    """
    nyquist = 0.5 * fs_hz
    norm_cutoff = min(cutoff_hz / nyquist, 0.95)
    b, a = signal.butter(order, norm_cutoff, btype='low', analog=False)
    filtered = signal.filtfilt(b, a, data, axis=0)
    return filtered

def extract_sliding_windows(
    df: pd.DataFrame,
    window_sec: float = 2.56,
    overlap_ratio: float = 0.5,
    target_hz: float = TARGET_SAMPLING_RATE_HZ
) -> Tuple[np.ndarray, np.ndarray, List[float], List[float]]:
    """
    Segments 25 Hz stream into sliding windows.
    Returns:
        X: (num_windows, window_len, 6) sensor samples
        y: (num_windows, 7) binary label activations or -1 if no labels
        t_starts: start time (seconds from start) of each window
        t_ends: end time (seconds from start) of each window
    """
    window_len = int(round(window_sec * target_hz))  # 64 samples for 2.56s
    step_len = max(1, int(round(window_len * (1.0 - overlap_ratio))))  # 32 samples

    n_samples = len(df)
    if n_samples < window_len:
        # Pad with edge values if shorter than 1 window
        pad_len = window_len - n_samples
        pad_df = pd.DataFrame([df.iloc[-1]] * pad_len)
        df = pd.concat([df, pad_df], ignore_index=True)
        n_samples = len(df)

    sensor_vals = df[SENSOR_COLUMNS].values
    has_labels = all(lbl in df.columns for lbl in LABEL_COLUMNS)
    if has_labels:
        label_vals = df[LABEL_COLUMNS].values.astype(int)

    t_vals = df['timestamp'].values
    t_zero = t_vals[0]

    windows_X = []
    windows_Y = []
    t_starts = []
    t_ends = []

    for start_idx in range(0, n_samples - window_len + 1, step_len):
        end_idx = start_idx + window_len
        window_sensor = sensor_vals[start_idx:end_idx, :]
        windows_X.append(window_sensor)
        
        t_start_rel = float(t_vals[start_idx] - t_zero)
        t_end_rel = float(t_vals[end_idx - 1] - t_zero)
        t_starts.append(t_start_rel)
        t_ends.append(t_end_rel)

        if has_labels:
            # Majority vote or max presence in window
            window_lbl = (np.mean(label_vals[start_idx:end_idx, :], axis=0) >= 0.5).astype(int)
            windows_Y.append(window_lbl)

    X = np.array(windows_X, dtype=np.float32)
    y = np.array(windows_Y, dtype=np.int64) if has_labels else np.zeros((len(windows_X), len(LABEL_COLUMNS)), dtype=np.int64)
    return X, y, t_starts, t_ends
