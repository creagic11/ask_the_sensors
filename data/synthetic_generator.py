"""
Realistic Wearable Sensor Recording Generator.
Generates 25 Hz multi-channel (Acc X/Y/Z, Gyro X/Y/Z) traces with realistic
human movement kinematics, cadence, gravity vectors, and sensor noise.
Directly models the canonical scenario from the CS60055 hackathon brief!
"""
import numpy as np
import pandas as pd
from pathlib import Path

def generate_benchmark_session(output_path: str = "data/sample_recording_25hz.csv", total_duration_sec: float = 3600.0) -> pd.DataFrame:
    """
    Generates a 25 Hz sensor stream reproducing the exact timeline from the challenge brief:
    - 0 to 905s: Sitting / Resting
    - 905 to 1420s: Walking (515s) [1.8 Hz step frequency, Acc-Y dominance]
    - 1420 to 1512s: Standing & moving
    - 1512 to 1980s: Running (468s) [3.1 Hz cadence, strong heel strike peaks, large gyro oscillations]
    - 1980 to 2110s: Standing in place
    - 2110 to 2295s: Walking (185s) -> Total walking = 515 + 185 = 700s!
    - 2295 to 2400s: Sitting
    - 2400 to 3120s: Wheeled/pedal-based movement (Cycling) [Smooth cyclic cadence, no heel strike spikes, sustained periodic gyro]
    - 3120 to 3600s: Prolonged lying down [Near-zero acceleration variance, minimal gyro]
    """
    fs = 25.0
    dt = 1.0 / fs
    num_samples = int(total_duration_sec * fs)
    t = np.arange(num_samples) * dt

    # Initialize channels
    acc_x = np.random.normal(0, 0.05, num_samples)
    acc_y = np.random.normal(0, 0.05, num_samples)
    acc_z = np.random.normal(9.81, 0.05, num_samples)  # Default gravity along Z

    gyro_x = np.random.normal(0, 0.02, num_samples)
    gyro_y = np.random.normal(0, 0.02, num_samples)
    gyro_z = np.random.normal(0, 0.02, num_samples)

    labels = {
        'lying_down': np.zeros(num_samples, dtype=int),
        'sitting': np.zeros(num_samples, dtype=int),
        'standing_in_place': np.zeros(num_samples, dtype=int),
        'standing_and_moving': np.zeros(num_samples, dtype=int),
        'walking': np.zeros(num_samples, dtype=int),
        'running': np.zeros(num_samples, dtype=int),
        'bicycling': np.zeros(num_samples, dtype=int)
    }

    def get_indices(t_start, t_end):
        return (t >= t_start) & (t < t_end)

    # 1. Sitting: 0 to 905s
    idx_sit = get_indices(0, 905)
    labels['sitting'][idx_sit] = 1
    acc_x[idx_sit] += np.random.normal(0.1, 0.02, np.sum(idx_sit))
    acc_y[idx_sit] += 3.2  # Thigh/wrist tilt
    acc_z[idx_sit] = np.sqrt(max(0, 9.81**2 - 3.2**2)) + np.random.normal(0, 0.03, np.sum(idx_sit))

    # 2. Walking Interval 1: 905 to 1420s (1.8 Hz cadence)
    idx_walk1 = get_indices(905, 1420)
    labels['walking'][idx_walk1] = 1
    t_w1 = t[idx_walk1]
    acc_y[idx_walk1] += 2.8 * np.sin(2 * np.pi * 1.8 * t_w1) + 0.8 * np.sin(4 * np.pi * 1.8 * t_w1)
    acc_z[idx_walk1] = 9.81 + 1.9 * np.cos(2 * np.pi * 1.8 * t_w1)
    acc_x[idx_walk1] += 0.9 * np.sin(2 * np.pi * 0.9 * t_w1)
    gyro_x[idx_walk1] += 1.2 * np.sin(2 * np.pi * 1.8 * t_w1)
    gyro_y[idx_walk1] += 0.8 * np.cos(2 * np.pi * 1.8 * t_w1)

    # 3. Standing and Moving: 1420 to 1512s
    idx_sm = get_indices(1420, 1512)
    labels['standing_and_moving'][idx_sm] = 1
    t_sm = t[idx_sm]
    acc_z[idx_sm] = 9.81 + 0.6 * np.sin(2 * np.pi * 0.5 * t_sm)
    acc_y[idx_sm] += 0.5 * np.random.normal(0, 0.3, np.sum(idx_sm))

    # 4. Running: 1512 to 1980s (3.1 Hz cadence, heavy impacts)
    idx_run = get_indices(1512, 1980)
    labels['running'][idx_run] = 1
    t_r = t[idx_run]
    acc_y[idx_run] += 6.5 * np.sin(2 * np.pi * 3.1 * t_r) + 2.5 * np.sin(4 * np.pi * 3.1 * t_r)
    acc_z[idx_run] = 9.81 + 5.2 * np.cos(2 * np.pi * 3.1 * t_r)
    acc_x[idx_run] += 2.2 * np.sin(2 * np.pi * 1.55 * t_r)
    gyro_x[idx_run] += 3.8 * np.sin(2 * np.pi * 3.1 * t_r)
    gyro_y[idx_run] += 2.5 * np.cos(2 * np.pi * 3.1 * t_r)

    # 5. Standing in Place: 1980 to 2110s
    idx_sip = get_indices(1980, 2110)
    labels['standing_in_place'][idx_sip] = 1
    acc_z[idx_sip] = 9.81 + np.random.normal(0, 0.02, np.sum(idx_sip))

    # 6. Walking Interval 2: 2110 to 2295s
    idx_walk2 = get_indices(2110, 2295)
    labels['walking'][idx_walk2] = 1
    t_w2 = t[idx_walk2]
    acc_y[idx_walk2] += 2.7 * np.sin(2 * np.pi * 1.85 * t_w2)
    acc_z[idx_walk2] = 9.81 + 1.8 * np.cos(2 * np.pi * 1.85 * t_w2)
    gyro_x[idx_walk2] += 1.1 * np.sin(2 * np.pi * 1.85 * t_w2)

    # 7. Sitting: 2295 to 2400s
    idx_sit2 = get_indices(2295, 2400)
    labels['sitting'][idx_sit2] = 1
    acc_y[idx_sit2] += 3.2
    acc_z[idx_sit2] = np.sqrt(max(0, 9.81**2 - 3.2**2))

    # 8. Cycling (Wheeled / Pedal-based): 2400 to 3120s
    # Smooth continuous cyclic cadence (1.3 Hz pedaling), no impact spikes, rhythmic gyro
    idx_cyc = get_indices(2400, 3120)
    labels['bicycling'][idx_cyc] = 1
    t_cyc = t[idx_cyc]
    acc_y[idx_cyc] += 1.4 * np.sin(2 * np.pi * 1.3 * t_cyc)
    acc_x[idx_cyc] += 1.1 * np.cos(2 * np.pi * 1.3 * t_cyc)
    acc_z[idx_cyc] = 9.81 + 0.4 * np.sin(2 * np.pi * 2.6 * t_cyc) # very low vertical impact!
    gyro_y[idx_cyc] += 2.2 * np.sin(2 * np.pi * 1.3 * t_cyc)
    gyro_z[idx_cyc] += 1.8 * np.cos(2 * np.pi * 1.3 * t_cyc)

    # 9. Prolonged Lying Down: 3120 to 3600s
    # Gravity along X or Y (horizontal), near-zero variance, quiescent gyro
    idx_lie = get_indices(3120, 3600)
    labels['lying_down'][idx_lie] = 1
    acc_x[idx_lie] = 9.78 + np.random.normal(0, 0.005, np.sum(idx_lie)) # horizontal body alignment
    acc_y[idx_lie] = 0.2 + np.random.normal(0, 0.005, np.sum(idx_lie))
    acc_z[idx_lie] = 0.1 + np.random.normal(0, 0.005, np.sum(idx_lie))
    gyro_x[idx_lie] = np.random.normal(0, 0.002, np.sum(idx_lie))
    gyro_y[idx_lie] = np.random.normal(0, 0.002, np.sum(idx_lie))
    gyro_z[idx_lie] = np.random.normal(0, 0.002, np.sum(idx_lie))

    df = pd.DataFrame({
        'timestamp': t,
        'acc_x': np.round(acc_x, 4),
        'acc_y': np.round(acc_y, 4),
        'acc_z': np.round(acc_z, 4),
        'gyro_x': np.round(gyro_x, 4),
        'gyro_y': np.round(gyro_y, 4),
        'gyro_z': np.round(gyro_z, 4),
        **labels
    })

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Generated benchmark 25 Hz recording: {output_path} ({len(df)} samples, {total_duration_sec}s)")
    return df

if __name__ == "__main__":
    generate_benchmark_session()
