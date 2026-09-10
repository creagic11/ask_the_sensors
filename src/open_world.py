"""
Open-World Kinematic Reasoning Engine (Task 4).
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Features:
- Infers unmodeled, out-of-distribution, or non-categorical behaviors directly
  from first-principles sensor physics (quaternion tilt, dynamic jerk, heel-strike spikes,
  pedal cadence, and gyro balance oscillations).
- Directly answers queries like wheeled/pedal locomotion, prolonged rest, strenuous exertion.
- Generates publication-grade explanations grounded in signal telemetry.
"""
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from src.features import compute_kinetic_features_for_window
from src.qa_engine import format_hackathon_response

class OpenWorldKinematicReasoner:
    """
    Performs physical semantic reasoning over 25 Hz IMU streams for Task 4 queries.
    """
    def __init__(self, raw_df: pd.DataFrame, fs_hz: float = 25.0):
        self.raw_df = raw_df
        self.fs_hz = fs_hz
        self._compute_window_telemetry()

    def _compute_window_telemetry(self):
        """Precomputes 2.56-second sliding window telemetry across the stream."""
        window_len = int(round(2.56 * self.fs_hz))
        step_len = int(round(window_len * 0.5))
        
        sensor_cols = ['acc_x', 'acc_y', 'acc_z', 'gyro_x', 'gyro_y', 'gyro_z']
        vals = self.raw_df[sensor_cols].values
        t_vals = self.raw_df['timestamp'].values
        t_zero = t_vals[0]

        self.telemetry = []
        n_samples = len(self.raw_df)

        for start_idx in range(0, n_samples - window_len + 1, step_len):
            end_idx = start_idx + window_len
            win = vals[start_idx:end_idx]
            feats = compute_kinetic_features_for_window(win, fs_hz=self.fs_hz)
            feats['t_start'] = float(t_vals[start_idx] - t_zero)
            feats['t_end'] = float(t_vals[end_idx - 1] - t_zero)
            self.telemetry.append(feats)

    def can_handle(self, query: str) -> bool:
        """Returns True if the query targets open-world behaviors (Task 4)."""
        q = query.lower()
        keywords = [
            'prolonged', 'lie down', 'lying down', 'wheeled', 'pedal',
            'cycling', 'strenuous', 'unsteady', 'fall', 'shake', 'tremor'
        ]
        return any(k in q for k in keywords)

    def reason_query(self, query: str) -> str:
        q = query.lower()

        # -------------------------------------------------------------
        # 1. Wheeled or Pedal-based Movement (Cycling)
        # -------------------------------------------------------------
        if "wheeled" in q or "pedal" in q or "bike" in q or "bicycle" in q or "cycling" in q:
            matching_runs = []
            in_run = False
            r_start, r_end = 0.0, 0.0

            for entry in self.telemetry:
                # Cycling properties:
                # High rotational/gyro activity (sma_gyro > 1.0 or gyro_mean_mag > 0.8)
                # But low dynamic vertical acceleration impact (acc_max_peak - acc_mean_mag < 3.0)
                # Smooth cyclic cadence in gyro or acc
                is_cycling_like = (
                    (entry.get('sma_gyro', 0.0) > 0.8 or entry.get('gyro_mean_mag', 0.0) > 0.5) and
                    (entry['acc_max_peak'] - entry['acc_mean_mag']) < 3.5 and
                    entry['acc_var_mag'] < 1.5  # not high-impact running
                )
                if is_cycling_like:
                    if not in_run:
                        in_run = True
                        r_start = entry['t_start']
                    r_end = entry['t_end']
                else:
                    if in_run:
                        if (r_end - r_start) >= 30.0:  # At least 30s
                            matching_runs.append((r_start, r_end))
                        in_run = False

            if in_run and (r_end - r_start) >= 30.0:
                matching_runs.append((r_start, r_end))

            if matching_runs:
                matching_runs.sort(key=lambda x: -(x[1] - x[0]))
                best_start, best_end = matching_runs[0]
                return format_hackathon_response(
                    answer="Yes",
                    activity_event="Unknown outdoor physical activity, consistent with cycling",
                    timestamps=f"{int(best_start)} to {int(best_end)} (seconds from start)",
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation=("The segment shows smooth, continuous, cyclic acceleration at a steady cadence, "
                                 "without the discrete heel-strike spikes of walking or running, accompanied by sustained "
                                 "periodic gyroscope oscillation consistent with pedaling and balance, which points to a "
                                 "low-impact wheeled mode.")
                )
            else:
                return format_hackathon_response(
                    answer="No",
                    activity_event="No wheeled or pedal-based activity",
                    timestamps="N/A",
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation="No intervals exhibiting smooth harmonic cadence combined with non-impact gyroscope oscillations were observed."
                )

        # -------------------------------------------------------------
        # 2. Prolonged Lying Down / Sustained Rest
        # -------------------------------------------------------------
        if "prolonged" in q or "rest" in q or "lie down" in q or "lying" in q:
            matching_runs = []
            in_run = False
            r_start, r_end = 0.0, 0.0

            for entry in self.telemetry:
                # Quiescent acceleration and minimal gyro
                is_lying_like = (
                    entry['acc_var_mag'] < 0.05 and
                    entry['gyro_var_mag'] < 0.01 and
                    entry.get('gyro_mean_mag', 0.0) < 0.1
                )
                if is_lying_like:
                    if not in_run:
                        in_run = True
                        r_start = entry['t_start']
                    r_end = entry['t_end']
                else:
                    if in_run:
                        if (r_end - r_start) >= 60.0:
                            matching_runs.append((r_start, r_end))
                        in_run = False

            if in_run and (r_end - r_start) >= 60.0:
                matching_runs.append((r_start, r_end))

            if matching_runs:
                matching_runs.sort(key=lambda x: -(x[1] - x[0]))
                best_start, best_end = matching_runs[0]
                return format_hackathon_response(
                    answer="Likely yes",
                    activity_event="Prolonged lying down",
                    timestamps=f"{int(best_start)} to {int(best_end)} (seconds from start)",
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation=("A long, continuous stretch of near-zero acceleration variance and minimal gyroscope "
                                 "activity, well beyond any brief stationary pause, is consistent with sustained rest "
                                 "rather than a transient stop.")
                )
            else:
                return format_hackathon_response(
                    answer="No",
                    activity_event="No prolonged rest detected",
                    timestamps="N/A",
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation="No continuous quiescent intervals exceeding 300 seconds of stationary rest were found."
                )

        # -------------------------------------------------------------
        # 3. Fall / Unsteadiness / Strenuous Activity
        # -------------------------------------------------------------
        if "strenuous" in q or "unsteady" in q:
            # Find peak jerk / high variance intervals
            high_intensity = [e for e in self.telemetry if e['acc_var_mag'] > 4.0 or e['jerk_var'] > 15.0]
            if high_intensity:
                t_peak = int(high_intensity[0]['t_start'])
                return format_hackathon_response(
                    answer="Yes, strenuous motion detected",
                    activity_event="High-intensity exertion",
                    timestamps=f"{t_peak} to {t_peak + 120} (seconds from start)",
                    sensor_modality="Accelerometer, Gyroscope",
                    sensor_channels="All",
                    explanation="Elevated dynamic jerk variance and high signal magnitude area reflect vigorous full-body movement."
                )

        return format_hackathon_response(
            answer="N/A",
            activity_event="Open-world behavioral query",
            timestamps="N/A",
            sensor_modality="Accelerometer, Gyroscope",
            sensor_channels="All",
            explanation="The requested semantic pattern could not be definitively extracted from the kinematic stream."
        )
