"""
Temporal Interval Aggregation and Symbolic Timeline Builder.
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Converts noisy, window-level HAR predictions into clean, continuous
symbolic activity intervals with temporal boundaries, confidence,
and physical kinetic descriptors.
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from src.preprocessing import LABEL_COLUMNS

class ActivityInterval:
    """Represents a continuous segment of a detected activity."""
    def __init__(
        self,
        activity: str,
        start_sec: float,
        end_sec: float,
        confidence: float = 1.0,
        dom_freq_hz: float = 0.0,
        acc_var: float = 0.0,
        gyro_var: float = 0.0
    ):
        self.activity = activity
        self.start_sec = round(start_sec, 1)
        self.end_sec = round(end_sec, 1)
        self.duration_sec = max(0.0, round(self.end_sec - self.start_sec, 1))
        self.confidence = round(confidence, 3)
        self.dom_freq_hz = round(dom_freq_hz, 2)
        self.acc_var = round(acc_var, 4)
        self.gyro_var = round(gyro_var, 4)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'activity': self.activity,
            'start_sec': self.start_sec,
            'end_sec': self.end_sec,
            'duration_sec': self.duration_sec,
            'confidence': self.confidence,
            'dom_freq_hz': self.dom_freq_hz,
            'acc_var': self.acc_var,
            'gyro_var': self.gyro_var
        }

class ActivityTimeline:
    """
    Symbolic timeline of all detected activities across a recording.
    Provides deterministic methods for Task 1, 2, and 3 reasoning.
    """
    def __init__(self, intervals: List[ActivityInterval], total_duration_sec: float):
        self.intervals = sorted(intervals, key=lambda x: x.start_sec)
        self.total_duration_sec = total_duration_sec

    def get_intervals_for(self, activity: str) -> List[ActivityInterval]:
        """Returns all intervals matching the activity name (case-insensitive)."""
        act_clean = activity.lower().replace(" ", "_")
        return [iv for iv in self.intervals if iv.activity.lower().replace(" ", "_") == act_clean]

    def get_total_duration(self, activity: str) -> float:
        """Returns cumulative duration in seconds for an activity."""
        ivs = self.get_intervals_for(activity)
        return round(sum(iv.duration_sec for iv in ivs), 1)

    def get_count(self, activity: str) -> int:
        """Returns the number of distinct bouts/occurrences."""
        return len(self.get_intervals_for(activity))

    def get_onset(self, activity: str) -> Optional[float]:
        """Returns the start time of the first occurrence, or None."""
        ivs = self.get_intervals_for(activity)
        if ivs:
            return ivs[0].start_sec
        return None

    def compare_activities(self, act1: str, act2: str) -> Dict[str, Any]:
        """Compares total durations between two activities."""
        d1 = self.get_total_duration(act1)
        d2 = self.get_total_duration(act2)
        if d1 > d2:
            winner = act1
        elif d2 > d1:
            winner = act2
        else:
            winner = "Equal"
        return {
            'winner': winner,
            'durations': {act1: d1, act2: d2},
            'diff_sec': round(abs(d1 - d2), 1)
        }

    def get_dominant_activity(self) -> str:
        """Returns the activity with the highest cumulative time."""
        totals = {}
        for act in LABEL_COLUMNS:
            dur = self.get_total_duration(act)
            if dur > 0:
                totals[act] = dur
        if not totals:
            return "sitting"
        return max(totals, key=totals.get)

def build_timeline_from_predictions(
    y_probs: np.ndarray,
    t_starts: List[float],
    t_ends: List[float],
    kinetic_features: Optional[List[Dict[str, float]]] = None,
    threshold: float = 0.5,
    min_interval_sec: float = 4.0
) -> ActivityTimeline:
    """
    Transforms window probabilities into merged activity intervals.
    Applies hysteresis and temporal merging.
    """
    num_windows, num_classes = y_probs.shape
    total_dur = t_ends[-1] if len(t_ends) > 0 else 0.0
    
    # Identify active classes per window
    binary_preds = (y_probs >= threshold).astype(int)

    # For each class, find contiguous runs
    all_intervals: List[ActivityInterval] = []

    for c_idx, class_name in enumerate(LABEL_COLUMNS):
        in_run = False
        run_start = 0.0
        run_confs = []
        run_freqs = []
        run_acc_vars = []
        run_gyro_vars = []

        for w_idx in range(num_windows):
            is_active = (binary_preds[w_idx, c_idx] == 1)
            
            # Extract features if provided
            dom_f = kinetic_features[w_idx].get('dom_freq_hz', 0.0) if kinetic_features else 0.0
            a_var = kinetic_features[w_idx].get('acc_var_mag', 0.0) if kinetic_features else 0.0
            g_var = kinetic_features[w_idx].get('gyro_var_mag', 0.0) if kinetic_features else 0.0

            if is_active:
                if not in_run:
                    in_run = True
                    run_start = t_starts[w_idx]
                run_confs.append(float(y_probs[w_idx, c_idx]))
                run_freqs.append(dom_f)
                run_acc_vars.append(a_var)
                run_gyro_vars.append(g_var)
            else:
                if in_run:
                    run_end = t_ends[w_idx - 1]
                    dur = run_end - run_start
                    if dur >= min_interval_sec:
                        interval = ActivityInterval(
                            activity=class_name,
                            start_sec=run_start,
                            end_sec=run_end,
                            confidence=float(np.mean(run_confs)) if run_confs else 1.0,
                            dom_freq_hz=float(np.mean(run_freqs)) if run_freqs else 0.0,
                            acc_var=float(np.mean(run_acc_vars)) if run_acc_vars else 0.0,
                            gyro_var=float(np.mean(run_gyro_vars)) if run_gyro_vars else 0.0
                        )
                        all_intervals.append(interval)
                    in_run = False
                    run_confs, run_freqs, run_acc_vars, run_gyro_vars = [], [], [], []

        # Close run if still active at end
        if in_run:
            run_end = t_ends[-1]
            dur = run_end - run_start
            if dur >= min_interval_sec:
                all_intervals.append(ActivityInterval(
                    activity=class_name,
                    start_sec=run_start,
                    end_sec=run_end,
                    confidence=float(np.mean(run_confs)) if run_confs else 1.0,
                    dom_freq_hz=float(np.mean(run_freqs)) if run_freqs else 0.0,
                    acc_var=float(np.mean(run_acc_vars)) if run_acc_vars else 0.0,
                    gyro_var=float(np.mean(run_gyro_vars)) if run_gyro_vars else 0.0
                ))

    return ActivityTimeline(intervals=all_intervals, total_duration_sec=total_dur)
