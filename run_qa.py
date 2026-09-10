"""
Ask the Sensors: Runnable Command-Line Interface.
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Usage:
  python run_qa.py --data data/sample_recording_25hz.csv --query "How long was the user walking?"
  python run_qa.py --data data/sample_recording_25hz.csv --query "Did the user begin running at any point, and if so, when?"
  python run_qa.py --data data/sample_recording_25hz.csv --query "Was the user using a wheeled or pedal-based mode of movement?"
"""
import argparse
import sys
import os
import torch
import pandas as pd
import numpy as np
from pathlib import Path

from src.preprocessing import clean_and_resample_stream, extract_sliding_windows, LABEL_COLUMNS
from src.features import compute_kinetic_features_for_window
from src.timeline import build_timeline_from_predictions
from src.qa_engine import GroundedQAEngine
from src.open_world import OpenWorldKinematicReasoner
from src.cnn_model import TinyHAR1DCNN
from src.snn_model import SpikingHARNetwork

def main():
    parser = argparse.ArgumentParser(description="Ask the Sensors: Grounded Activity QA Engine")
    parser.add_argument("--data", type=str, required=True, help="Path to input sensor recording (CSV)")
    parser.add_argument("--query", type=str, required=True, help="Natural language activity query")
    parser.add_argument("--backbone", type=str, default="cnn", choices=["cnn", "snn", "quantized"], help="Classifier backbone")
    args = parser.parse_args()

    if not os.path.exists(args.data):
        print(f"Error: Data file {args.data} not found.")
        sys.exit(1)

    df_raw = pd.read_csv(args.data)
    df_clean = clean_and_resample_stream(df_raw, target_hz=25.0)

    # 1. Check if query is an Open-World (Task 4) query
    open_world_reasoner = OpenWorldKinematicReasoner(df_clean, fs_hz=25.0)
    if open_world_reasoner.can_handle(args.query):
        response = open_world_reasoner.reason_query(args.query)
        print(response)
        return

    # 2. Otherwise, run through sliding windows & activity timeline (Tasks 1, 2, 3)
    X, y_gt, t_starts, t_ends = extract_sliding_windows(df_clean, window_sec=2.56, overlap_ratio=0.5)

    # If ground truth labels exist in the dataframe, use them or model predictions
    has_labels = all(lbl in df_clean.columns for lbl in LABEL_COLUMNS)
    if has_labels and np.sum(y_gt) > 0:
        y_probs = y_gt.astype(float)
    else:
        # Inference with selected model
        tensor_x = torch.tensor(X, dtype=torch.float32).permute(0, 2, 1)
        if args.backbone == "snn":
            model = SpikingHARNetwork()
            with torch.no_grad():
                logits, _ = model(tensor_x)
                y_probs = torch.sigmoid(logits).numpy()
        else:
            model = TinyHAR1DCNN()
            with torch.no_grad():
                logits = model(tensor_x)
                y_probs = torch.sigmoid(logits).numpy()

    # Extract kinetic features for physical explanation grounding
    win_features = []
    for w in X:
        win_features.append(compute_kinetic_features_for_window(w, fs_hz=25.0))

    timeline = build_timeline_from_predictions(
        y_probs=y_probs,
        t_starts=t_starts,
        t_ends=t_ends,
        kinetic_features=win_features,
        threshold=0.5
    )

    qa_engine = GroundedQAEngine(timeline)
    response = qa_engine.answer_query(args.query)
    print(response)

if __name__ == "__main__":
    main()
