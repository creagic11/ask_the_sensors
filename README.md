# Ask the Sensors: Grounded, Explainable Activity Question Answering from Wearable Signals

**CS60055 Ubiquitous Computing | Hackathon Challenge 1**  
*Department of Computer Science and Engineering, IIT Kharagpur*

---

## 🌟 Overview

**Ask the Sensors** is an end-to-end question-answering system operating over 6-channel continuous wearable sensor recordings (triaxial accelerometer and triaxial gyroscope resampled to 25 Hz). It bridges the gap between raw motion dynamics and clinical explainability across 4 evaluation tiers:

1. **Task 1: Activity Identification** (Open identification and binary verification).
2. **Task 2: Temporal & Quantitative Reasoning** (Durations, counts, intervals, and comparative activities across continuous recordings).
3. **Task 3: Evidence Grounding** (Temporal span matching evaluated via Intersection-over-Union (IoU), sensor modality, channel attribution, and physical kinetic explanation).
4. **Task 4: Open-World Semantic Reasoning** (Semantic reasoning over unlabeled or out-of-distribution motion behaviors such as wheeled locomotion/cycling or sustained recumbency).

### ⚡ Neuromorphic Spiking Neural Network (SNN) Innovation
To address resource constraints on wearable edge devices, this project introduces a **Delta-Encoded Leaky Integrate-and-Fire (LIF) Spiking Neural Network**. During sedentary behaviors (sitting, lying down), motion derivatives remain quiescent, causing spike rates to plummet by up to **88%** and reducing dynamic energy consumption from dense floating-point Multiply-Accumulates ($E_{\text{MAC}} \approx 4.6\,\text{pJ}$) to sparse synaptic additions ($E_{\text{AC}} \approx 0.1\,\text{pJ}$). I demonstrate a clear **Accuracy vs. Overhead Pareto Frontier** comparing FP32 1D-CNN, INT8 Quantized CNN, and the neuromorphic SNN.

---

## 🏗️ Architecture

```
[Raw Acc + Gyro @ 25Hz] ──► [2.56s Sliding Window + Physics Featurizer]
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
     [Branch A: Continuous CNN (FP32/INT8)]     [Branch B: Neuromorphic LIF SNN]
                 └──────────────────────┬──────────────────────┘
                                        ▼
                         [Temporal Interval Aggregator]
                       (Median Filtered Activity Timeline)
                                        │
                 ┌──────────────────────┴──────────────────────┐
                 ▼                                             ▼
    [Symbolic Quantitative Engine]                [Kinematic Open-World Reasoner]
    (Tasks 1, 2, 3: Zero-Hallucination)           (Task 4: Physical Body Dynamics)
                 │                                             │
                 └──────────────────────┬──────────────────────┘
                                        ▼
                             [Strict Formatted Output]
```

---

## 📋 Strict Output Format

All query responses conform to the following schema:
```text
Answer: <direct answer to the query, or N/A>
Activity/Event: <activity or event, or N/A>
Evidence:
  Timestamp(s): <time range or ranges in seconds from start, or N/A>
  Sensor Modality: <accelerometer, gyroscope, or both, or N/A>
  Sensor Channel(s): <for example Acc X/Y/Z, Gyro X/Y/Z, or All, or N/A>
Explanation: <reasoning grounded in the observed signal, or N/A>
```

---

## 🚀 Quickstart & Reproducibility

### 1. Environment Setup
```bash
git clone <repo_url>
cd ask_the_sensors
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Fetch Sample Sensor Trace
```bash
python -m data.download_sample
```

### 3. Run Query Answering
```bash
# Single natural-language query:
python run_qa.py --data data/sample_recording_25hz.csv --query "How long was the user walking?"
python run_qa.py --data data/sample_recording_25hz.csv --query "Did the user begin running at any point, and if so, when?"
python run_qa.py --data data/sample_recording_25hz.csv --query "Was the user using a wheeled or pedal-based mode of movement?"

# Batch evaluation over questions file (at evaluation time):
python run_qa.py --data data/sample_recording_25hz.csv --questions data/sample_questions.txt
```

### 4. Generate the 5 Mandatory Figures
```bash
python generate_figures.py
```
This produces:
- `figures/fig1_accuracy_by_question_type.png`
- `figures/fig2_activity_confusion_matrix.png`
- `figures/fig3_accuracy_vs_strictness.png`
- `figures/fig4_accuracy_vs_overhead.png` (Pareto Frontier)
- `figures/fig5_robustness_curve.png`

---

## 📄 Technical Report
The complete technical report detailing the mathematical formulation, neuromorphic SNN edge architecture, benchmark evaluations across all 4 tasks, and academic integrity disclosures is available:
- **Compiled PDF Report:** [`report/ANUBHAV_23EE10091.pdf`](report/ANUBHAV_23EE10091.pdf)
- **LaTeX Source:** [`report/CS60055_Report.tex`](report/CS60055_Report.tex)
- **Markdown Report:** [`report/CS60055_Report.md`](report/CS60055_Report.md)

---

## 👥 Teaching Team Access & Collaboration
Teaching team members with access:
- `sandipc-iitkgp`
- `sayantan-kuila`
- `debjit2001`
