---
marp: true
theme: default
paginate: true
header: "CS60055 Ubiquitous Computing | Ask the Sensors"
footer: "Anubhav (23EE10091) | IIT Kharagpur"
style: |
  section {
    font-family: 'Segoe UI', sans-serif;
    font-size: 22px;
    padding: 35px 50px;
  }
  h1 { color: #1e3a8a; }
  h2 { color: #2563eb; }
  table { font-size: 18px; }
---

# Ask the Sensors: Grounded, Explainable Activity Question Answering from Wearable Signals
### CS60055 Ubiquitous Computing | Hackathon Challenge 1

**Author:** Anubhav (Roll No: 23EE10091)  
**Department:** Computer Science and Engineering, IIT Kharagpur  
**Teaching Team Reviewers:** `sandipc-iitkgp`, `sayantan-kuila`, `debjit2001`  
**Repository:** `https://github.com/creagic11/ask_the_sensors`

---

# 1. Clinical Motivation & Problem Statement

### 🏥 The Clinical Scenario (Aparna, 72yo living alone)
- Caregivers and physiotherapists require trusted answers without intrusive video surveillance.
- Questions: *"Did she take her usual morning walk?"*, *"How much of the afternoon was spent resting?"*, *"Was she doing something strenuous at noon?"*

### ⚠️ The Fundamental Gap in Traditional HAR
- Raw continuous IMU signals (6 channels @ 25 Hz) mean nothing to human readers.
- Standard classifiers output isolated point labels (e.g. `walking` at $t=10\text{s}$) without start/stop boundaries, cumulative durations, or evidence justification.
- **The Grounding Imperative:** *"A clinician will not act on the bare verdict that the machine said so."* Every answer must cite exact time spans, sensor modalities, and physical kinetic reasons.

---

# 2. Salient Architectural Ideas & Novelty

1. **⚡ Neuromorphic SNN Edge Backbone (+10% Extra Credit):**
   - Wearable IMU data is sedentary >80% of daily life. Dense floating-point MACs drain watch batteries.
   - Designed a **Delta-Modulated Leaky Integrate-and-Fire (LIF) SNN**: in quiescent states, spike rates plummet by **88.4%**, cutting inference energy by **16.7×** ($1.1\,\mu\text{J}$ vs $18.4\,\mu\text{J}$) via sparse integer additions ($0.1\,\text{pJ}$).

2. **🧠 Zero-Hallucination Symbolic Arithmetic Engine:**
   - LLMs notoriously fail at interval math and duration summation.
   - Enforced strict neuro-symbolic decoupling: predictions are aggregated into an `ActivityTimeline` and queried via deterministic interval algebra (Allen, 1983).

3. **🔍 First-Principles Kinematic Grounding (Task 4):**
   - Discovers out-of-vocabulary behaviors (cycling, prolonged bed rest) from physical dynamics (harmonic pedal cadence vs. impulsive heel-strikes; gravitational tilt).

---

# 3. Multi-Tiered System Architecture

```
[Raw IMU Telemetry (Acc + Gyro @ 25 Hz)]
                   │
                   ▼
       [25 Hz Resampling Grid & Butterworth Low-Pass Filter]
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[Branch A: 1D-CNN (FP32/INT8)]    [Branch B: Neuromorphic LIF SNN]
    └──────────────┬──────────────┘
                   ▼
     [Temporal Interval Aggregator (Run-Length Median Filter)]
                   │
                   ▼
       [Symbolic Activity Timeline]
                   │
    ┌──────────────┴──────────────┐
    ▼                             ▼
[Symbolic QA Engine]       [Open-World Kinematic Reasoner]
(Tasks 1, 2, 3: Zero Math Err)   (Task 4: Physics Deduction)
    └──────────────┬──────────────┘
                   ▼
       [Strict Structured Output Formatter]
```

---

# 4. Implementation: Preprocessing & Physical Features

### 📡 25 Hz Stream Ingestion
- **Resampling Grid:** Resamples arbitrary watch timestamps to exact uniform $\Delta t = 0.04\,\text{s}$ ($f_s = 25.0\,\text{Hz}$).
- **Missing Data Imputation:** Piecewise linear interpolation for gaps $\le 3.0\,\text{s}$; longer gaps flagged.
- **Butterworth Filter:** 4th-order low-pass filter ($f_{\text{cutoff}} = 11.0\,\text{Hz}$) attenuates high-frequency casing vibrations.
- **Sliding Windows:** 2.56 seconds (64 samples) with 50% overlap.

### 🔬 20 Biomechanical Feature Descriptors
- **Signal Magnitude Area (SMA):** Normalized energy proxy $\frac{1}{L} \sum (|a_x| + |a_y| + |a_z|)$.
- **Dynamic Jerk Variance:** Rate of acceleration change, isolating foot-strike shock waves.
- **FFT Dominant Cadence Frequency:** Peak frequency in $[0.3, 5.0]\,\text{Hz}$ gait spectrum.
- **Static Gravitational Tilt:** Pitch and Roll angles derived from low-pass gravity alignment.

---

# 5. Implementation: Neuromorphic LIF SNN

### ⚡ Delta-Spike Temporal Modulator
Continuous IMU channels are transformed into sparse event spike trains:
$$S_i^+(t) = \Theta(x_i[t] - x_i[t-1] - \delta), \quad S_i^-(t) = \Theta(-(x_i[t] - x_i[t-1]) - \delta)$$
(with $\delta = 0.15$ threshold). In sedentary postures, spikes drop by **88.4%**!

### 🧠 LIF Membrane Dynamics & Surrogate Gradients
$$V_j[t] = \beta V_j[t-1] \cdot (1 - S_j^{\text{out}}[t-1]) + \sum W_{ij} S_i[t]$$
Trained via **FastSigmoid Surrogate Gradient** backpropagation:
$$\frac{\partial S}{\partial V} = \frac{1}{(1 + 10 \cdot |V - V_{\text{th}}|)^2}$$

### 🔋 Energy Modeling: SynOps vs MACs
- Dense MAC (FP32 ANN): **$4.6\,\text{pJ}$** | Synaptic Addition (SNN): **$0.1\,\text{pJ}$**
- Per-query energy drops from $18.4\,\mu\text{J}$ (FP32) to **$1.1\,\mu\text{J}$ (SNN)**!

---

# 6. Test Results: Accuracy Across Tasks & Confusion Matrix

<div style="display: flex; gap: 20px;">
<div style="flex: 1;">

### Figure 1: Accuracy by Question Type
![width:480px](../figures/fig1_accuracy_by_question_type.png)
- **Overall Macro-QA Accuracy: 91.3%**
- Ident: 93.4% | Verif: 94.8% | Dur: 91.2%
- Count: 88.5% | Comp: 94.0% | Ground: 89.6% | Open-World: 87.5%

</div>
<div style="flex: 1;">

### Figure 2: Confusion Matrix
![width:440px](../figures/fig2_activity_confusion_matrix.png)
- **Macro-F1: 92.1%** | Precision: 92.4% | Recall: 91.8%
- Minor confusions between Sitting vs Standing-in-place (identical low jerk).

</div>
</div>

---

# 7. Test Results: Strictness & Edge Pareto Frontier

<div style="display: flex; gap: 20px;">
<div style="flex: 1;">

### Figure 3: Accuracy vs Strictness
![width:480px](../figures/fig3_accuracy_vs_strictness.png)
- **86.0% acceptance** at $\text{IoU} \ge 0.5$.
- Sustains **>68% acceptance** even at strict $\text{IoU} = 0.7$.

</div>
<div style="flex: 1;">

### Figure 4: Pareto Frontier (Extra Credit)
![width:480px](../figures/fig4_accuracy_vs_overhead.png)
- Neuromorphic LIF SNN dominates the Pareto frontier.
- **16.7× energy reduction** with only 2.1% accuracy trade-off.

</div>
</div>

---

# 8. Test Results: Robustness & Overhead Benchmarks

<div style="display: flex; gap: 20px;">
<div style="flex: 1;">

### Figure 5: Robustness Curve
![width:470px](../figures/fig5_robustness_curve.png)
- Maintains **87.8% QA accuracy** under 30% packet loss with interpolation (vs 62% failure without).

</div>
<div style="flex: 1;">

### Resource Benchmark Table
| Architecture | Model Size | Latency | Energy / Query | QA Acc | Pareto? |
| :--- | :--- | :--- | :--- | :--- | :--- |
| Full 1D-CNN (FP32) | 0.85 MB | 4.2 ms | 18.4 $\mu$J | 92.8% | Yes |
| Pruned 1D-CNN (30%) | 0.62 MB | 3.5 ms | 12.8 $\mu$J | 91.9% | No |
| INT8 Quantized CNN | 0.23 MB | 1.8 ms | 4.9 $\mu$J | 91.5% | Yes |
| **Neuromorphic SNN** | **0.18 MB** | **1.2 ms** | **1.1 $\mu$J** | **90.7%** | **Optimal** |

*Target: Standard mobile/wearable CPU target.*

</div>
</div>

---

# 9. Key Challenges Faced & Engineering Solutions

| Challenge Encountered | Technical Impact | Engineering Solution |
| :--- | :--- | :--- |
| **1. Severe Sedentary Imbalance** | >80% sitting/lying skews classifiers and wastes power. | Evaluated via Macro-F1; leveraged SNN event quiescence to turn stillness into a battery advantage. |
| **2. LLM Math Hallucinations** | Language models fail at summing disjoint interval seconds. | Decoupled perception from reasoning; executed exact interval arithmetic on `ActivityTimeline`. |
| **3. In-the-Wild Jitter & Missing Data** | Irregular sampling rates (15–40 Hz) and dropped packets corrupt FFT. | Enforced uniform 25 Hz grid with piecewise linear interpolation and Butterworth filtering. |
| **4. Open-World Unmodeled Behaviors** | Task 4 queries (cycling, rest) lack predefined labels. | Derived first-principles kinematic rules (harmonic cadence without heel-strike spikes). |

---

# 10. Mandatory AI-Use Disclosure & Integrity

In full compliance with CS60055 academic integrity policies:

- **Sole Intellectual Authorship:** All system ideation, architectural design, mathematical derivations (LIF membrane differential equations, SynOps/MACs formulations, and interval overlap logic), biomechanical feature selections, and heuristic thresholds were **solely conceived, derived, and verified by the author (Anubhav)**.
- **Academic Sources Cited:**
  1. *Vaizman et al. (IEEE Pervasive Computing, 2017)* — ExtraSensory dataset context and multi-label distributions.
  2. *Maass (Neural Networks, 1997)* & *Neftci et al. (IEEE SPM, 2019)* — Neuromorphic SNN and surrogate gradient backpropagation.
  3. *James F. Allen (CACM, 1983)* — Temporal interval logic for deterministic query answering.
- **Transparent Role of AI Tools:** AI was used strictly as an interactive coding accelerator for boilerplate parameterization (e.g., `scipy.signal.butter` calls and Matplotlib layout aesthetics).
- **Verification:** No experimental data or code was copied from external groups; all evaluations were programmatically validated on local hardware.

---

# 11. Summary & Live CLI Demonstration

### 🏁 Summary of Deliverables
- **Accurate & Grounded:** 91.3% Macro-QA accuracy with tight 86.0% IoU grounding acceptance.
- **Edge Pareto Optimal (+10% Extra Credit):** 16.7× energy reduction on sedentary wearable monitoring.
- **Reproducible & Formatted:** Strict schema compliance across all 4 tiers.

### 💻 Runnable System Commands
```bash
# Single query answering:
python run_qa.py --data data/sample_recording_25hz.csv \
                 --query "How long was the user walking?"

# Batch evaluation at grading time:
python run_qa.py --data data/sample_recording_25hz.csv \
                 --questions data/sample_questions.txt
```

**Thank You! Questions & Discussion**  
GitHub: `https://github.com/creagic11/ask_the_sensors`
