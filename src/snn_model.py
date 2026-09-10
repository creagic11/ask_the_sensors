"""
Neuromorphic Spiking Neural Network (SNN) Module.
CS60055 Ubiquitous Computing | Hackathon Challenge 1

Features:
- Delta-modulation temporal spike encoder: converts continuous IMU signals
  into sparse event-driven spikes S(t) in {0, 1}.
- Leaky Integrate-and-Fire (LIF) spiking neurons with surrogate gradient
  (FastSigmoid) for direct gradient-based backpropagation.
- Synaptic Operation (SynOps) counter for computing event-driven energy savings:
  E_SNN = SynOps * 0.1 pJ  vs  E_ANN = MACs * 4.6 pJ.
- Delivers extreme energy efficiency during sedentary postures (sitting/lying down).
"""
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Tuple, Dict

class FastSigmoidSurrogate(torch.autograd.Function):
    """
    Fast Sigmoid surrogate gradient for non-differentiable Heaviside step function:
    Forward: S = 1 if V >= V_th else 0
    Backward: dS/dV = 1 / (1 + beta * |V - V_th|)^2
    """
    @staticmethod
    def forward(ctx, v, v_th=1.0, beta=10.0):
        ctx.save_for_backward(v)
        ctx.v_th = v_th
        ctx.beta = beta
        return (v >= v_th).float()

    @staticmethod
    def backward(ctx, grad_output):
        v, = ctx.saved_tensors
        v_diff = torch.abs(v - ctx.v_th)
        surrogate_grad = 1.0 / (1.0 + ctx.beta * v_diff)**2
        return grad_output * surrogate_grad, None, None

def spike_activation(v, v_th=1.0):
    return FastSigmoidSurrogate.apply(v, v_th)

class LIFCell(nn.Module):
    """
    Leaky Integrate-and-Fire (LIF) Spiking Neuron Cell.
    Membrane equation:
      V[t] = beta * V[t-1] * (1 - S[t-1]) + I[t]
      S[t] = spike_activation(V[t])
    """
    def __init__(self, in_features: int, out_features: int, decay_beta: float = 0.85, v_th: float = 1.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.decay_beta = decay_beta
        self.v_th = v_th
        self.fc = nn.Linear(in_features, out_features)

    def forward(self, x_spikes_t: torch.Tensor, v_prev: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # Synaptic input
        syn_in = self.fc(x_spikes_t)
        # Membrane update with soft/hard reset
        v_next = self.decay_beta * v_prev + syn_in
        s_out = spike_activation(v_next, self.v_th)
        # Reset membrane potential after spike
        v_next = v_next * (1.0 - s_out)
        return s_out, v_next

class DeltaSpikeEncoder(nn.Module):
    """
    Biologically-inspired Delta Modulator:
    Generates spikes whenever |x[t] - x[t-1]| exceeds threshold delta.
    Produces sparse event streams from continuous 25 Hz IMU data.
    """
    def __init__(self, delta_threshold: float = 0.15):
        super().__init__()
        self.delta = delta_threshold

    def forward(self, x_continuous: torch.Tensor) -> torch.Tensor:
        """
        Input: (batch, channels=6, time_steps=64)
        Output: (time_steps=64, batch, channels*2=12) [positive & negative spikes]
        """
        diff = torch.diff(x_continuous, dim=2, prepend=x_continuous[:, :, :1])
        pos_spikes = (diff > self.delta).float()
        neg_spikes = (diff < -self.delta).float()
        # Combine pos and neg spikes: (batch, 12, time_steps)
        spikes = torch.cat([pos_spikes, neg_spikes], dim=1)
        # Permute to (time_steps, batch, features) for recurrent temporal processing
        return spikes.permute(2, 0, 1)

class SpikingHARNetwork(nn.Module):
    """
    End-to-End Spiking Neural Network for Human Activity Recognition.
    Processes sparse temporal event trains and performs rate-coded readout.
    """
    def __init__(self, in_channels: int = 6, hidden_dim: int = 48, num_classes: int = 7, delta: float = 0.15):
        super().__init__()
        self.encoder = DeltaSpikeEncoder(delta_threshold=delta)
        self.lif1 = LIFCell(in_features=in_channels * 2, out_features=hidden_dim, decay_beta=0.85)
        self.lif2 = LIFCell(in_features=hidden_dim, out_features=num_classes, decay_beta=0.85)
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        x: (batch, window_len, 6) or (batch, 6, window_len)
        Returns:
            logits: (batch, num_classes)
            stats: spike rate and energy metrics
        """
        if x.dim() == 3 and x.size(1) != 6 and x.size(2) == 6:
            x = x.permute(0, 2, 1)

        batch_size = x.size(0)
        time_steps = x.size(2)

        # 1. Delta Spike Encoding
        in_spikes = self.encoder(x)  # (time_steps, batch, 12)

        # 2. LIF Temporal Integration
        device = x.device
        v1 = torch.zeros(batch_size, self.hidden_dim, device=device)
        v2 = torch.zeros(batch_size, self.num_classes, device=device)

        out_spikes_accum = torch.zeros(batch_size, self.num_classes, device=device)
        total_l1_spikes = 0
        total_l2_spikes = 0
        total_in_spikes = float(torch.sum(in_spikes).item())

        for t in range(time_steps):
            s1, v1 = self.lif1(in_spikes[t], v1)
            s2, v2 = self.lif2(s1, v2)
            out_spikes_accum += s2
            total_l1_spikes += float(torch.sum(s1).item())
            total_l2_spikes += float(torch.sum(s2).item())

        # Rate-coded output logits (average spikes per timestep)
        logits = out_spikes_accum / time_steps

        # Compute Neuromorphic SynOps vs Classical MACs
        # Dense MACs for equivalent network:
        # Layer 1: 12 * hidden_dim * time_steps
        # Layer 2: hidden_dim * num_classes * time_steps
        dense_macs = (12 * self.hidden_dim + self.hidden_dim * self.num_classes) * time_steps * batch_size
        
        # Event-driven Synaptic Operations (occur only on input/hidden spikes):
        synops = (total_in_spikes * self.hidden_dim) + (total_l1_spikes * self.num_classes)

        # Energy models:
        # E_MAC = 4.6 pJ (32-bit floating point MAC in 45nm CMOS)
        # E_AC  = 0.1 pJ (32-bit integer synaptic addition in neuromorphic hardware)
        e_ann_pj = dense_macs * 4.6
        e_snn_pj = synops * 0.1
        energy_reduction_ratio = (e_ann_pj / (e_snn_pj + 1e-9))

        stats = {
            'input_spikes': total_in_spikes,
            'hidden_spikes': total_l1_spikes,
            'synops': synops,
            'dense_macs': dense_macs,
            'energy_snn_pj': e_snn_pj,
            'energy_ann_pj': e_ann_pj,
            'energy_savings_ratio': energy_reduction_ratio
        }

        return logits, stats

def train_snn(
    model: SpikingHARNetwork,
    X_train: np.ndarray,
    y_train: np.ndarray,
    epochs: int = 15,
    batch_size: int = 32,
    lr: float = 2e-3
) -> Dict[str, float]:
    """Trains the neuromorphic SNN via surrogate gradient backpropagation."""
    model.train()
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    tensor_x = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
    tensor_y = torch.tensor(y_train, dtype=torch.float32)

    dataset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        for bx, by in loader:
            optimizer.zero_grad()
            logits, _ = model(bx)
            loss = criterion(logits, by)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(bx)

    model.eval()
    return {"final_loss": epoch_loss / len(X_train)}
