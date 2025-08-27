import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import os
import torch
import torch.nn as nn
import numpy as np
# Set academic plot style
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (8, 6),
    'savefig.dpi': 300,
    'lines.linewidth': 2,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})

# Academic color palette (colorblind-friendly, inspired by Tableau or ColorBrewer)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Load data from a JSON file (replace "your_data.json" with the actual path to your JSON file)
# Load data from a JSON file (replace "your_data.json" with the actual path to your JSON file)
save_path = 'results/Quantum_Harmonic_Oscillator_1D_Compare'

results_file = os.path.join(save_path, 'results_QHO_1D.json')
with open(results_file, "r") as f:
    data = json.load(f)
selected_n = [0]  # Adjust as needed
filtered_data = [entry for entry in data if not selected_n or entry["n"] in selected_n]
# contine to filter
filtered_data = [entry for entry in filtered_data if entry["technique"] in ["BC"]]

# filter for L2_error < 1e-2
# filtered_data = [entry for entry in filtered_data if entry["L2_error"] < 1e-2]

# # filter the data to one entry per method with the lowest L2_error
# best_entries = {}
# for entry in filtered_data:
#     method = entry["method"]
#     if method not in best_entries or entry["L2_error"] < best_entries[method]["L2_error"]:
#         best_entries[method] = entry
# filtered_data = list(best_entries.values())

# print the number of entries after filtering
print(f"Number of entries after filtering: {len(filtered_data)}")
# Prepare figures
fig_loss, ax_loss = plt.subplots()
fig_l2, ax_l2 = plt.subplots()

# To handle multiple entries, we'll plot all with unique labels
for idx, entry in enumerate(filtered_data):
    method = entry["method"]
    n = entry["n"]
    technique = entry["technique"]
    label = f"{method} (n={n}, {technique})"
    color = colors[idx % len(colors)]

    # Load losses and L2 errors from .npy files
    losses_path = entry["losses"]
    l2_errors_path = entry["L2_errors"]
    losses = np.load(losses_path)
    l2_errors = np.load(l2_errors_path)

    # Assume the arrays are over epochs (starting from 0)
    epochs = np.arange(len(losses))

    # Plot losses (semilogy for academic style, as losses often span orders of magnitude)
    ax_loss.semilogy(epochs, losses, label=label, color=color)

    # Plot L2 errors (semilogy)
    ax_l2.semilogy(epochs, l2_errors, label=label, color=color)

# Configure loss plot
ax_loss.set_xlabel('Epochs')
ax_loss.set_ylabel('Loss')
ax_loss.set_title('Loss Evolution During Training')
ax_loss.legend(loc='upper right', frameon=True, shadow=True)
fig_loss.tight_layout()
fig_loss.savefig('loss_evolution.pdf', bbox_inches='tight')

# Configure L2 error plot
ax_l2.set_xlabel('Epochs')
ax_l2.set_ylabel('L2 Error')
ax_l2.set_title('L2 Error Evolution During Training')
ax_l2.legend(loc='upper right', frameon=True, shadow=True)
fig_l2.tight_layout()
fig_l2.savefig('l2_error_evolution.pdf', bbox_inches='tight')

# Note: For "predicted along training", assuming this refers to the evolution of predictions (e.g., wavefunction or eigenvalue),
# but no direct data is available in the JSON (e.g., no arrays for predicted values over epochs).
# If you have checkpoint models or additional .npy files for predictions, you can extend the code below.
# Here's a placeholder for plotting the final predicted wavefunction (requires PyTorch and model definition).

import math
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import argparse
import torch
import torch.nn as nn
from collections import defaultdict
from mpl_toolkits.axes_grid1.inset_locator import inset_axes, mark_inset
# Set academic plot style
rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman'],
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'legend.fontsize': 12,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'figure.figsize': (8, 6),
    'savefig.dpi': 300,
    'lines.linewidth': 1,
    'axes.grid': True,
    'grid.linestyle': '--',
    'grid.alpha': 0.5,
})

# Academic color palette (colorblind-friendly, inspired by Tableau or ColorBrewer)
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']

# Define the neural network classes
class FCN(nn.Module):
    def __init__(self, layers, num_states=1, L=1.0):
        super(FCN, self).__init__()
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        self.num_states = num_states
        self.L = L
        self.init_weights()

    def forward(self, x):
        return self.net(x)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

class FCN_WAN(nn.Module):
    def __init__(self, layers, num_states=1, L=1.0, enforce_bc=False):
        super(FCN_WAN, self).__init__()
        self.enforce_bc = enforce_bc
        self.num_states = num_states
        self.L = L
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        self.init_weights()
        self.energies = nn.Parameter(torch.tensor(Energy(num_states), dtype=torch.float32))
    def forward(self, x):
        y = self.net(x)
        if self.enforce_bc:
            return x * (self.L - x) * y
        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

# Exact solution function
def Exact_solution(L, n, x):
    return torch.sqrt(torch.tensor(2.0/L, dtype=x.dtype, device=x.device)) * torch.sin(n * torch.pi * x / L)
def phys_hermite(n, x):
    """
    Compute the physicists' Hermite polynomial H_n(x)
    by recurrence, entirely in PyTorch.
    """
    if n == 0:
        return torch.ones_like(x)
    elif n == 1:
        return 2 * x
    H_nm2 = torch.ones_like(x)    # H_0
    H_nm1 = 2 * x                  # H_1
    for k in range(2, n + 1):
        H_n = 2 * x * H_nm1 - 2 * (k - 1) * H_nm2
        H_nm2, H_nm1 = H_nm1, H_n
    return H_n

def Exact_solution_QHO(n, x, omega=math.sqrt(2)):
    # Compute H_n(√ω x) in torch
    Hn = phys_hermite(n, torch.sqrt(torch.tensor(omega)) * x)
    # Normalization (norm is a scalar)
    norm = (omega / np.pi) ** 0.25 / math.sqrt(2 ** n * math.factorial(n))
    # Use torch.exp so it works on tensors
    return norm * Hn * torch.exp(-omega * x**2 / 2)
def Potential(x, omega=math.sqrt(2)):
    return 0.5 * omega**2 * x ** 2
# Define the neural network for the PINN
def Energy(n, omega=math.sqrt(2)):
    # Energy levels for the quantum harmonic oscillator
    return (n + 0.5) * omega
class SineActivation(nn.Module):
    def forward(self, x):
        return torch.sin(x)
class FCN(nn.Module):
    def __init__(self, layers):
        """
        layers: list of neuron numbers for each layer, e.g. [1, 50, 50, 1]
        """
        super(FCN, self).__init__()
        self.activation = SineActivation()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
            
    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.activation(layer(x))
        x = self.layers[-1](x)
        return x
class FCN_Single(nn.Module):
    def __init__(self, layers, num_states=1, domain_length=20.0, enforce_bc=False):
        """
        layers: list of neuron numbers for each layer, e.g. [1, 50, 50, 1]
        num_states: number of states to be predicted
        """
        super(FCN_Single, self).__init__()
        self.net = FCN(layers)
        self.num_states = num_states
        self.domain_length = domain_length
        self.energies = nn.Parameter(torch.tensor(Energy(num_states), dtype=torch.float32))
        self.enforce_bc = enforce_bc
    def forward(self, x):
        L = self.domain_length/2.0
        trial = (1 - torch.exp(-(x + L))) * (1 - torch.exp(x - L))
        raw = self.net(x)
        if self.enforce_bc:
            raw = raw * trial
        return raw

# Group data by n
grouped_data = defaultdict(list)
for entry in filtered_data:
    grouped_data[entry["n"]].append(entry)

# Define network layers (assumed consistent across models; adjust if needed)
layers = [1, 200, 200, 200, 1]
L = 12.0  # Domain length consistent with plot

# Define zoom regions for insets
# First inset: near boundary (e.g., for BC detail)
zoom_x_min_boundary = - L/2
zoom_x_max_boundary = - 0.9 * L/2

# Second inset: middle region (e.g., around the peak for n=1)
zoom_x_min_middle = -0.1 * L/2
zoom_x_max_middle = 0.1 * L/2

# Create a plot for each n
for n in sorted(grouped_data.keys()):
    fig_pred, ax_pred = plt.subplots()

    # Generate x values and exact solution for this n
    x = np.linspace(-6, 6, 1000)
    x_tensor = torch.tensor(x.reshape(-1, 1), dtype=torch.float32)
    exact_psi = Exact_solution_QHO(n, x_tensor).numpy().flatten()
    ax_pred.plot(x, exact_psi, label='Exact', color='black', linestyle='--')

    # Create first inset axes (boundary, lower left)
    ax_inset_boundary = inset_axes(ax_pred, width="35%", height="35%", loc=8)  # loc=3: lower left

    ax_inset_boundary.plot(x, exact_psi, color='black', linestyle='--')
    ax_inset_boundary.set_xticks([])
    ax_inset_boundary.set_yticks([])
    # Hide the axes for the inset

    # Create second inset axes (middle, upper center)
    ax_inset_middle = inset_axes(ax_pred, width="35%", height="35%", loc=2)  # loc=9: upper center
    ax_inset_middle.plot(x, exact_psi, color='black', linestyle='--')
    ax_inset_middle.set_xticks([])
    ax_inset_middle.set_yticks([])
    #Prepare to collect y min/max for insets
    mask_boundary = (x >= zoom_x_min_boundary) & (x <= zoom_x_max_boundary)
    mask_middle = (x >= zoom_x_min_middle) & (x <= zoom_x_max_middle)
    y_mins_boundary = [exact_psi[mask_boundary].min()]
    y_maxs_boundary = [exact_psi[mask_boundary].max()]
    y_mins_middle = [exact_psi[mask_middle].min()]
    y_maxs_middle = [exact_psi[mask_middle].max()]

    # Plot predictions for each entry with this n
    for idx, entry in enumerate(grouped_data[n]):
        method = entry["method"]
        technique = entry["technique"]
        label = f"{method} (n={n}, {technique})"
        color = colors[idx % len(colors)]

        # Determine if enforce_bc based on weight_bc (hard if weight_bc == 0, else soft)

        # Load best model (use FCN_WAN for flexibility with enforce_bc)
        model_path = entry["best_model_path"]
        if method == "WAN":
            print(f"Using WAN model for method {method}, n={n}")
            model = FCN_WAN(layers=layers, num_states=n, L=L, enforce_bc=False)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        if method == "PINN" or method == "DRM":
            model = FCN_Single(layers=layers, num_states=n, domain_length= L, enforce_bc=False)
            model.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
        model.eval()

        with torch.no_grad():
            pred_psi = model(x_tensor).numpy().flatten()

        # Plot in main axes
        ax_pred.plot(x, pred_psi, label=label, color=color)
        # Plot in boundary inset
        ax_inset_boundary.plot(x, pred_psi, color=color)

        # Plot in middle inset
        ax_inset_middle.plot(x, pred_psi, color=color)

        # Update y min/max for insets
        y_mins_boundary.append(pred_psi[mask_boundary].min())
        y_maxs_boundary.append(pred_psi[mask_boundary].max())
        y_mins_middle.append(pred_psi[mask_middle].min())
        y_maxs_middle.append(pred_psi[mask_middle].max())

    # Set boundary inset limits
    ax_inset_boundary.set_xlim(zoom_x_min_boundary, zoom_x_max_boundary)
    y_range_boundary = max(y_maxs_boundary) - min(y_mins_boundary)
    ax_inset_boundary.set_ylim(min(y_mins_boundary) - 0.05 * y_range_boundary, max(y_maxs_boundary) + 0.05 * y_range_boundary)

    # Set middle inset limits
    ax_inset_middle.set_xlim(zoom_x_min_middle, zoom_x_max_middle)
    y_range_middle = max(y_maxs_middle) - min(y_mins_middle)
    ax_inset_middle.set_ylim(min(y_mins_middle) - 0.05 * y_range_middle, max(y_maxs_middle) + 0.05 * y_range_middle)

    # # Add markers to indicate inset regions
    mark_inset(ax_pred, ax_inset_boundary, loc1=2, loc2=4, fc="none", ec="0.5")
    mark_inset(ax_pred, ax_inset_middle, loc1=1, loc2=3, fc="none", ec="0.5")

    # Configure main plot
    ax_pred.set_xlabel('x')
    ax_pred.set_ylabel(r'$\psi(x)$')
    ax_pred.set_title(f'Predicted Wavefunction for n={n}')
    ax_pred.legend(loc='upper right', frameon=True, shadow=True)
    # Removed tight_layout to avoid compatibility issues with insets
    fig_pred.savefig(f'predicted_wavefunction_n{n}_QHO.pdf', bbox_inches='tight')

print("Plots have been generated and saved as PDF files.")