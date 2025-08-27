import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# torch.manual_seed(0)
# np.random.seed(0)
import json
import datetime
from time import time
import scipy.special as sp
import math
save_path = 'results/Quantum_Harmonic_Oscillator_1D'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_file = os.path.join(save_path, 'results_QHO_1D.json')
if not os.path.exists(results_file):
    with open(results_file, 'w') as f:
        json.dump([], f)  # Initialize with an empty list
        print(f"Created results file at {results_file}")

# problem
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
def Exact_solution(n, x, omega=math.sqrt(2)):
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
    def __init__(self, layers, num_states=1, domain_length=20.0, enforce_bc=False, FN=False):
        """
        layers: list of neuron numbers for each layer, e.g. [1, 50, 50, 1]
        num_states: number of states to be predicted
        """
        super(FCN_Single, self).__init__()
        self.net = FCN(layers)
        self.num_states = num_states
        self.domain_length = domain_length
        # self.energies = nn.Parameter(torch.tensor(Energy(num_states), dtype=torch.float32))
        self.enforce_bc = enforce_bc
    def forward(self, x):
        L = self.domain_length/2.0
        trial = (1 - torch.exp(-(x + L))) * (1 - torch.exp(x - L))
        raw = self.net(x)
        if self.enforce_bc:
            raw = raw * trial
        if self.FN:
            raw = raw * trial

        return raw

class FCN_Single(nn.Module):
    def __init__(self, layers, num_states=1, domain_length=20.0, enforce_bc=False, FN=False):
        """
        layers: list of neuron numbers for each layer, e.g. [1, 50, 50, 1]
        num_states: number of states to be predicted
        domain_length: length of the domain [-L, L]
        enforce_bc: if True, enforce boundary conditions at x = ±L
        FN: if True, enforce nodes for quantum harmonic oscillator states
        """
        super(FCN_Single, self).__init__()
        self.net = FCN(layers)  # Assumes FCN is defined elsewhere
        self.num_states = num_states
        self.domain_length = domain_length
        self.enforce_bc = enforce_bc
        self.FN = FN
        self.energies = nn.Parameter(torch.tensor(Energy(num_states), dtype=torch.float32))
        
        # Node positions for n = 1 to 5 for quantum harmonic oscillator (m=1, ℏ=1, ω=√2)
        scale = 2 ** (-1/4)  # ≈ 0.841
        self.nodes = {
            1: torch.tensor([0.0], dtype=torch.float32),
            2: torch.tensor([-2**(-3/4), 2**(-3/4)], dtype=torch.float32),  # ±0.595
            3: torch.tensor([0.0, -2**(-3/4)*math.sqrt(3), 2**(-3/4)*math.sqrt(3)], dtype=torch.float32),  # 0, ±1.030
            4: torch.tensor([
                -scale*math.sqrt((3+math.sqrt(6))/2),
                -scale*math.sqrt((3-math.sqrt(6))/2),
                scale*math.sqrt((3-math.sqrt(6))/2),
                scale*math.sqrt((3+math.sqrt(6))/2)
            ], dtype=torch.float32),  # ±0.441, ±1.388
            5: torch.tensor([
                0.0,
                -scale*math.sqrt((5+math.sqrt(10))/2),
                -scale*math.sqrt((5-math.sqrt(10))/2),
                scale*math.sqrt((5-math.sqrt(10))/2),
                scale*math.sqrt((5+math.sqrt(10))/2)
            ], dtype=torch.float32)  # 0, ±0.806, ±1.699
        }

    def forward(self, x):
        L = self.domain_length / 2.0  # e.g., L = 10.0
        output = self.net(x)  # Shape: [batch_size, num_states]
        
        if self.FN and self.num_states in self.nodes:
            # Node-enforcing trial function for the n-th state
            nodes = self.nodes[self.num_states].to(x.device)  # Get nodes for current state
            trial = torch.ones_like(x)
            for node in nodes:
                trial = trial * (x - node)  # Enforce zero at each node
            # Optionally enforce boundary conditions
            if self.enforce_bc:
                trial = trial * (1 - torch.exp(-(x + L))) * (1 - torch.exp(x - L))
            output = output * trial  # Apply trial function
        elif self.enforce_bc:
            # Original boundary condition enforcement
            trial = (1 - torch.exp(-(x + L))) * (1 - torch.exp(x - L))
            output = output * trial
        # Else, return raw output
        return output
def compute_derivatives(psi, x):
    psi_x = torch.autograd.grad(psi, x, grad_outputs=torch.ones_like(psi),
                                create_graph=True, retain_graph=True)[0]
    psi_xx = torch.autograd.grad(psi_x, x, grad_outputs=torch.ones_like(psi_x),
                                 create_graph=True, retain_graph=True)[0]
    return psi_x, psi_xx
def PINN_loss(model, x):
    """
    Compute the loss for the PINN based on the Rayleigh quotient.
    The loss is defined as the mean squared error of the trial function's gradient
    divided by the mean squared value of the trial function.
    """
    # PDE Loss
    u_interior = model(x)

    du_dx, d2u_dx2 = compute_derivatives(u_interior, x)

    # residual = -0.5 * d2u_dx2 + Potential(x) * u_interior - model.energies * u_interior
    residual = -0.5 * d2u_dx2 + Potential(x) * u_interior - Energy(model.num_states) * u_interior
    return torch.mean(residual**2)

def DRM_loss(model, x):
    """
    Compute the DRM loss for the trial function.
    The loss is defined as the mean squared value of the trial function's gradient.
    """
    u_interior = model(x)
    du_dx, _ = compute_derivatives(u_interior, x)
    numerator = 0.5 * du_dx**2 + Potential(x) * u_interior**2
    denominator = u_interior**2
    return numerator.mean() / denominator.mean()

def normalization_loss(model, x):
    """
    Compute the normalization loss for the trial function.
    The loss is defined as the mean squared value of the trial function.
    """
    u_interior = model(x)
    dx = x[1] - x[0]  # Assuming uniform spacing
    norm = torch.sqrt(torch.sum(u_interior**2) * dx)
    return (norm - 1)**2

def Orthogonal_loss(model, x, n, domain_length):
    #Sample points for computing the orthogonal loss
    u_pred = model(x)
    if n == 0:
        return torch.tensor(0.0, device=x.device)  # No orthogonal loss for n=0
    else:
        ortho_loss = 0.0
        for k in range(0, n):
            u_exact = Exact_solution(k, x)
            # Approximate inner product <u_pred, u_exact> ≈ mean(u_pred * u_exact) * domain_length
            inner = torch.mean(u_pred * u_exact) *  2 * domain_length
            # Approximate norm squared <u_exact, u_exact> ≈ mean(u_exact**2) * domain_length
            norm_sq = torch.mean(u_exact ** 2) * 2 * domain_length
            # Add the squared projection coefficient
            ortho_loss += (inner ** 2) / norm_sq
        return ortho_loss

def train_pinn_single(n, X_max=6, epochs=3000, lr=0.001, layers=[1, 200, 200, 200, 1], LBFGS=False, method='DRM', technique='BC'):
    # Define the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    # sample points
    x_interior = torch.linspace(-X_max, X_max, 1000).view(-1, 1).to(device)
    x_interior.requires_grad = True  # Enable gradients for interior points

    x_bc = torch.tensor([[-X_max], [X_max]], dtype=torch.float32).to(device)
    u_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float32).to(device)

    x_full_data = torch.linspace(-X_max, X_max, 1000).view(-1, 1).to(device)
    u_full_data = Exact_solution(n, x_full_data)

    # set the percentage of data to use for training
    percentage = 0
    # Sample a subset of the data for training
    n_data = int(percentage * x_full_data.shape[0])

    x_data = x_full_data[n_data: 2*n_data:10]  # Sample every 10th point for training
    u_data = u_full_data[n_data: 2*n_data:10]
    # model
    if technique == 'BC':
        model = FCN_Single(layers, num_states=n, domain_length=2 * X_max, enforce_bc=False).to(device)
    else:
        model = FCN_Single(layers, num_states=n, domain_length=2 * X_max, enforce_bc=True).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if LBFGS:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=100, line_search_fn='strong_wolfe')
    if technique == 'OG':
        weight_orth = 100.0
    else:
        weight_orth = 0.0
    weight_data = 1000.0
    if model.enforce_bc:
        weight_bc = 0.0
    else:
        weight_bc = 10.0  # Force the boundary condition
    if method == 'DRM':
        weight_pde = 0.0
        weight_drm = 10.0
        weight_norm = 10.0
    elif method == 'PINN':
        weight_pde = 10.0
        weight_drm = 0.0
        weight_norm = 10.0
    Losses = []
    L2_errors = []
    best_L2 = float('inf')
    best_state = None
    best_epoch = -1
    pbar = tqdm(range(epochs))
    start_time = time()
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        u_data_pred = model(x_data)
        data_loss = torch.mean((u_data_pred - u_data)**2)
        bc_loss = torch.mean((model(x_bc) - u_bc)**2)  # Boundary condition loss
        # PDE Loss
        u_interior = model(x_interior)
        total_loss = weight_pde * PINN_loss(model, x_interior) + \
                weight_drm * DRM_loss(model, x_interior) + \
            weight_norm * normalization_loss(model, x_interior) + \
                weight_data * data_loss + \
                    weight_orth * Orthogonal_loss(model, x_interior, n, model.domain_length) + \
                    weight_bc * bc_loss

        total_loss.backward()
        optimizer.step()
        Losses.append(total_loss.item())
        # pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4e}, Energy: {model.energies.item():.4f}")

        with torch.no_grad():
            u_test = model(x_full_data)
            u_exact = Exact_solution(n, x_full_data)
            # sometimes the model can predict negative values, so we take the minimum of the L2 error
            L2_error = torch.mean((u_test - u_exact) ** 2).item()
            L2_errors.append(L2_error)
            if L2_error < best_L2:
                best_L2 = L2_error
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

    end_time = time()
    print(f'Training for {method} completed in {end_time - start_time:.2f} seconds')

    # Save the best model state
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if best_state is not None:
        model_filename = f'{method}_best_model_n{n}_{technique}_{timestamp}.pth'
        torch.save(best_state, os.path.join(save_path, model_filename))
        print(f'Best model saved for {method} n={n} with L2 error {best_L2:.4e} at epoch {best_epoch} as {model_filename}')
    # Save Losses and L2_errors as numpy arrays
    losses_filename = f'{method}_losses_n{n}_{technique}_{timestamp}.npy'
    l2_errors_filename = f'{method}_L2_errors_n{n}_{technique}_{timestamp}.npy'
    np.save(os.path.join(save_path, losses_filename), np.array(Losses))
    np.save(os.path.join(save_path, l2_errors_filename), np.array(L2_errors))
    L2_error = min(L2_errors)
    # find the order of the minimum L2 error
    min_epoch = L2_errors.index(L2_error)
    print(f'Minimum L2 Error for n={n}: {L2_error:.4e} at epoch {min_epoch}')
    import json

    results = {
        'method': method,
        'n': n,
        'epochs': epochs,
        'LBFGS': LBFGS,
        'L2_error': L2_error,
        'min_epoch': min_epoch,
        'best_model_path': os.path.join(save_path, model_filename),
        'losses': os.path.join(save_path, losses_filename),
        'L2_errors': os.path.join(save_path, l2_errors_filename),
        'time': end_time - start_time,
        'time_of_best_model': best_epoch * (end_time - start_time) / epochs,
        'technique': technique,
        'weight_pde': weight_pde,
        'weight_drm': weight_drm,
        'weight_data': weight_data,
        'weight_norm': weight_norm,
        'weight_bc': weight_bc,
        'weight_orth': weight_orth,
        'percentage': percentage,
        'timestamp': timestamp
    }
    # Load existing data if the file exists
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        if not isinstance(all_results, list):
            all_results = [all_results]  # Convert to list if it was a single dict before
    else:
        all_results = []

    # Append the new results
    all_results.append(results)

    # Save back to the file
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)
    # load the best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("No best state found, using the final model state.")
    model.eval()
    with torch.no_grad():
        u_test = model(x_full_data)
        u_exact = Exact_solution(n, x_full_data)
    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_full_data.cpu().numpy(), u_test.cpu().numpy(), label=f'{method} Solution', color='blue')
    plt.plot(x_full_data.cpu().numpy(), u_exact.cpu().numpy(), label='Exact Solution', color='red', linestyle='--')
    plt.scatter(x_data.cpu().numpy(), u_data.cpu().numpy(), color='orange', s=10, label='Training Data', alpha=0.5)
    plt.title(f'Solution for n={n}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'{method}_solution_n{n}_seperate_method_{technique}_{timestamp}.png'))
    # energies = model.energies.item()
    # print(f"Best model for n={n} at epoch {best_epoch} with L2 error {best_L2:.4e}, energy {energies:.4f}, true energy {Energy(n):.4f}")
    print(f"Best model for n={n} at epoch {best_epoch} with L2 error {best_L2:.4e}")
    return model, Losses, L2_errors


def train_pinn_single(n, X_max=6, epochs=3000, lr=0.001, layers=[1, 200, 200, 200, 1], LBFGS=False, method='DRM', technique='BC'):
    """
    Train a PINN for the quantum harmonic oscillator for state n.
    
    Parameters:
    - n: Quantum state number (1, 2, 3, ...)
    - X_max: Domain half-length [-X_max, X_max]
    - epochs: Number of training epochs
    - lr: Learning rate
    - layers: Neural network architecture, e.g., [1, 200, 200, 200, 1]
    - LBFGS: If True, use LBFGS optimizer; else, use Adam
    - method: 'DRM' or 'PINN' for loss computation
    - technique: 'BC' (boundary conditions), 'OG' (orthogonal), 'FN' (forced nodes)
    
    Returns:
    - model: Trained model
    - Losses: List of total loss per epoch
    - L2_errors: List of L2 errors per epoch
    """
    # Define the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    # Sample points
    x_interior = torch.linspace(-X_max, X_max, 1000).view(-1, 1).to(device)
    x_interior.requires_grad = True  # Enable gradients for interior points

    x_bc = torch.tensor([[-X_max], [X_max]], dtype=torch.float32).to(device)
    u_bc = torch.tensor([[0.0], [0.0]], dtype=torch.float32).to(device)

    x_full_data = torch.linspace(-X_max, X_max, 1000).view(-1, 1).to(device)
    u_full_data = Exact_solution(n, x_full_data)

    # Set the percentage of data to use for training
    percentage = 0.25
    n_data = int(percentage * x_full_data.shape[0])
    x_data = x_full_data[n_data: 2*n_data:10]  # Sample every 10th point
    u_data = u_full_data[n_data: 2*n_data:10]

    # Model initialization
    if technique == 'BC':
        model = FCN_Single(layers, num_states=n, domain_length=2 * X_max, enforce_bc=False, FN=False).to(device)
    elif technique == 'OG':
        model = FCN_Single(layers, num_states=n, domain_length=2 * X_max, enforce_bc=True, FN=False).to(device)
    elif technique == 'FN':
        model = FCN_Single(layers, num_states=n, domain_length=2 * X_max, enforce_bc=True, FN=True).to(device)
    else:
        raise ValueError(f"Unknown technique: {technique}. Choose 'BC', 'OG', or 'FN'.")

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if LBFGS:
        optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, history_size=100, line_search_fn='strong_wolfe')

    # Loss weights
    if technique == 'OG':
        weight_orth = 100.0
    else:
        weight_orth = 0.0
    weight_data = 1000.0
    weight_bc = 0.0 if model.enforce_bc or model.FN else 10.0  # Reduce BC loss if nodes/boundaries are enforced
    if method == 'DRM':
        weight_pde = 0.0
        weight_drm = 10.0
        weight_norm = 10.0
    elif method == 'PINN':
        weight_pde = 10.0
        weight_drm = 0.0
        weight_norm = 10.0
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'DRM' or 'PINN'.")

    Losses = []
    L2_errors = []
    best_L2 = float('inf')
    best_state = None
    best_epoch = -1
    pbar = tqdm(range(epochs))
    start_time = time()
    for epoch in pbar:
        model.train()
        optimizer.zero_grad()
        u_data_pred = model(x_data)
        data_loss = torch.mean((u_data_pred - u_data)**2)
        bc_loss = torch.mean((model(x_bc) - u_bc)**2)  # Boundary condition loss
        # PDE Loss
        u_interior = model(x_interior)
        total_loss = (
            weight_pde * PINN_loss(model, x_interior) +
            weight_drm * DRM_loss(model, x_interior) +
            weight_norm * normalization_loss(model, x_interior) +
            weight_data * data_loss +
            weight_orth * Orthogonal_loss(model, x_interior, n, model.domain_length) +
            weight_bc * bc_loss
        )

        total_loss.backward()
        optimizer.step()
        Losses.append(total_loss.item())
        pbar.set_description(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss.item():.4e}, Energy: {model.energies.item():.4f}")

        with torch.no_grad():
            u_test = model(x_full_data)
            u_exact = Exact_solution(n, x_full_data)
            L2_error = torch.mean((u_test - u_exact) ** 2).item()
            L2_errors.append(L2_error)
            if L2_error < best_L2:
                best_L2 = L2_error
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch

    end_time = time()
    print(f'Training for {method} completed in {end_time - start_time:.2f} seconds')

    # Save the best model state
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    if best_state is not None:
        model_filename = f'{method}_best_model_n{n}_{technique}_{timestamp}.pth'
        torch.save(best_state, os.path.join(save_path, model_filename))
        print(f'Best model saved for {method} n={n} with L2 error {best_L2:.4e} at epoch {best_epoch} as {model_filename}')

    # Save Losses and L2_errors as numpy arrays
    losses_filename = f'{method}_losses_n{n}_{technique}_{timestamp}.npy'
    l2_errors_filename = f'{method}_L2_errors_n{n}_{technique}_{timestamp}.npy'
    np.save(os.path.join(save_path, losses_filename), np.array(Losses))
    np.save(os.path.join(save_path, l2_errors_filename), np.array(L2_errors))
    L2_error = min(L2_errors)
    min_epoch = L2_errors.index(L2_error)
    print(f'Minimum L2 Error for n={n}: {L2_error:.4e} at epoch {min_epoch}')

    # Save results to JSON
    results = {
        'method': method,
        'n': n,
        'epochs': epochs,
        'LBFGS': LBFGS,
        'L2_error': L2_error,
        'min_epoch': min_epoch,
        'best_model_path': os.path.join(save_path, model_filename),
        'losses': os.path.join(save_path, losses_filename),
        'L2_errors': os.path.join(save_path, l2_errors_filename),
        'time': end_time - start_time,
        'time_of_best_model': best_epoch * (end_time - start_time) / epochs,
        'technique': technique,
        'weight_pde': weight_pde,
        'weight_drm': weight_drm,
        'weight_data': weight_data,
        'weight_norm': weight_norm,
        'weight_bc': weight_bc,
        'weight_orth': weight_orth,
        'percentage': percentage,
        'timestamp': timestamp
    }
    if os.path.exists(results_file):
        with open(results_file, 'r') as f:
            all_results = json.load(f)
        if not isinstance(all_results, list):
            all_results = [all_results]
    else:
        all_results = []
    all_results.append(results)
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=4)

    # Load the best model state
    if best_state is not None:
        model.load_state_dict(best_state)
    else:
        print("No best state found, using the final model state.")
    model.eval()
    with torch.no_grad():
        u_test = model(x_full_data)
        u_exact = Exact_solution(n, x_full_data)

    # Plot the results
    plt.figure(figsize=(10, 6))
    plt.plot(x_full_data.cpu().numpy(), u_test.cpu().numpy(), label=f'{method} Solution', color='blue')
    plt.plot(x_full_data.cpu().numpy(), u_exact.cpu().numpy(), label='Exact Solution', color='red', linestyle='--')
    plt.scatter(x_data.cpu().numpy(), u_data.cpu().numpy(), color='orange', s=10, label='Training Data', alpha=0.5)
    plt.title(f'Solution for n={n}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'{method}_solution_n{n}_separate_method_{technique}_{timestamp}.png'))
    plt.close()
    energies = model.energies.item()
    print(f"Best model for n={n} at epoch {best_epoch} with L2 error {best_L2:.4e}, energy {energies:.4f}, true energy {Energy(n):.4f}")
    return model, Losses, L2_errors
def run_seperate_method(n_values, epochs=3000, LBFGS=False):
    for n in n_values:
        print(f"Training for n={n}...")
        model = train_pinn_single(n, epochs=epochs, LBFGS=LBFGS, method='PINN', technique='FN')
        model_drm = train_pinn_single(n, epochs=epochs, LBFGS=LBFGS, method='DRM', technique='FN')
        print(f"Training completed for n={n}.")
        # Save results


if __name__ == "__main__":
    n_values = [0]  # You can adjust this list for more eigenstates
    run_seperate_method(n_values, epochs=10000, LBFGS=False)
