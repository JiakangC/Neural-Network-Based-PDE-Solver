from datetime import datetime
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
from time import time
import os

save_path = 'results/Infinite_Potential_Well_1D'
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_file = os.path.join(save_path, 'results_IPW_1D.json')
if not os.path.exists(results_file):
    with open(results_file, 'w') as f:
        json.dump([], f)
        print(f"Created results file at {results_file}")

def Exact_solution(L, n, x):
    return torch.sqrt(torch.tensor(2.0/L, dtype=x.dtype, device=x.device)) * torch.sin(n * torch.pi * x / L)

class FCN(nn.Module):
    def __init__(self, layers, num_states=1, L=2.0, enforce_bc=False, FN=False):
        super(FCN, self).__init__()
        self.enforce_bc = enforce_bc
        self.FN = FN
        self.num_states = num_states
        self.L = L
        modules = []
        for i in range(len(layers) - 2):
            modules.append(nn.Linear(layers[i], layers[i+1]))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*modules)
        self.init_weights()
        
        # Node positions for n = 1 to 10
        self.nodes = {
            n: torch.tensor([k * L / n for k in range(1, n)], dtype=torch.float32) for n in range(1, 11)
        }

    def forward(self, x):
        y = self.net(x)
        if self.FN and self.num_states in self.nodes:
            trial = x * (self.L - x)
            nodes = self.nodes[self.num_states].to(x.device)
            for node in nodes:
                trial = trial * (x - node)
            return y * trial
        elif self.enforce_bc:
            trial = x * (self.L - x)
            return y * trial
        return y

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

def PINN_loss(model, x, n, L):
    hbar = 1.0
    m = 1.0
    E = (n * np.pi * hbar) ** 2 / (2 * m * L**2)
    k_squared = (2 * m * E) / (hbar ** 2)
    u_interior = model(x)
    du_dx = torch.autograd.grad(
        outputs=u_interior,
        inputs=x,
        grad_outputs=torch.ones_like(u_interior),
        create_graph=True
    )[0]
    d2u_dx2 = torch.autograd.grad(
        outputs=du_dx,
        inputs=x,
        grad_outputs=torch.ones_like(du_dx),
        create_graph=True
    )[0]
    residual = d2u_dx2 + k_squared * u_interior
    pde_loss = torch.mean(residual**2)
    return pde_loss

def DRM_loss(model, x):
    u = model(x)
    u_x = torch.autograd.grad(
        u, x, grad_outputs=torch.ones_like(u), create_graph=True
    )[0]
    return torch.mean(u_x**2) / torch.mean(u**2)

def Orthogonal_loss(model, x, n, L):
    u_pred = model(x)
    if n == 1:  # No orthogonal loss for n=1 (no lower states)
        return torch.tensor(0.0, device=x.device)
    else:
        ortho_loss = 0.0
        for k in range(1, n):
            u_exact = Exact_solution(L, k, x)
            inner = torch.mean(u_pred * u_exact) * L  # Approximate <u_pred, u_exact>
            norm_sq = torch.mean(u_exact ** 2) * L  # Approximate <u_exact, u_exact>
            ortho_loss += (inner ** 2) / (norm_sq + 1e-8)  # Avoid division by zero
        return ortho_loss

def train_seperate(n, L=2.0, epochs=3000, lr=0.001, layers=[1, 50, 50, 50, 1], LBFGS=False, method='DRM', technique='FN'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)
    start_time = time()
    x_interior = torch.linspace(0, L, 1000).view(-1, 1).to(device)
    x_interior.requires_grad = True
    x_full_data = torch.linspace(0, L, 1000).view(-1, 1).to(device)
    u_full_data = Exact_solution(L, n, x_full_data)
    percentage = 0.25
    n_data = int(percentage * x_full_data.shape[0])
    x_data = x_full_data[0:n_data:10]
    u_data = u_full_data[0:n_data:10]
    if technique == 'BC':
        model = FCN(layers, num_states=n, L=L, enforce_bc=False, FN=False).to(device)
    elif technique == 'FBC':
        model = FCN(layers, num_states=n, L=L, enforce_bc=True, FN=False).to(device)
    elif technique == 'FN':
        model = FCN(layers, num_states=n, L=L, enforce_bc=True, FN=True).to(device)
    elif technique == 'OG':
        model = FCN(layers, num_states=n, L=L, enforce_bc=True, FN=False).to(device)
    else:
        raise ValueError(f"Unknown technique: {technique}. Choose 'BC', 'FBC', 'FN', or 'OG'.")
    weight_data = 10000.0
    weight_bc = 0.0 if model.enforce_bc or model.FN else 1000.0
    weight_orth = 1000.0 if technique == 'OG' else 0.0
    if method == 'DRM':
        weight_pde = 0.0
        weight_drm = 10.0
        weight_norm = 0.0
    elif method == 'PINN':
        weight_pde = 1.0
        weight_drm = 0.0
        weight_norm = 1.0
    else:
        raise ValueError(f"Unknown method: {method}. Choose 'DRM' or 'PINN'.")
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if LBFGS:
        optimizer_LBFGS = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=200, line_search_fn='strong_wolfe')
    Losses = []
    L2_errors = []
    best_L2 = float('inf')
    best_state = None
    best_epoch = -1
    x_bc = torch.tensor([[0.0], [L]], dtype=x_interior.dtype, device=device)
    u_bc = torch.tensor([[0.0], [0.0]], dtype=x_interior.dtype, device=device)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()
        u_data_pred = model(x_data)
        data_loss = torch.mean((u_data_pred - u_data)**2)
        u_norm = model(x_interior)
        norm_loss = torch.mean((u_norm**2 - 1)**2)
        loss_pde = PINN_loss(model, x_interior, n, L)
        loss_drm = DRM_loss(model, x_interior)
        bc_loss = torch.mean((model(x_bc) - u_bc)**2)
        ortho_loss = Orthogonal_loss(model, x_interior, n, L)
        loss = (
            weight_pde * loss_pde +
            weight_drm * loss_drm +
            weight_data * data_loss +
            weight_norm * norm_loss +
            weight_bc * bc_loss +
            weight_orth * ortho_loss
        )
        loss.backward()
        optimizer.step()
        Losses.append(loss.item())
        with torch.no_grad():
            u_test = model(x_full_data)
            u_exact = Exact_solution(L, n, x_full_data)
            L2_error = torch.mean((u_test - u_exact) ** 2).item()
            L2_errors.append(L2_error)
            if L2_error < best_L2:
                best_L2 = L2_error
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
    if LBFGS:
        def closure():
            optimizer_LBFGS.zero_grad()
            u_data_pred = model(x_data)
            data_loss = torch.mean((u_data_pred - u_data)**2)
            u_norm = model(x_interior)
            norm_loss = torch.mean((u_norm**2 - 1)**2)
            bc_loss = torch.mean((model(x_bc) - u_bc)**2)
            ortho_loss = Orthogonal_loss(model, x_interior, n, L)
            loss_LBFGS = (
                weight_pde * PINN_loss(model, x_interior, n, L) +
                weight_drm * DRM_loss(model, x_interior) +
                weight_data * data_loss +
                weight_norm * norm_loss +
                weight_bc * bc_loss +
                weight_orth * ortho_loss
            )
            loss_LBFGS.backward()
            return loss_LBFGS
        optimizer_LBFGS.step(closure)
    end_time = time()
    print(f'Training for {method} completed in {end_time - start_time:.2f} seconds')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if best_state is not None:
        model_filename = f'{method}_best_model_n{n}_{technique}_{timestamp}.pth'
        torch.save(best_state, os.path.join(save_path, model_filename))
        print(f'Best model saved for {method} n={n} with L2 error {best_L2:.4e} at epoch {best_epoch} as {model_filename}')
    losses_filename = f'{method}_losses_n{n}_{technique}_{timestamp}.npy'
    l2_errors_filename = f'{method}_L2_errors_n{n}_{technique}_{timestamp}.npy'
    np.save(os.path.join(save_path, losses_filename), np.array(Losses))
    np.save(os.path.join(save_path, l2_errors_filename), np.array(L2_errors))
    L2_error = min(L2_errors)
    min_epoch = L2_errors.index(L2_error)
    print(f'Minimum L2 Error for n={n}: {L2_error:.4e} at epoch {min_epoch}')
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
    return model

def run_seperate_method(n_values, epochs=3000, LBFGS=False):
    for n in n_values:
        print(f"Training for n={n}...")
        model_pinn_bc = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='PINN', technique='BC')
        model_pinn_fbc = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='PINN', technique='FBC')
        model_pinn_fn = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='PINN', technique='FN')
        model_pinn_og = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='PINN', technique='OG')
        model_drm_bc = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='DRM', technique='BC')
        model_drm_fbc = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='DRM', technique='FBC')
        model_drm_fn = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='DRM', technique='FN')
        model_drm_og = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='DRM', technique='OG')
        print(f"Training completed for n={n}.")

if __name__ == "__main__":
    n_values = [1]
    run_seperate_method(n_values, epochs=50, LBFGS=False)