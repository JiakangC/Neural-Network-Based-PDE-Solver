import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import math
import json
from time import time
import datetime
import os

# torch.manual_seed(0)
# np.random.seed(0)

save_path = 'results/Infinite_Potential_Well_2D'
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_file = os.path.join(save_path, 'results_IPW_2D.json')
if not os.path.exists(results_file):
    with open(results_file, 'w') as f:
        json.dump([], f)  # Initialize with an empty list
        print(f"Created results file at {results_file}")

def plot_pinn_vs_exact(x, y, u_pinn, u_exact, title='PINN vs Exact Solution'):
    """
    x, y:    torch tensors of shape (N,N)
    u_pinn:  torch tensor (N,N) PINN solution
    u_exact: torch tensor (N,N) exact solution
    """
    X = x.cpu().numpy()
    Y = y.cpu().numpy()
    Zp = u_pinn.cpu().numpy()
    Ze = u_exact.cpu().numpy()

    fig = plt.figure(figsize=(12,10))

    # 2D heatmaps
    ax1 = fig.add_subplot(2,2,1)
    hm1 = ax1.pcolormesh(X, Y, Zp, shading='auto')
    fig.colorbar(hm1, ax=ax1, label='u_pinn')
    ax1.set_title('2D PINN heatmap')
    ax1.set_xlabel('x'); ax1.set_ylabel('y')

    ax2 = fig.add_subplot(2,2,2)
    hm2 = ax2.pcolormesh(X, Y, Ze, shading='auto')
    fig.colorbar(hm2, ax=ax2, label='u_exact')
    ax2.set_title('2D Exact heatmap')
    ax2.set_xlabel('x'); ax2.set_ylabel('y')

    # 3D surfaces
    ax3 = fig.add_subplot(2,2,3, projection='3d')
    surf1 = ax3.plot_surface(X, Y, Zp, rcount=100, ccount=100, cmap='viridis', edgecolor='none')
    fig.colorbar(surf1, ax=ax3, shrink=0.6, label='u_pinn')
    ax3.set_title('3D PINN surface')
    ax3.set_xlabel('x'); ax3.set_ylabel('y'); ax3.set_zlabel('u')

    ax4 = fig.add_subplot(2,2,4, projection='3d')
    surf2 = ax4.plot_surface(X, Y, Ze, rcount=100, ccount=100, cmap='plasma', edgecolor='none')
    fig.colorbar(surf2, ax=ax4, shrink=0.6, label='u_exact')
    ax4.set_title('3D Exact surface')
    ax4.set_xlabel('x'); ax4.set_ylabel('y'); ax4.set_zlabel('u')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path, f'pinn_vs_exact_{title}.png'))
    #plt.show()

# --- 1) Exact 2D infinite-well solution ---
def Exact_solution(L, nx, ny, x, y):
    coef = 2.0 / L
    return coef * torch.sin(nx * torch.pi * x / L) * torch.sin(ny * torch.pi * y / L)

# --- 2) Fully-connected network for 2D ---
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FCN(nn.Module):
    def __init__(self, layers, nx, ny, technique):
        super().__init__()
        self.nx, self.ny = nx, ny
        self.technique = technique
        mods = []
        for i in range(len(layers)-2):
            mods += [nn.Linear(layers[i],layers[i+1]), Sin()]
        mods.append(nn.Linear(layers[-2],layers[-1]))
        self.net = nn.Sequential(*mods)

    def forward(self, x, y, L=2.0):
        XY = torch.stack((x, y), dim=-1)      # shape (…, 2)
        XY_flat = XY.view(-1, 2)              # shape (batch, 2)
        u_flat  = self.net(XY_flat)           # shape (batch, 1)
        if self.technique == 'FBC' or self.technique == 'OG':
            # enforce BC x=0,L and y=0,L
            bc_factor = x*(L-x)*y*(L-y)
            return bc_factor * u_flat.view(*x.shape)
        elif self.technique == 'FN':
            # enforce BC x=0,L and y=0,L
            bc_factor = x*(L-x)*y*(L-y)
            # enforce nodal lines
            node_factor_x = torch.ones_like(x)
            for k in range(1, self.nx):
                node_factor_x *= (x - k * L / self.nx)
            node_factor_y = torch.ones_like(y)
            for k in range(1, self.ny):
                node_factor_y *= (y - k * L / self.ny)
            node_factor = node_factor_x * node_factor_y
            return bc_factor * node_factor * u_flat.view(*x.shape)  # back to (N,N) or (M,1)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")

def orthogonal_loss(model, x, y, nx, ny, L):
    u_pred = model(x, y)
    E_current = nx**2 + ny**2
    ortho = 0.0
    max_n = max(nx, ny)
    for i in range(1, max_n + 1):
        for j in range(1, max_n + 1):
            if i**2 + j**2 < E_current:
                u_exact = Exact_solution(L, i, j, x, y)
                inner = torch.mean(u_pred * u_exact) * L * L
                norm_sq = torch.mean(u_exact ** 2) * L * L
                ortho += (inner ** 2) / (norm_sq + 1e-8)
    return ortho

# Training function for the PINN
def train_pinn_seperate(nx, ny, L=2.0, epochs=10000, lr=0.001, LBFGS=False, method='PINN', technique='FBC'):
    # Define the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)
    # Constants
    hbar = 1.0
    m = 1.0

    # sample points
    x_interior = torch.linspace(0, L, 200).to(device)
    y_interior = torch.linspace(0, L, 200).to(device)
    x_interior, y_interior = torch.meshgrid(x_interior, y_interior)
    
    x_interior.requires_grad = True  # Enable gradients for interior points
    y_interior.requires_grad = True  # Enable gradients for interior points

    # sample data
    x_data_lin = torch.linspace(0, L, 50).to(device)
    y_data_lin = torch.linspace(0, L, 50).to(device)
    X_data, Y_data = torch.meshgrid(x_data_lin, y_data_lin)
    u_data = Exact_solution(L, nx, ny, X_data, Y_data)
    # only use first 25% of data points
    X_data = X_data[:25, :25].reshape(-1, 1)
    Y_data = Y_data[:25, :25].reshape(-1, 1)
    u_data = u_data[:25, :25].reshape(-1, 1)

    # boundary points for OG technique
    num_b = 200
    # bottom and top
    x_bt = torch.linspace(0, L, num_b).to(device).view(-1,1)
    y_bottom = torch.zeros_like(x_bt)
    y_top = L * torch.ones_like(x_bt)
    # left and right
    y_lr = torch.linspace(0, L, num_b).to(device).view(-1,1)
    x_left = torch.zeros_like(y_lr)
    x_right = L * torch.ones_like(y_lr)

    # model
    model =  FCN([2, 50, 50, 50, 50, 1], nx, ny, technique).to(device)
    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if LBFGS:
        optimizer_LBFGS = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=500, line_search_fn='strong_wolfe')
    
    # losses weights
    lambda_data = 0
    lambda_pde = 10.0 if method == 'PINN' else 0.0
    lambda_drm = 0.0 if method == 'PINN' else 100.0
    lambda_ortho = 0.0 if method == 'PINN' else 10000.0
    lambda_bc = 10000.0 if technique == 'OG' else 0.0
    

    Losses = []
    L2_errors = []
    best_L2 = float('inf')
    best_state = None
    best_epoch = -1    
    # Training loop
    start_time = time()
    if method == 'PINN':
        # 2D energy levels
        E = (nx * np.pi * hbar) ** 2 / (2 * m * L**2) + (ny * np.pi * hbar) ** 2 / (2 * m * L**2)
        k_squared = (2 * m * E) / (hbar ** 2)
    for epoch in tqdm(range(epochs)):
        model.train()
        optimizer.zero_grad()

        u_interior = model(x_interior, y_interior)
        du_dx = torch.autograd.grad(
            outputs=u_interior,
            inputs=x_interior,
            grad_outputs=torch.ones_like(u_interior),
            create_graph=True
        )[0]
        du_dy = torch.autograd.grad(
            outputs=u_interior,
            inputs=y_interior,
            grad_outputs=torch.ones_like(u_interior),
            create_graph=True
        )[0]

        if method == 'PINN':
            d2u_dx2 = torch.autograd.grad(
                outputs=du_dx,
                inputs=x_interior,
                grad_outputs=torch.ones_like(du_dx),
                create_graph=True
            )[0]
            d2u_dy2 = torch.autograd.grad(
                outputs=du_dy,
                inputs=y_interior,
                grad_outputs=torch.ones_like(du_dy),
                create_graph=True
            )[0]
            residual = d2u_dx2 + d2u_dy2 + k_squared * u_interior
            pde_loss = torch.mean(residual**2)
            drm_loss = torch.tensor(0.0, device=device)
        else:  # DRM
            grad_norm_sq = du_dx**2 + du_dy**2
            drm_loss = torch.mean(grad_norm_sq) / torch.mean(u_interior**2 + 1e-8)
            pde_loss = torch.tensor(0.0, device=device)

        # Data Loss
        u_data_pred = model(X_data, Y_data)
        data_loss = torch.mean((u_data_pred - u_data)**2) 

        # symmetry loss
        u_swapped  = model(y_interior, x_interior)
        symmetry_loss = torch.mean((u_interior - u_swapped)**2) if nx == ny else 0.0
        # parity loss
        is_even_x = (nx % 2 == 1)
        is_even_y = (ny % 2 == 1)
        u_px = model(L - x_interior, y_interior)
        u_py = model(x_interior, L - y_interior)
        sign_x = 1.0 if is_even_x else -1.0
        sign_y = 1.0 if is_even_y else -1.0
        parity_loss_x = torch.mean((u_interior - sign_x * u_px)**2)
        parity_loss_y = torch.mean((u_interior - sign_y * u_py)**2)
        # ortho loss
        ortho_loss = orthogonal_loss(model, x_interior, y_interior, nx, ny, L) if method == 'DRM' else 0.0
        # BC loss for OG
        if technique == 'OG':
            u_bottom = model(x_bt, y_bottom)
            u_top = model(x_bt, y_top)
            u_left = model(x_left, y_lr)
            u_right = model(x_right, y_lr)
            bc_loss = torch.mean(u_bottom**2 + u_top**2 + u_left**2 + u_right**2)
        else:
            bc_loss = torch.tensor(0.0, device=device)
        # Total Loss
        total_loss = lambda_pde * pde_loss + lambda_drm * drm_loss + lambda_data * data_loss + parity_loss_x +   parity_loss_y + symmetry_loss + lambda_ortho * ortho_loss + lambda_bc * bc_loss
        total_loss.backward()
        optimizer.step()
        Losses.append(total_loss.item())
        with torch.no_grad():
            u_interior = model(x_interior, y_interior)
            u_exact = Exact_solution(L, nx, ny, x_interior, y_interior)
            L2_error = torch.mean((u_interior - u_exact) ** 2).item()
            L2_errors.append(L2_error)
            if L2_error < best_L2:
                best_L2 = L2_error
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best_epoch = epoch
    if LBFGS:
        def closure():
            optimizer_LBFGS.zero_grad()
            u_interior = model(x_interior, y_interior)
            du_dx = torch.autograd.grad(u_interior, x_interior, torch.ones_like(u_interior), create_graph=True)[0]
            du_dy = torch.autograd.grad(u_interior, y_interior, torch.ones_like(u_interior), create_graph=True)[0]
            if method == 'PINN':
                d2u_dx2 = torch.autograd.grad(du_dx, x_interior, torch.ones_like(du_dx), create_graph=True)[0]
                d2u_dy2 = torch.autograd.grad(du_dy, y_interior, torch.ones_like(du_dy), create_graph=True)[0]
                residual = d2u_dx2 + d2u_dy2 + k_squared * u_interior
                pde_loss = torch.mean(residual**2)
                drm_loss = torch.tensor(0.0, device=device)
            else:
                grad_norm_sq = du_dx**2 + du_dy**2
                drm_loss = torch.mean(grad_norm_sq) / torch.mean(u_interior**2 + 1e-8)
                pde_loss = torch.tensor(0.0, device=device)
            u_data_pred = model(X_data, Y_data)
            data_loss = torch.mean((u_data_pred - u_data)**2) 
            u_swapped  = model(y_interior, x_interior)
            symmetry_loss = torch.mean((u_interior - u_swapped)**2) if nx == ny else 0.0
            is_even_x = (nx % 2 == 1)
            is_even_y = (ny % 2 == 1)
            u_px = model(L - x_interior, y_interior)
            u_py = model(x_interior, L - y_interior)
            sign_x = 1.0 if is_even_x else -1.0
            sign_y = 1.0 if is_even_y else -1.0
            parity_loss_x = torch.mean((u_interior - sign_x * u_px)**2)
            parity_loss_y = torch.mean((u_interior - sign_y * u_py)**2)
            ortho_loss = orthogonal_loss(model, x_interior, y_interior, nx, ny, L) if method == 'DRM' else 0.0
            if technique == 'OG':
                u_bottom = model(x_bt, y_bottom)
                u_top = model(x_bt, y_top)
                u_left = model(x_left, y_lr)
                u_right = model(x_right, y_lr)
                bc_loss = torch.mean(u_bottom**2 + u_top**2 + u_left**2 + u_right**2)
            else:
                bc_loss = torch.tensor(0.0, device=device)
            total_loss = lambda_pde * pde_loss + lambda_drm * drm_loss + lambda_data * data_loss + parity_loss_x +   parity_loss_y + symmetry_loss + lambda_ortho * ortho_loss + lambda_bc * bc_loss
            total_loss.backward()
            return total_loss

        optimizer_LBFGS.step(closure)

    end_time = time()
    print(f'Training for {method} with {technique} completed in {end_time - start_time:.2f} seconds')


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if best_state is not None:
        model_filename = f'{method}_{technique}_best_model_n{nx}_{ny}_{timestamp}.pth'
        torch.save(best_state, os.path.join(save_path, model_filename))
        print(f'Best model saved for {method} {technique} n={nx, ny} with L2 error {best_L2:.4e} at epoch {best_epoch} as {model_filename}')
    # Save Losses and L2_errors as numpy arrays
    losses_filename = f'{method}_{technique}_losses_n{nx}_{ny}_{timestamp}.npy'
    l2_errors_filename = f'{method}_{technique}_L2_errors_n{nx}_{ny}_{timestamp}.npy'
    np.save(os.path.join(save_path, losses_filename), np.array(Losses))
    np.save(os.path.join(save_path, l2_errors_filename), np.array(L2_errors))
    L2_error = min(L2_errors)
    # find the order of the minimum L2 error
    min_epoch = L2_errors.index(L2_error)
    print(f'Minimum L2 Error for n={nx, ny}: {L2_error:.4e} at epoch {min_epoch}')
    results = {
        'method': method,
        'n': [nx, ny],
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
        'weight_pde': lambda_pde,
        'weight_drm': lambda_drm,
        'weight_data': lambda_data,
        'weight_norm': 0,
        'weight_bc': lambda_bc,
        'weight_orth': lambda_ortho,
        'percentage': 0.25,
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
    model.eval()
    with torch.no_grad():
        x_test = torch.linspace(0, L, 100).to(device)
        y_test = torch.linspace(0, L, 100).to(device)
        x_test, y_test = torch.meshgrid(x_test, y_test)
        u_test = model(x_test, y_test)
        u_exact = Exact_solution(L, nx, ny, x_test, y_test)

        plot_pinn_vs_exact(x_test, y_test, u_test, u_exact, title=f'n={nx,ny}_{method}_{technique}')

        L2_error = torch.mean((u_test - u_exact) ** 2).item()
        print(f'L2 Error for n={nx,ny} with {method} {technique}: {L2_error:.4e}')
    return model, Losses, L2_error

def run_seperate_method(epochs=10000, LBFGS=False):
    n_values = [[1, 1]]
    L = 2.0
    methods = ['PINN']
    techniques = ['FBC']  # FBC baseline, OG and FN options
    for nx, ny in n_values:
        for method in methods:
            for technique in techniques:
                print(f"Training {method} with {technique} for n={nx, ny}...")
                model, losses, l2_error = train_pinn_seperate(nx, ny, L, epochs=epochs, LBFGS=LBFGS, method=method, technique=technique)
                print(f"Training completed for n={nx, ny} with {method} {technique}. L2 Error: {l2_error:.4e}")

if __name__ == "__main__":
    run_seperate_method(100, LBFGS=False)