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

torch.manual_seed(0)
np.random.seed(0)

save_path = 'results/Quantum_Harmonic_Oscillator_2D_Compare_For_Paper'
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_file = os.path.join(save_path, 'results_QHO_2D_energy.json')
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

# --- 1) Exact 2D quantum harmonic oscillator solution ---
def phys_hermite(n, z):
    """
    Compute the physicists' Hermite polynomial H_n(z)
    by recurrence, entirely in PyTorch.
    """
    if n == 0:
        return torch.ones_like(z)
    elif n == 1:
        return 2 * z
    H_nm2 = torch.ones_like(z)    # H_0
    H_nm1 = 2 * z                  # H_1
    for k in range(2, n + 1):
        H_n = 2 * z * H_nm1 - 2 * (k - 1) * H_nm2
        H_nm2, H_nm1 = H_nm1, H_n
    return H_n

def Exact_solution_1d(n, x, omega=math.sqrt(2)):
    sqrt_omega = torch.sqrt(torch.tensor(omega))
    Hn = phys_hermite(n, sqrt_omega * x)
    norm = (omega / np.pi) ** 0.25 / math.sqrt(2 ** n * math.factorial(n))
    return norm * Hn * torch.exp(-omega * x**2 / 2)

def Exact_solution(L, nx, ny, x, y):
    return Exact_solution_1d(nx, x) * Exact_solution_1d(ny, y)

def Exact_energy(nx, ny, L):
    omega = math.sqrt(2)
    return (nx + ny + 1) * omega

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
        self.nodes_x = self.get_nodes(nx)
        self.nodes_y = self.get_nodes(ny)

    def get_nodes(self, n):
        omega = math.sqrt(2)
        scale = 2 ** (-1/4)
        if n == 0:
            return torch.tensor([], dtype=torch.float32)
        elif n == 1:
            return torch.tensor([0.0], dtype=torch.float32)
        elif n == 2:
            return torch.tensor([-2**(-3/4), 2**(-3/4)], dtype=torch.float32)
        elif n == 3:
            return torch.tensor([0.0, -2**(-3/4)*math.sqrt(3), 2**(-3/4)*math.sqrt(3)], dtype=torch.float32)
        elif n == 4:
            return torch.tensor([
                -scale*math.sqrt((3+math.sqrt(6))/2),
                -scale*math.sqrt((3-math.sqrt(6))/2),
                scale*math.sqrt((3-math.sqrt(6))/2),
                scale*math.sqrt((3+math.sqrt(6))/2)
            ], dtype=torch.float32)
        elif n == 5:
            return torch.tensor([
                0.0,
                -scale*math.sqrt((5+math.sqrt(10))/2),
                -scale*math.sqrt((5-math.sqrt(10))/2),
                scale*math.sqrt((5-math.sqrt(10))/2),
                scale*math.sqrt((5+math.sqrt(10))/2)
            ], dtype=torch.float32)
        else:
            raise ValueError(f"Nodes not defined for n={n}")

    def forward(self, x, y, L=6.0):
        XY = torch.stack((x, y), dim=-1)      # shape (…, 2)
        XY_flat = XY.view(-1, 2)              # shape (batch, 2)
        u_flat  = self.net(XY_flat)           # shape (batch, 1)
        if self.technique == 'FBC' or self.technique == 'OG':
            # enforce approximate BC at x,y=±L
            bc_factor_x = (1 - torch.exp(-(x + L))) * (1 - torch.exp(x - L))
            bc_factor_y = (1 - torch.exp(-(y + L))) * (1 - torch.exp(y - L))
            bc_factor = bc_factor_x * bc_factor_y
            return bc_factor * u_flat.view(*x.shape)
        elif self.technique == 'FN':
            # enforce approximate BC
            bc_factor_x = (1 - torch.exp(-(x + L))) * (1 - torch.exp(x - L))
            bc_factor_y = (1 - torch.exp(-(y + L))) * (1 - torch.exp(y - L))
            bc_factor = bc_factor_x * bc_factor_y
            # enforce nodal lines
            node_factor_x = torch.ones_like(x)
            for node in self.nodes_x:
                node_factor_x *= (x - node.to(x.device))
            node_factor_y = torch.ones_like(y)
            for node in self.nodes_y:
                node_factor_y *= (y - node.to(y.device))
            node_factor = node_factor_x * node_factor_y
            return bc_factor * node_factor * u_flat.view(*x.shape)  # back to (N,N) or (M,1)
        else:
            raise ValueError(f"Unknown technique: {self.technique}")

def function_w(x, y, L):
    low, up = -L, L
    I1 = 0.210987
    h = (up - low) / 2.0
    center = (up + low) / 2.0
    t_x = (x - center) / h
    t_y = (y - center) / h
    mask_x = torch.abs(t_x) < 1.0
    mask_y = torch.abs(t_y) < 1.0
    phi_x = torch.where(mask_x,
                        torch.exp(1.0 / (t_x.pow(2) - 1.0 + 1e-10)) / I1,
                        torch.zeros_like(t_x))
    phi_y = torch.where(mask_y,
                        torch.exp(1.0 / (t_y.pow(2) - 1.0 + 1e-10)) / I1,
                        torch.zeros_like(t_y))
    w = phi_x * phi_y
    dw_dx = torch.autograd.grad(
        outputs=w,
        inputs=x,
        grad_outputs=torch.ones_like(w),
        create_graph=True
    )[0]
    dw_dy = torch.autograd.grad(
        outputs=w,
        inputs=y,
        grad_outputs=torch.ones_like(w),
        create_graph=True
    )[0]
    dw_dx = torch.nan_to_num(dw_dx)
    dw_dy = torch.nan_to_num(dw_dy)
    return w, dw_dx, dw_dy

def WAN_loss(u_model, v_model, x, y, nx, ny, L, weight_pde=1.0, weight_norm=1.0):
    w, dw_dx, dw_dy = function_w(x, y, L)
    u_interior = u_model(x, y)
    v_interior = v_model(x, y)
    du_dx = torch.autograd.grad(u_interior, x, grad_outputs=torch.ones_like(u_interior), create_graph=True)[0]
    du_dy = torch.autograd.grad(u_interior, y, grad_outputs=torch.ones_like(u_interior), create_graph=True)[0]
    dv_dx = torch.autograd.grad(v_interior, x, grad_outputs=torch.ones_like(v_interior), create_graph=True)[0]
    dv_dy = torch.autograd.grad(v_interior, y, grad_outputs=torch.ones_like(v_interior), create_graph=True)[0]
    phi = w * v_interior
    dphi_dx = dw_dx * v_interior + w * dv_dx
    dphi_dy = dw_dy * v_interior + w * dv_dy
    prefactor = 0.5
    V = 0.5 * math.sqrt(2)**2 * (x**2 + y**2)
    E = Exact_energy(nx, ny, L)
    integral = prefactor * (du_dx * dphi_dx + du_dy * dphi_dy) + (V * u_interior - E * u_interior) * phi
    weak_residual = torch.mean(integral)
    norm = torch.mean(phi**2)
    loss_pde = weak_residual**2 / (norm + 1e-8)  # Add small epsilon for stability
    loss_norm = (4 * L * L * torch.mean(u_interior**2) - 1.0)**2  # Enforce ∫u² dx dy ≈ 1
    total_loss = weight_pde * loss_pde + weight_norm * loss_norm
    loss_v = -torch.log(loss_pde + 1e-8)  # Avoid log(0)
    return total_loss, loss_v, loss_pde, loss_norm

def orthogonal_loss(model, x, y, nx, ny, L):
    u_pred = model(x, y)
    E_current = nx + ny + 1
    ortho = 0.0
    max_n = max(nx, ny) + 1
    for i in range(max_n):
        for j in range(max_n):
            if i + j + 1 < E_current:
                u_exact = Exact_solution(L, i, j, x, y)
                inner = torch.mean(u_pred * u_exact) * 4 * L * L
                norm_sq = torch.mean(u_exact ** 2) * 4 * L * L
                ortho += (inner ** 2) / (norm_sq + 1e-8)
    return ortho

# Training function for the PINN
def train_pinn_seperate(nx, ny, L=6.0, epochs=10000, lr=0.001, LBFGS=False, method='PINN', technique='FBC', trainable_energy=False):
    # Define the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)

    # sample points
    x_interior = torch.linspace(-L, L, 200).to(device)
    y_interior = torch.linspace(-L, L, 200).to(device)
    x_interior, y_interior = torch.meshgrid(x_interior, y_interior)
    
    x_interior.requires_grad = True  # Enable gradients for interior points
    y_interior.requires_grad = True  # Enable gradients for interior points

    # sample data
    x_data_lin = torch.linspace(-L, L, 50).to(device)
    y_data_lin = torch.linspace(-L, L, 50).to(device)
    X_data, Y_data = torch.meshgrid(x_data_lin, y_data_lin)
    u_data = Exact_solution(L, nx, ny, X_data, Y_data)
    # only use first 25% of data points (lower left quadrant approximation)
    X_data = X_data[:25, :25].reshape(-1, 1)
    Y_data = Y_data[:25, :25].reshape(-1, 1)
    u_data = u_data[:25, :25].reshape(-1, 1)

    # boundary points for OG technique
    num_b = 200
    # horizontal for bottom and top
    x_horiz = torch.linspace(-L, L, num_b).to(device).view(-1,1)
    y_bottom = -L * torch.ones_like(x_horiz)
    y_top = L * torch.ones_like(x_horiz)
    # vertical for left and right
    y_vert = torch.linspace(-L, L, num_b).to(device).view(-1,1)
    x_left = -L * torch.ones_like(y_vert)
    x_right = L * torch.ones_like(y_vert)

    # model
    if method == 'WAN':
        technique_u = technique
        technique_v = 'FBC'
        u_model = FCN([2, 50, 50, 50, 50, 1], nx, ny, technique_u).to(device)
        v_model = FCN([2, 20, 20, 20, 1], nx, ny, technique_v).to(device)
    else:
        u_model = FCN([2, 50, 50, 50, 50, 1], nx, ny, technique).to(device)
        v_model = None
    # Optimizer
    if trainable_energy and method == 'PINN':
        E_train = nn.Parameter(torch.tensor(Exact_energy(nx, ny, L), device=device))
        optimizer = torch.optim.Adam([{'params': u_model.parameters()}, {'params': E_train}], lr=lr)
    else:
        optimizer = torch.optim.Adam(u_model.parameters(), lr=lr)
    if method == 'WAN':
        v_optimizer = torch.optim.Adam(v_model.parameters(), lr=lr)
    if LBFGS:
        optimizer_LBFGS = torch.optim.LBFGS(u_model.parameters(), lr=lr, max_iter=500, line_search_fn='strong_wolfe')
    
    # losses weights
    if method == 'WAN':
        lambda_data = 10000.0
        lambda_pde = 10.0
        lambda_drm = 0.0
        lambda_ortho = 10000.0 if technique == 'OG' else 0.0
        lambda_norm = 1000.0
        # lambda_bc = 0.0 if technique_u != 'OG' else 10000.0
    else:
        lambda_data = 10000.0
        lambda_pde = 100.0 if method == 'PINN' else 0.0
        lambda_drm = 0.0 if method == 'PINN' else 100.0
        lambda_ortho = 0.0 if method == 'PINN' else 10000.0
        lambda_norm = 0.0
        # lambda_bc = 10000.0 if technique == 'OG' else 0.0
    lambda_parity = 1000.0
    lambda_symmetry = 1000.0

    Losses = []
    L2_errors = []
    best_L2 = float('inf')
    best_state = None
    best_state_v = None
    best_epoch = -1    
    # Training loop
    start_time = time()
    if method == 'PINN':
        # 2D energy levels
        E = Exact_energy(nx, ny, L)
    for epoch in tqdm(range(epochs)):
        u_model.train()
        if method == 'WAN':
            v_model.train()
        optimizer.zero_grad()
        if method == 'WAN':
            v_optimizer.zero_grad()

        u_interior = u_model(x_interior, y_interior)
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

        if method == 'WAN':
            # Train v (multiple steps)
            for p in u_model.parameters():
                p.requires_grad = False
            for p in v_model.parameters():
                p.requires_grad = True
            for _ in range(5):
                total_loss, loss_v, _, _ = WAN_loss(u_model, v_model, x_interior, y_interior, nx, ny, L, lambda_pde, lambda_norm)
                v_optimizer.zero_grad()
                loss_v.backward()
                v_optimizer.step()
            # Train u
            for p in u_model.parameters():
                p.requires_grad = True
            for p in v_model.parameters():
                p.requires_grad = False
            total_loss, _, loss_pde, loss_norm = WAN_loss(u_model, v_model, x_interior, y_interior, nx, ny, L, lambda_pde, lambda_norm)
            drm_loss = torch.tensor(0.0, device=device)
            pde_loss = loss_pde
        else:
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
                V = 0.5 * math.sqrt(2)**2 * (x_interior**2 + y_interior**2)
                E_use = E_train if trainable_energy else E
                residual = -0.5 * (d2u_dx2 + d2u_dy2) + V * u_interior - E_use * u_interior
                pde_loss = torch.mean(residual**2)
                drm_loss = torch.tensor(0.0, device=device)
            else:  # DRM
                grad_norm_sq = du_dx**2 + du_dy**2
                V = 0.5 * math.sqrt(2)**2 * (x_interior**2 + y_interior**2)
                drm_loss = torch.mean(0.5 * grad_norm_sq + V * u_interior**2) / torch.mean(u_interior**2 + 1e-8)
                pde_loss = torch.tensor(0.0, device=device)
            total_loss = lambda_pde * pde_loss + lambda_drm * drm_loss

        # Data Loss
        u_data_pred = u_model(X_data, Y_data)
        data_loss = torch.mean((u_data_pred - u_data)**2) 

        # symmetry loss
        u_swapped  = u_model(y_interior, x_interior)
        symmetry_loss = torch.mean((u_interior - u_swapped)**2) if nx == ny else 0.0
        # parity loss
        sign_x = (-1)**nx
        sign_y = (-1)**ny
        u_px = u_model(-x_interior, y_interior)
        u_py = u_model(x_interior, -y_interior)
        parity_loss_x = torch.mean((u_interior - sign_x * u_px)**2)
        parity_loss_y = torch.mean((u_interior - sign_y * u_py)**2)
        # ortho loss
        ortho_loss = orthogonal_loss(u_model, x_interior, y_interior, nx, ny, L) if method != 'PINN' else 0.0

        # Total Loss
        total_loss = total_loss + lambda_data * data_loss + lambda_parity * (parity_loss_x + parity_loss_y) + lambda_symmetry * symmetry_loss + lambda_ortho * ortho_loss
        total_loss.backward()
        optimizer.step()
        Losses.append(total_loss.item())
        with torch.no_grad():
            u_interior = u_model(x_interior, y_interior)
            u_exact = Exact_solution(L, nx, ny, x_interior, y_interior)
            l2_pos = torch.mean((u_interior - u_exact) ** 2).item()
            l2_neg = torch.mean((u_interior + u_exact) ** 2).item()
            L2_error = min(l2_pos, l2_neg)
            L2_errors.append(L2_error)
            if L2_error < best_L2:
                best_L2 = L2_error
                best_state = {k: v.detach().cpu().clone() for k, v in u_model.state_dict().items()}
                if method == 'WAN':
                    best_state_v = {k: v.detach().cpu().clone() for k, v in v_model.state_dict().items()}
                best_epoch = epoch
    if LBFGS:
        def closure():
            optimizer_LBFGS.zero_grad()
            u_interior = u_model(x_interior, y_interior)
            du_dx = torch.autograd.grad(u_interior, x_interior, torch.ones_like(u_interior), create_graph=True)[0]
            du_dy = torch.autograd.grad(u_interior, y_interior, torch.ones_like(u_interior), create_graph=True)[0]
            if method == 'WAN':
                total_loss, _, _, _ = WAN_loss(u_model, v_model, x_interior, y_interior, nx, ny, L, lambda_pde, lambda_norm)
                drm_loss = torch.tensor(0.0, device=device)
                pde_loss = torch.tensor(0.0, device=device)
            elif method == 'PINN':
                d2u_dx2 = torch.autograd.grad(du_dx, x_interior, torch.ones_like(du_dx), create_graph=True)[0]
                d2u_dy2 = torch.autograd.grad(du_dy, y_interior, torch.ones_like(du_dy), create_graph=True)[0]
                V = 0.5 * math.sqrt(2)**2 * (x_interior**2 + y_interior**2)
                E_use = E_train if trainable_energy else E
                residual = -0.5 * (d2u_dx2 + d2u_dy2) + V * u_interior - E_use * u_interior
                pde_loss = torch.mean(residual**2)
                drm_loss = torch.tensor(0.0, device=device)
                total_loss = lambda_pde * pde_loss + lambda_drm * drm_loss
            else:
                grad_norm_sq = du_dx**2 + du_dy**2
                V = 0.5 * math.sqrt(2)**2 * (x_interior**2 + y_interior**2)
                drm_loss = torch.mean(0.5 * grad_norm_sq + V * u_interior**2) / torch.mean(u_interior**2 + 1e-8)
                pde_loss = torch.tensor(0.0, device=device)
                total_loss = lambda_pde * pde_loss + lambda_drm * drm_loss
            u_data_pred = u_model(X_data, Y_data)
            data_loss = torch.mean((u_data_pred - u_data)**2) 
            u_swapped  = u_model(y_interior, x_interior)
            symmetry_loss = torch.mean((u_interior - u_swapped)**2) if nx == ny else 0.0
            sign_x = (-1)**nx
            sign_y = (-1)**ny
            u_px = u_model(-x_interior, y_interior)
            u_py = u_model(x_interior, -y_interior)
            parity_loss_x = torch.mean((u_interior - sign_x * u_px)**2)
            parity_loss_y = torch.mean((u_interior - sign_y * u_py)**2)
            ortho_loss = orthogonal_loss(u_model, x_interior, y_interior, nx, ny, L) if method != 'PINN' else 0.0
            if (method != 'WAN' and technique == 'OG') or (method == 'WAN' and technique_u == 'OG'):
                u_bottom = u_model(x_horiz, y_bottom)
                u_top = u_model(x_horiz, y_top)
                u_left = u_model(x_left, y_vert)
                u_right = u_model(x_right, y_vert)
                bc_loss = torch.mean(u_bottom**2 + u_top**2 + u_left**2 + u_right**2)
            else:
                bc_loss = torch.tensor(0.0, device=device)
            total_loss = total_loss + lambda_data * data_loss + parity_loss_x +   parity_loss_y + symmetry_loss + lambda_ortho * ortho_loss + lambda_bc * bc_loss
            total_loss.backward()
            return total_loss

        optimizer_LBFGS.step(closure)

    end_time = time()
    technique_disp = technique
    print(f'Training for {method} with {technique_disp} completed in {end_time - start_time:.2f} seconds')


    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if best_state is not None:
        model_filename = f'{method}_{technique_disp}_best_model_n{nx}_{ny}_{timestamp}.pth'
        torch.save(best_state, os.path.join(save_path, model_filename))
        if method == 'WAN':
            v_model_filename = f'{method}_{technique_disp}_best_model_v_n{nx}_{ny}_{timestamp}.pth'
            torch.save(best_state_v, os.path.join(save_path, v_model_filename))
        print(f'Best model saved for {method} {technique_disp} n={nx, ny} with L2 error {best_L2:.4e} at epoch {best_epoch} as {model_filename}')
    # Save Losses and L2_errors as numpy arrays
    losses_filename = f'{method}_{technique_disp}_losses_n{nx}_{ny}_{timestamp}.npy'
    l2_errors_filename = f'{method}_{technique_disp}_L2_errors_n{nx}_{ny}_{timestamp}.npy'
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
            'technique': technique_disp,
            'weight_pde': lambda_pde,
            'weight_drm': lambda_drm,
            'weight_data': lambda_data,
            'weight_norm': lambda_norm,
            'weight_bc': 0,
            'weight_orth': lambda_ortho,
            'percentage': 0.25,
            'timestamp': timestamp,
            'learned_energy': E_train.item() if (trainable_energy and method == 'PINN') else Exact_energy(nx, ny, L)
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
    u_model.eval()
    with torch.no_grad():
        x_test = torch.linspace(-L, L, 100).to(device)
        y_test = torch.linspace(-L, L, 100).to(device)
        x_test, y_test = torch.meshgrid(x_test, y_test)
        u_test = u_model(x_test, y_test)
        u_exact = Exact_solution(L, nx, ny, x_test, y_test)
        l2_pos = torch.mean((u_test - u_exact) ** 2).item()
        l2_neg = torch.mean((u_test + u_exact) ** 2).item()
        if l2_neg < l2_pos:
            u_test = -u_test

        plot_pinn_vs_exact(x_test, y_test, u_test, u_exact, title=f'n={nx,ny}_{method}_{technique_disp}')

        L2_error = min(l2_pos, l2_neg)
        print(f'L2 Error for n={nx,ny} with {method} {technique_disp}: {L2_error:.4e}')
    return u_model, Losses, L2_error

def run_seperate_method(epochs=10000, LBFGS=False):
    n_values = [[0, 0], [1, 0],  [1, 1], [2, 0], [2, 1],  [2, 2]]
    L = 6.0
    methods = ['PINN', 'DRM', 'WAN']  # PINN baseline, DRM and WAN options
    techniques = ['FN', 'OG']  # FBC baseline, OG and FN options
    for nx, ny in n_values:
        for method in methods:
            if method == 'DRM':
                for technique in techniques:
                    print(f"Training {method} with {technique} for n={nx, ny}...")
                    model, losses, l2_error = train_pinn_seperate(nx, ny, L, epochs=epochs, LBFGS=LBFGS, method=method, technique=technique, trainable_energy=True)
                    print(f"Training completed for n={nx, ny} with {method} {technique}. L2 Error: {l2_error:.4e}")
            elif method == 'WAN':
                for technique in techniques:
                    print(f"Training {method} with {technique} for n={nx, ny}...")
                    model, losses, l2_error = train_pinn_seperate(nx, ny, L, epochs=epochs, LBFGS=LBFGS, method=method, technique=technique, trainable_energy=True)
                    print(f"Training completed for n={nx, ny} with {method} {technique}. L2 Error: {l2_error:.4e}")
            else:  # PINN
                for technique in ['FBC', 'FN']:
                    print(f"Training {method} with {technique} for n={nx, ny}...")
                    model, losses, l2_error = train_pinn_seperate(nx, ny, L, epochs=epochs, LBFGS=LBFGS, method=method, technique=technique, trainable_energy=True)
                    print(f"Training completed for n={nx, ny} with {method} {technique}. L2 Error: {l2_error:.4e}")

if __name__ == "__main__":
    run_seperate_method(50000, LBFGS=False)