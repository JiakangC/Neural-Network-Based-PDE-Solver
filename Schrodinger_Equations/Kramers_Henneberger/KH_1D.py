import os
import json
import math
import datetime
from time import time

import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# ============== seeds & device ==============
torch.manual_seed(0)
np.random.seed(0)
device_default = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"[info] Using device: {device_default}")

# ============== potentials ==============
# Short-range bare potential:
# V(x) = V0 * exp(-sqrt(x^2 + 16)) / sqrt(x^2 + 6.27^2)
def V_base(x, V0=-24.856):
    return V0 * torch.exp(-(x**2 + 16.0).sqrt()) / (x**2 + (6.27**2)).sqrt()

# Shifted potential: V(x + alpha)
def V_KH_shift(x, alpha=0.0, V0=-24.856):
    return V_base(x + alpha, V0=V0)

# KH cycle-averaged potential for alpha(t) = alpha0 * sin(ωt)
# \bar V(x) = (1/2π) ∫_0^{2π} V(x + alpha0 sin θ) dθ
def V_KH_avg(x, alpha0=0.0, V0=-24.856, n_theta=500):
    if alpha0 == 0.0:
        return V_base(x, V0=V0)
    theta = torch.linspace(0, 2*torch.pi, n_theta, device=x.device)
    sin_th = torch.sin(theta)
    x_shift = x[..., None] + alpha0 * sin_th[None, ...]
    V_mat = V_base(x_shift, V0=V0)
    return V_mat.mean(dim=-1)

# Backward-compatible alias; default to averaged KH potential
def V_KH(x, alpha=0.0, V0=-24.856, use_avg=True, n_theta=500):
    return V_KH_avg(x, alpha0=alpha, V0=V0, n_theta=n_theta) if use_avg else V_KH_shift(x, alpha=alpha, V0=V0)

# ============== reference eigen-solver (finite difference) ==============
def reference_eigensystem(L=10.0, N=2000, alpha=0.0, V0=-24.856, k_max=10, use_avg=True, n_theta=500):
    """
    Build H = -1/2 d2/dx2 + V(x) on [-L, L] with Dirichlet BC, return first k_max eigenpairs.
    """
    x = torch.linspace(-L, L, N, dtype=torch.float64)
    dx = (2*L) / (N - 1)

    # -1/2 d2/dx2 with [1, -2, 1]/dx^2 stencil on interior points (N-2)
    diag = torch.full((N-2,), (-1/2) * (-2.0) / dx**2, dtype=torch.float64)
    offd = torch.full((N-3,), (-1/2) * (1.0) / dx**2, dtype=torch.float64)

    # Potential on interior points
    x_int_f32 = x[1:-1].to(torch.float32)
    V_int = (V_KH_avg(x_int_f32, alpha0=alpha, V0=V0, n_theta=n_theta)
             if use_avg else V_KH_shift(x_int_f32, alpha=alpha, V0=V0)).to(torch.float64)

    # Dense tri-diagonal Hamiltonian
    H = torch.zeros((N-2, N-2), dtype=torch.float64)
    H[range(N-2), range(N-2)] = diag + V_int
    H[range(N-3), range(1, N-2)] = offd
    H[range(1, N-2), range(N-3)] = offd

    evals, evecs = torch.linalg.eigh(H)
    idx = torch.argsort(evals)
    evals = evals[idx][:k_max]
    evecs = evecs[:, idx][:, :k_max]

    # Put zeros at boundaries and L2-normalize
    psi = torch.zeros((N, k_max), dtype=torch.float64)
    psi[1:-1, :] = evecs
    w = torch.ones(N, dtype=torch.float64)
    w[0] = w[-1] = 0.5
    for k in range(k_max):
        norm = torch.sqrt(dx * torch.sum(w * psi[:, k]**2))
        psi[:, k] /= norm

    return x.to(torch.float32), evals.to(torch.float32), psi.to(torch.float32)

# ============== interpolation helpers (torch I/O, numpy backend) ==============
def interp1d(x_src: torch.Tensor, y_src: torch.Tensor, x_tgt: torch.Tensor) -> torch.Tensor:
    x_src_np = x_src.detach().cpu().numpy()
    y_src_np = y_src.detach().cpu().numpy()
    x_tgt_np = x_tgt.detach().cpu().numpy()
    y_tgt_np = np.interp(x_tgt_np, x_src_np, y_src_np)
    return torch.tensor(y_tgt_np, device=x_tgt.device, dtype=y_src.dtype)

def resample_ref_vec(x_ref, psi_ref_k, x_train):
    return interp1d(x_ref, psi_ref_k, x_train)

def resample_ref_matrix(x_ref, psi_ref_mat, x_train, n_cols):
    cols = [resample_ref_vec(x_ref, psi_ref_mat[:, k], x_train) for k in range(n_cols)]
    return torch.stack(cols, dim=1)

# ============== network & utilities ==============
class Sin(nn.Module):
    def forward(self, x):
        return torch.sin(x)

class FCN1D(nn.Module):
    def __init__(self, layers, technique='RAW'):
        super().__init__()
        self.technique = technique  # 'RAW' or 'FBC'
        mods = []
        for i in range(len(layers) - 2):
            mods += [nn.Linear(layers[i], layers[i+1]), Sin()]
        mods.append(nn.Linear(layers[-2], layers[-1]))
        self.net = nn.Sequential(*mods)

    def forward(self, x, L=10.0):
        x_in = x.view(-1, 1)
        u = self.net(x_in).view_as(x)
        if self.technique == 'FBC':
            # smooth window ~0 at ±L
            bc = (1 - torch.exp(-(x + L))) * (1 - torch.exp(x - L))
            return bc * u
        elif self.technique == 'RAW':
            return u
        else:
            raise ValueError(f"Unknown technique {self.technique}")

# integration helpers (grid average × domain length)
def integral_mean(f, L):
    return (2 * L) * torch.mean(f)

def inner_product(u, v, L):
    return integral_mean(u * v, L)

def normalize(u, L):
    n = torch.sqrt(integral_mean(u*u, L) + 1e-12)
    return u / n

# ============== WAN weight fn ==============
def weight_fn_w(x, L):
    low, up = -L, L
    I1 = 0.210987
    h = (up - low) / 2.0
    center = (up + low) / 2.0
    t = (x - center) / h
    mask = torch.abs(t) < 1.0
    phi = torch.where(mask, torch.exp(1.0 / (t**2 - 1.0 + 1e-10)) / I1, torch.zeros_like(t))
    dphi_dx = torch.autograd.grad(phi, x, grad_outputs=torch.ones_like(phi), create_graph=True)[0]
    dphi_dx = torch.nan_to_num(dphi_dx)
    return phi, dphi_dx

# ============== Ground Truth loader (GT only) ==============
class KH1DGroundTruth:
    """
    Holds GT on a dense grid:
      - x: (N,), V(x): (N,), E: (n_levels,), psi: (N, n_levels)
    """
    def __init__(self, *,
                 alpha=0.0, V0=-24.856, L=10.0, N=4000,
                 n_levels=5, use_avg=True, n_theta=500,
                 device=None):
        self.device = torch.device('cuda' if (device is None and torch.cuda.is_available()) else (device or 'cpu'))

        x_ref, E_ref, psi_ref = reference_eigensystem(
            L=L, N=N, alpha=alpha, V0=V0, k_max=max(n_levels, 1),
            use_avg=use_avg, n_theta=n_theta
        )

        if use_avg:
            Vx = V_KH_avg(x_ref, alpha0=alpha, V0=V0, n_theta=n_theta)
        else:
            Vx = V_KH_shift(x_ref, alpha=alpha, V0=V0)

        self.x   = x_ref.to(self.device)
        self.V   = Vx.to(self.device)
        self.E   = E_ref[:n_levels].to(self.device)
        self.psi = psi_ref[:, :n_levels].to(self.device)

        self.alpha, self.V0, self.L = float(alpha), float(V0), float(L)
        self.N, self.n_levels = int(N), int(n_levels)
        self.use_avg, self.n_theta = bool(use_avg), int(n_theta)

    @property
    def grid(self):
        return self.x

    def energy(self, n:int):
        return float(self.E[n].item())

    def wavefunction(self, n:int):
        return self.psi[:, n]

    def level(self, n:int):
        return {'x': self.x, 'V': self.V, 'E': self.energy(n), 'psi': self.wavefunction(n)}

    def resample(self, x_new: torch.Tensor):
        x_new = x_new.detach().to(self.device)
        if self.use_avg:
            V_new = V_KH_avg(x_new, alpha0=self.alpha, V0=self.V0, n_theta=self.n_theta)
        else:
            V_new = V_KH_shift(x_new, alpha=self.alpha, V0=self.V0)
        psi_new = resample_ref_matrix(self.x, self.psi, x_new, self.n_levels)
        return x_new, V_new, psi_new

class _KH1DGTDataset(Dataset):
    def __init__(self, gt: KH1DGroundTruth):
        self.gt = gt

    def __len__(self):
        return self.gt.N

    def __getitem__(self, idx):
        return (self.gt.x[idx], self.gt.V[idx], self.gt.psi[idx, :])

# ============== Unified model (trainable energy) ==============
class UnifiedEigenModel(nn.Module):
    def __init__(self, layers=[1,64,64,64,1], technique='RAW', E_init=0.0, device=None):
        super().__init__()
        self.u_model = FCN1D(layers, technique=technique)
        self.energy = nn.Parameter(torch.tensor(float(E_init)))
        if device is not None:
            self.to(device)

    def forward(self, x, L=10.0):
        return self.u_model(x, L=L)

# ============== Losses ==============
def pinn_loss(model: UnifiedEigenModel, x, alpha, V0, use_avg=True, n_theta=500):
    L_here = x.abs().max().item()
    u = model(x, L=L_here)
    du  = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    d2u = torch.autograd.grad(du, x, grad_outputs=torch.ones_like(du), create_graph=True)[0]
    Vx = V_KH(x, alpha=alpha, V0=V0, use_avg=use_avg, n_theta=n_theta)
    E  = model.energy
    r  = -0.5 * d2u + Vx * u - E * u
    return torch.mean(r**2)

def drm_loss(model: UnifiedEigenModel, x, alpha, V0, L, use_avg=True, n_theta=500):
    u  = model(x, L=L)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    Vx = V_KH(x, alpha=alpha, V0=V0, use_avg=use_avg, n_theta=n_theta)
    num = integral_mean(0.5 * du**2 + Vx * u**2, L)
    den = integral_mean(u**2, L) + 1e-12
    return num / den

def wan_loss(model: UnifiedEigenModel, v_model: nn.Module, x, alpha, V0, L, use_avg=True, n_theta=500):
    """
    Weak form residual with φ = w v:
      ∫ [ 1/2 u' φ' + V u φ ] dx  - E * ∫ u φ dx = 0
    Returns: pde_loss, norm_u
    """
    u  = model(x, L=L)
    v  = v_model(x, L=L)
    du = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    dv = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    w, dw = weight_fn_w(x, L)
    phi   = w * v
    dphi  = dw * v + w * dv

    Vx = V_KH(x, alpha=alpha, V0=V0, use_avg=use_avg, n_theta=n_theta)
    E  = model.energy

    I_kin_pot = integral_mean(0.5 * du * dphi + Vx * u * phi, L)
    I_u_phi   = integral_mean(u * phi, L)
    I_full    = I_kin_pot - E * I_u_phi

    norm_phi  = integral_mean(phi**2, L) + 1e-12
    pde_loss  = (I_full / norm_phi)**2
    norm_u    = (integral_mean(u**2, L) - 1.0)**2
    return pde_loss, norm_u

def orth_loss_from_gt(u, L, lower_refs_train):
    if lower_refs_train is None or lower_refs_train.numel() == 0:
        return torch.tensor(0.0, device=u.device)
    orth = 0.0
    for k in range(lower_refs_train.shape[1]):
        psi_k = lower_refs_train[:, k]
        ip    = inner_product(u, psi_k, L)
        denom = inner_product(psi_k, psi_k, L) + 1e-12
        orth += (ip**2) / denom
    return orth

# ============== helpers ==============
def first_fraction_indices(m, fraction=0.25, max_points=None, device='cpu'):
    k = max(1, int(m * fraction))
    if max_points is not None:
        k = min(k, int(max_points))
    return torch.arange(k, device=device, dtype=torch.long)

# ============== training (PINN / DRM / WAN) ==============
def train_state_v2(method, n, gt: KH1DGroundTruth, *,
                   x_train=None, L=None,
                   layers=[1,64,64,64,1], technique='RAW',
                   epochs=10000, lr=1e-3,
                   # weights
                   lambda_pde=1.0, lambda_data=1.0, lambda_orth=1e4, lambda_norm=1e3, lambda_bc=1e4, lambda_party=0.0,
                   # data usage
                   data_fraction=0.25, max_data_points=None,
                   # WAN
                   v_layers=[1,50,50,50,1], v_steps=3,
                   # physics flags
                   use_avg=True, n_theta=500,
                   # device / params
                   device=None, alpha=None, V0=None):
    """
    Energy parameter initialized from GT; PINN and WAN use model.energy in their PDE/weak losses.
    Data loss: first 25% of GT points on train grid (capped by max_data_points).
    Orthogonality: strictly against GT lower states (resampled).
    """
    device = torch.device('cuda' if (device is None and torch.cuda.is_available()) else (device or 'cpu'))
    alpha = gt.alpha if alpha is None else alpha
    V0    = gt.V0 if V0 is None else V0
    L     = gt.L if L is None else float(L)

    # training grid
    if x_train is None:
        x = torch.linspace(-L, L, 1024, device=device, dtype=torch.float32)
    else:
        x = x_train.detach().to(device).float()
    x.requires_grad_(True)

    # GT on train grid
    with torch.no_grad():
        _, V_train, psi_train = gt.resample(x)
        psi_n_train = psi_train[:, n]
        lower_refs_train = psi_train[:, :n] if n > 0 else None

    idx_data = first_fraction_indices(x.shape[0], fraction=data_fraction, max_points=max_data_points, device=device)

    # model & optim
    E_init = gt.energy(n)
    model  = UnifiedEigenModel(layers=layers, technique=technique if method != 'WAN' else 'RAW', E_init=E_init, device=device)

    if method == 'WAN':
        v_model = FCN1D(v_layers, technique='RAW').to(device)
        opt_u = torch.optim.Adam(list(model.parameters()), lr=lr)
        opt_v = torch.optim.Adam(v_model.parameters(), lr=lr * 2.0)
    else:
        opt = torch.optim.Adam(list(model.parameters()), lr=lr)

    best = {'metric': float('inf'), 'state': None, 'E': None, 'epoch': -1}
    Losses, L2s, E_track = [], [], []

    for ep in tqdm(range(epochs), desc=f"{method} n={n}"):
        if method == 'WAN':
            # update v (ascent)
            for _ in range(v_steps):
                opt_v.zero_grad(set_to_none=True)
                pde_loss_v, _ = wan_loss(model, v_model, x, alpha, V0, L, use_avg=use_avg, n_theta=n_theta)
                (-pde_loss_v).backward()
                opt_v.step()

            # update u & E (descent)
            opt_u.zero_grad(set_to_none=True)
            u = model(x, L=L)
            pde_loss_u, norm_u = wan_loss(model, v_model, x, alpha, V0, L, use_avg=use_avg, n_theta=n_theta)

            if lambda_data != 0:
                u_data   = u.index_select(0, idx_data)
                psi_data = psi_n_train.index_select(0, idx_data)
                data_loss = torch.mean((u_data - psi_data)**2)
            else:
                data_loss = torch.tensor(0.0, device=device)

            orth = orth_loss_from_gt(u, L, lower_refs_train)

            if lambda_party != 0.0:
                u_swap = model(-x, L=L)
                party_loss = torch.mean((u - u_swap)**2) if (n % 2 == 0) else torch.mean((u + u_swap)**2)
            else:
                party_loss = torch.tensor(0.0, device=device)

            bc = (u[0]**2 + u[-1]**2)

            loss = (lambda_pde * pde_loss_u
                    + lambda_norm * norm_u
                    + lambda_data * data_loss
                    + lambda_orth * orth
                    + lambda_bc * bc
                    + lambda_party * party_loss)
            loss.backward()
            opt_u.step()

            E_track.append(float(model.energy.detach().cpu().item()))

        else:
            opt.zero_grad(set_to_none=True)
            u = model(x, L=L)

            if method == 'PINN':
                core = pinn_loss(model, x, alpha, V0, use_avg=use_avg, n_theta=n_theta)
                E_track.append(float(model.energy.detach().cpu().item()))
            elif method == 'DRM':
                core = drm_loss(model, x, alpha, V0, L, use_avg=use_avg, n_theta=n_theta)
                with torch.no_grad():
                    du_for_E = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=False, retain_graph=True)[0]
                    Vx = V_KH(x, alpha=alpha, V0=V0, use_avg=use_avg, n_theta=n_theta)
                    num = integral_mean(0.5*du_for_E**2 + Vx*u**2, L)
                    den = integral_mean(u**2, L) + 1e-12
                    E_track.append(float((num/den).detach().cpu().item()))
            else:
                raise ValueError("method must be 'PINN' | 'DRM' | 'WAN'")

            if lambda_data != 0:
                u_data   = u.index_select(0, idx_data)
                psi_data = psi_n_train.index_select(0, idx_data)
                data_loss = torch.mean((u_data - psi_data)**2)
            else:
                data_loss = torch.tensor(0.0, device=device)

            orth = orth_loss_from_gt(u, L, lower_refs_train)
            norm_pen = (integral_mean(u**2, L) - 1.0)**2
            bc = (u[0]**2 + u[-1]**2)

            if lambda_party != 0.0:
                u_swap = model(-x, L=L)
                party_loss = torch.mean((u - u_swap)**2) if (n % 2 == 0) else torch.mean((u + u_swap)**2)
            else:
                party_loss = torch.tensor(0.0, device=device)

            loss = (lambda_pde * core
                    + lambda_data * data_loss
                    + lambda_orth * orth
                    + lambda_norm * norm_pen
                    + lambda_bc * bc
                    + lambda_party * party_loss)

            loss.backward()
            opt.step()

        # track best on train grid (sign-ambiguous)
        with torch.no_grad():
            u_pred = model(x, L=L)
            l2_pos = torch.mean((u_pred - psi_n_train)**2).item()
            l2_neg = torch.mean((u_pred + psi_n_train)**2).item()
            L2 = min(l2_pos, l2_neg)
            L2s.append(L2)
            Losses.append(float(loss.detach().cpu().item()))
            if L2 < best['metric']:
                best['metric'] = L2
                best['state']  = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                best['E']      = float(model.energy.detach().cpu().item())
                best['epoch']  = ep

    if best['state'] is not None:
        model.load_state_dict(best['state'])

    return {
        'model': model,
        'best_epoch': best['epoch'],
        'E_est': best['E'],
        'L2': best['metric'],
        'Losses': Losses,
        'E_track': E_track,
        'L2s': L2s,
        'idx_data': idx_data.detach().cpu().numpy(),
    }

# ============== plotting helper ==============
def plot_solution_gt(x_plot, psi_ref_plot, u_pred_plot, Vx_plot, E_est, method, n, L, out_png):
    x_np   = x_plot.detach().cpu().numpy()
    ref_np = psi_ref_plot.detach().cpu().numpy()
    up_np  = u_pred_plot.detach().cpu().numpy()
    V_np   = Vx_plot.detach().cpu().numpy()

    # align sign for visual
    if np.mean((up_np - ref_np)**2) > np.mean(((-up_np) - ref_np)**2):
        up_np = -up_np

    plt.figure(figsize=(10,6))
    plt.plot(x_np, ref_np, label=f"ref ψ_n (n={n})", linewidth=2)
    plt.plot(x_np, up_np,  label=f"{method} ψ_pred", linestyle='--')
    plt.plot(x_np, V_np,   label="V_KH(x)", alpha=0.7)
    if E_est is not None:
        plt.title(f"{method} vs Reference | n={n} | Ê≈{E_est:.6f}")
    else:
        plt.title(f"{method} vs Reference | n={n} | Ê (tracked)")
    plt.xlabel('x (a.u.)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=180)
    plt.close()

# ============== full run_compare ==============
def run_compare(*,
                # physics / GT
                alpha=10.0, V0=-24.856, L=60.0, N_ref=5000, n_max=4,
                use_avg=True, n_theta=500,
                # training grid / nets
                train_N=1024, layers=[1,100,100,100,1], technique='FBC',
                v_layers=[1,50,50,50,1], v_steps=3,
                # optim / epochs
                epochs=10000, lr=1e-3,
                # loss weights
                lambda_pde=10.0, lambda_data=1e4, lambda_orth=1e4, lambda_norm=10.0, lambda_bc=1e4, lambda_party=1e4,
                # data usage
                data_fraction=0.25, max_data_points=128,
                # methods
                methods=('PINN','DRM','WAN'),
                # IO
                save_dir='results/KH_1D_Unified',
                results_filename='results_KH_1D_unified.json',
                # device
                device=None):
    """
    Unified comparison: builds GT once, trains methods×levels, saves models/plots/curves, returns a summary list.
    """
    device = torch.device('cuda' if (device is None and torch.cuda.is_available()) else (device or 'cpu'))
    os.makedirs(save_dir, exist_ok=True)
    results_json = os.path.join(save_dir, results_filename)
    if not os.path.exists(results_json):
        with open(results_json, 'w') as f:
            json.dump([], f)
            print(f"[run_compare] Created results file at {results_json}")

    # GT once
    n_levels = max(n_max + 2, 10)
    gt = KH1DGroundTruth(alpha=alpha, V0=V0, L=L, N=N_ref,
                         n_levels=n_levels, use_avg=use_avg, n_theta=n_theta, device=device)
    x_ref  = gt.grid
    V_ref  = gt.V
    E_ref  = gt.E
    psi_gt = gt.psi

    # training grid
    x_train = torch.linspace(-L, L, train_N, device=device, dtype=torch.float32)
    x_train.requires_grad_(True)

    summary_all = []
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')

    for n in range(n_max):
        for method in methods:
            t0 = time()

            res = train_state_v2(
                method=method, n=n, gt=gt,
                x_train=x_train, L=L,
                layers=layers,
                technique=('RAW' if method == 'WAN' else technique),
                v_layers=v_layers, v_steps=v_steps,
                epochs=epochs, lr=lr,
                lambda_pde=lambda_pde, lambda_data=lambda_data, lambda_orth=lambda_orth,
                lambda_norm=lambda_norm, lambda_bc=lambda_bc, lambda_party=lambda_party,
                data_fraction=data_fraction, max_data_points=max_data_points,
                use_avg=use_avg, n_theta=n_theta,
                device=device, alpha=alpha, V0=V0
            )
            elapsed = time() - t0

            with torch.no_grad():
                u_pred_dense = res['model'](x_ref, L=L)
                psi_n_ref_dense = psi_gt[:, n]
                l2_pos = torch.mean((u_pred_dense - psi_n_ref_dense)**2).item()
                l2_neg = torch.mean((u_pred_dense + psi_n_ref_dense)**2).item()
                L2_dense = min(l2_pos, l2_neg)

            # save model
            model_path = os.path.join(save_dir, f"KH1D_{method}_best_n{n}_alpha{alpha:+.3f}_{timestamp}.pth")
            torch.save({k: v.cpu() for k, v in res['model'].state_dict().items()}, model_path)

            # save plot
            plot_path = os.path.join(save_dir, f"KH1D_{method}_n{n}_alpha{alpha:+.3f}_{timestamp}.png")
            plot_solution_gt(x_ref, psi_n_ref_dense, u_pred_dense, V_ref, res['E_est'], method, n, L, plot_path)

            # save curves
            losses_npy = os.path.join(save_dir, f"KH1D_{method}_n{n}_losses_{timestamp}.npy")
            l2s_npy    = os.path.join(save_dir, f"KH1D_{method}_n{n}_L2_{timestamp}.npy")
            Etrack_npy = os.path.join(save_dir, f"KH1D_{method}_n{n}_Etrack_{timestamp}.npy")
            np.save(losses_npy, np.array(res['Losses'], dtype=np.float64))
            np.save(l2s_npy,    np.array(res['L2s'],    dtype=np.float64))
            np.save(Etrack_npy, np.array(res['E_track'],dtype=np.float64))

            row = {
                'method': method,
                'n': int(n),
                'alpha': float(alpha),
                'V0': float(V0),
                'L': float(L),
                'use_avg': bool(use_avg),
                'n_theta': int(n_theta),
                'train_N': int(train_N),
                'epochs': int(epochs),
                'lr': float(lr),
                'technique': ('RAW' if method == 'WAN' else technique),
                'E_ref': float(E_ref[n].item()),
                'E_est': float(res['E_est']) if (res['E_est'] is not None) else None,
                'L2_error_train_best': float(res['L2']),
                'L2_error_dense': float(L2_dense),
                'elapsed_time_sec': float(elapsed),
                'best_epoch': int(res['best_epoch']),
                'time_of_best_epoch_est': (elapsed * res['best_epoch'] / epochs) if res['best_epoch'] >= 0 else None,
                'model_path': model_path,
                'plot_path': plot_path,
                'losses_npy': losses_npy,
                'l2s_npy': l2s_npy,
                'Etrack_npy': Etrack_npy,
                'timestamp': timestamp,
                'data_fraction': float(data_fraction),
                'max_data_points': (int(max_data_points) if max_data_points is not None else None),
                'v_steps': (int(v_steps) if method == 'WAN' else None),
            }
            summary_all.append(row)

            print(f"[n={n}] {method}: "
                  f"E_ref={row['E_ref']:.6f}, E_est={row['E_est'] if row['E_est'] is not None else 'N/A'}, "
                  f"L2_train_best={row['L2_error_train_best']:.3e}, L2_dense={row['L2_error_dense']:.3e}, "
                  f"time={elapsed:.1f}s")

    # append to JSON
    if os.path.exists(results_json):
        with open(results_json, 'r') as f:
            blob = json.load(f)
        if not isinstance(blob, list):
            blob = [blob]
    else:
        blob = []
    blob.extend(summary_all)
    with open(results_json, 'w') as f:
        json.dump(blob, f, indent=2)

    return summary_all

# ============== quick demo ==============
if __name__ == '__main__':
    # Example quick run (reduce epochs if you want a smoke test)
    out = run_compare(
        alpha=10.0, V0=-24.856, L=60.0, N_ref=5000, n_max=4,
        use_avg=True, n_theta=500,
        train_N=1024, layers=[1,100,100,100,1], technique='FBC',
        v_layers=[1,50,50,50,1], v_steps=3,
        epochs=10000, lr=1e-3,  # tip: increase epochs for better accuracy
        lambda_pde=10.0, lambda_data=1e4, lambda_orth=1e4, lambda_norm=10.0, lambda_bc=1e4, lambda_party=1e4,
        data_fraction=0.5, max_data_points=500,
        methods=('PINN','DRM','WAN'),
        save_dir='results/KH_1D',
        results_filename='results_KH_1D.json',
        device=device_default
    )
    # alphas = [0, 5, 10, 15, 20]
    # for alpha in alphas:
    #     run_compare(
    #         alpha=alpha, V0=-24.856, L=60.0, N_ref=5000, n_max=1,
    #         use_avg=True, n_theta=500,
    #         train_N=1024, layers=[1,100,100,100,1], technique='FBC',
    #         v_layers=[1,50,50,50,1], v_steps=5,
    #         epochs=10000, lr=1e-3,
    #         lambda_pde=10.0, lambda_data=1e5, lambda_orth=1e4, lambda_norm=10.0, lambda_bc=1e4, lambda_party=1e4,
    #         data_fraction=0.5, max_data_points=500,
    #         methods=('PINN','DRM','WAN'),
    #         save_dir='results/KH_1D',
    #         results_filename=f'results_KH_1D_alpha{alpha}.json',
    #         device=device_default
    #     )
