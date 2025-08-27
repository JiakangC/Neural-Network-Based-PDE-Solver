import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
import os, json, datetime

# ======================= Nets =======================
class Sin(nn.Module):
    def forward(self, x): return torch.sin(x)

class SolutionNet(nn.Module):

    def __init__(self, dim, width=64, depth=5, bc_mode='FBC'):
        super().__init__()
        self.dim = dim
        self.bc_mode = bc_mode
        layers = []
        in_f = dim
        for _ in range(depth-1):
            layers += [nn.Linear(in_f, width), Sin()]
            in_f = width
        layers += [nn.Linear(in_f, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, X, L=2.0):
        u = self.net(X)
        if self.bc_mode == 'FBC':
            bf = torch.prod(X*(L - X), dim=1, keepdim=True)
            return bf * u
        elif self.bc_mode == 'RB':
            return u
        else:
            raise ValueError("bc_mode must be 'FBC' or 'RB'")

class CriticNet(nn.Module):

    def __init__(self, dim, width=64, depth=3):
        super().__init__()
        layers = []
        in_f = dim
        for _ in range(depth-1):
            layers += [nn.Linear(in_f, width), Sin()]
            in_f = width
        layers += [nn.Linear(in_f, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, X): return self.net(X)

# =================== Manufactured sol ===================
def exact_u_prod_sin(X, L, ks):
    # u*(x) = Π sin(k_i π x_i / L)
    vals = [torch.sin(k*torch.pi*X[:, i:i+1] / L) for i, k in enumerate(ks)]
    return torch.prod(torch.cat(vals, dim=1), dim=1, keepdim=True)

def rhs_f_for_u_sin(X, L, ks):

    u = exact_u_prod_sin(X, L, ks)
    s = sum((k*np.pi/L)**2 for k in ks)
    return X.new_tensor(s).view(1,1) * u

# =================== Autograd helpers ===================
def grad_scalar_field(u, X):
    return torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]  # (N,d)

def laplacian(u, X):
    g = grad_scalar_field(u, X)  # (N,d)
    parts = []
    for i in range(X.shape[1]):
        gi = g[:, i:i+1]
        dgi_dX = torch.autograd.grad(gi, X, grad_outputs=torch.ones_like(gi), create_graph=True)[0]
        parts.append(dgi_dX[:, i:i+1])  # 取对 x_i 的二阶
    return torch.sum(torch.cat(parts, dim=1), dim=1, keepdim=True)  # (N,1)

# ======================= Bump w =========================
def bump_w(X: torch.Tensor, L: float):
    low, up = 0.0, L
    h = (up - low) / 2.0
    center = (up + low) / 2.0
    I1 = 0.210987  

    t = (X - center) / h              # (N,d)
    mask = (t.abs() < 1.0).to(X.dtype)
    phi = torch.exp(1.0 / (t*t - 1.0)) / I1
    phi = phi * mask

    w = torch.prod(phi, dim=1, keepdim=True)         # (N,1)
    dw = torch.autograd.grad(w, X, grad_outputs=torch.ones_like(w), create_graph=True)[0]
    dw = torch.nan_to_num(dw)
    return w, dw

# ===================== Loss functions ====================
def pinn_residual_loss(model, X_in, f_in, L):
    """ PINN: minimize || -Δu - f ||^2 """
    u = model(X_in, L)
    lap = laplacian(u, X_in)
    res = -lap - f_in
    return torch.mean(res**2)

def drm_energy_loss(model, X_in, f_in, L):
    """ Deep Ritz: minimize E[ 1/2 |∇u|^2 - f u ] """
    u = model(X_in, L)
    g = grad_scalar_field(u, X_in)
    energy_density = 0.5*torch.sum(g*g, dim=1, keepdim=True) - f_in*u
    return torch.mean(energy_density)

@torch.enable_grad()
def wan_losses(u_model, v_model, X, f_vals, L, eps=1e-8, v_reg_weight=0.0):

    assert X.requires_grad, "X must require_grad=True"

    u = u_model(X, L)
    v = v_model(X)

    w, dw = bump_w(X, L)
    gu = torch.autograd.grad(u, X, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    gv = torch.autograd.grad(v, X, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    phi = w * v
    grad_phi = dw * v + w * gv

    integrand = torch.sum(gu * grad_phi, dim=1, keepdim=True) - f_vals * phi
    weak_residual = torch.mean(integrand)
    phi_norm = torch.mean(phi * phi)

    loss_pde_u = (weak_residual ** 2) / (phi_norm + eps)
    v_reg = torch.mean(torch.sum(gv*gv, dim=1, keepdim=True) + v*v)
    loss_v = -torch.log(loss_pde_u + eps) + v_reg_weight * v_reg

    return loss_pde_u, loss_v, weak_residual.detach(), phi_norm.detach()

def boundary_loss_dirichlet(model, L, N_b_per_face, dim, device):
    def sample_face(i, at_L):
        X = torch.rand(N_b_per_face, dim, device=device)*L
        X[:, i] = L if at_L else 0.0
        X.requires_grad_(True)
        return X
    losses = []
    for i in range(dim):
        X0 = sample_face(i, False)
        XL = sample_face(i, True)
        losses += [torch.mean(model(X0, L)**2), torch.mean(model(XL, L)**2)]
    return sum(losses)/len(losses)

def norm_loss(u, mode='nontrivial', eps=1e-8):
    m2 = torch.mean(u**2)
    if mode == 'nontrivial': return 1.0/(m2 + eps)
    elif mode == 'l2':      return m2
    else: raise ValueError("norm mode should be 'nontrivial' or 'l2'")

# ======================= Training ========================
def train_poisson_nd(
    dim=2, L=2.0, ks=None,
    method='PINN',              # 'PINN' | 'DRM' | 'WAN'
    bc_mode='FBC',              # 'FBC'  | 'RB'
    n_interior=20000, n_boundary=4000, n_data=0,
    epochs=10000, lr=1e-3, width=64, depth=5,
    critic_width=64, critic_depth=3, critic_steps=3, wan_reg=1.0,
    norm_mode='nontrivial',
    weights=None,               # {'pde':1.0,'bc':1e4,'data':1e3,'norm':0.0}
    seed=0, save_path='results/ND_Poisson',
    save_best=True
):

    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_path, exist_ok=True)
    ks = ks if ks is not None else [1]*dim

    # weights
    w = {'pde':1.0,
         'bc': 1e4 if bc_mode=='RB' else 0.0,
         'data': 1e3 if n_data>0 else 0.0,
         'norm': 0.0}
    if weights: w.update(weights)

    # model and optimizer
    model = SolutionNet(dim, width, depth, bc_mode=bc_mode).to(device)
    opt_u = torch.optim.Adam(model.parameters(), lr=lr)

    # WAN critic
    if method == 'WAN':
        critic = CriticNet(dim, critic_width, critic_depth).to(device)
        opt_v = torch.optim.Adam(critic.parameters(), lr=lr)
    else:
        critic = None

    # sampling function
    def sample_interior(N):
        X = torch.rand(N, dim, device=device)*L
        X.requires_grad_(True)
        return X

    # sample interior points and rhs
    X_in = sample_interior(n_interior)
    f_in = rhs_f_for_u_sin(X_in, L, ks).detach()

    # optional data points
    if n_data > 0:
        X_data = torch.rand(n_data, dim, device=device)*L
        with torch.no_grad():
            u_data = exact_u_prod_sin(X_data, L, ks)
    else:
        X_data = None; u_data = None

    # history
    history = {'total':[], 'pde':[], 'bc':[], 'data':[], 'norm':[], 'l2':[]}
    # WAN specific
    history.update({'wan_loss_v':[], 'wan_weak':[], 'wan_phi_norm':[]})

    # save best
    best_l2 = float('inf')
    best_state_u = None
    best_state_v = None
    N_test = 10000

    for ep in tqdm(range(epochs)):
        if method in ['PINN','DRM']:
            opt_u.zero_grad()

            if method == 'PINN':
                pde_loss = pinn_residual_loss(model, X_in, f_in, L)
            else:
                pde_loss = drm_energy_loss(model, X_in, f_in, L)

            if bc_mode == 'RB':
                per_face = max(1, n_boundary // (2*dim))
                bc_l = boundary_loss_dirichlet(model, L, per_face, dim, device)
            else:
                bc_l = torch.tensor(0.0, device=device)

            if n_data > 0:
                u_pred_data = model(X_data, L)
                data_l = torch.mean((u_pred_data - u_data)**2)
            else:
                data_l = torch.tensor(0.0, device=device)

            u_in = model(X_in, L)
            norm_l = norm_loss(u_in, mode=norm_mode) if w['norm']>0 else torch.tensor(0.0, device=device)

            loss = w['pde']*pde_loss + w['bc']*bc_l + w['data']*data_l + w['norm']*norm_l
            loss.backward(); opt_u.step()

        elif method == 'WAN':
            # v-step: 
            for _ in range(critic_steps):
                Xc = sample_interior(n_interior)
                fc = rhs_f_for_u_sin(Xc, L, ks).detach()
                loss_pde_u, loss_v, weak_r, phi_n = wan_losses(model, critic, Xc, fc, L, v_reg_weight=wan_reg)
                opt_v.zero_grad(); loss_v.backward(); opt_v.step()

            # u-step
            Xu = sample_interior(n_interior)
            fu = rhs_f_for_u_sin(Xu, L, ks).detach()
            loss_pde_u, _, weak_r, phi_n = wan_losses(model, critic, Xu, fu, L, v_reg_weight=wan_reg)

            if bc_mode == 'RB':
                per_face = max(1, n_boundary // (2*dim))
                bc_l = boundary_loss_dirichlet(model, L, per_face, dim, device)
            else:
                bc_l = torch.tensor(0.0, device=device)

            if n_data > 0:
                u_pred_data = model(X_data, L)
                data_l = torch.mean((u_pred_data - u_data)**2)
            else:
                data_l = torch.tensor(0.0, device=device)

            u_in = model(Xu, L)
            norm_l = norm_loss(u_in, mode=norm_mode) if w['norm']>0 else torch.tensor(0.0, device=device)

            loss = w['pde']*loss_pde_u + w['bc']*bc_l + w['data']*data_l + w['norm']*norm_l
            opt_u.zero_grad(); loss.backward(); opt_u.step()

            history['wan_loss_v'].append(float(loss_v.detach().cpu()))
            history['wan_weak'].append(float(weak_r))
            history['wan_phi_norm'].append(float(phi_n))
            pde_loss = loss_pde_u 
        else:
            raise ValueError("method must be one of {'PINN','DRM','WAN'}")

        # evaluate L2 error
        with torch.no_grad():
            X_te = torch.rand(N_test, dim, device=device)*L
            u_pred = model(X_te, L)
            u_ex = exact_u_prod_sin(X_te, L, ks)
            l2 = torch.mean((u_pred - u_ex)**2).sqrt().item()

        # record history
        history['total'].append(float(loss.detach().cpu()))
        history['pde'].append(float(pde_loss.detach().cpu()))
        history['bc'].append(float(bc_l.detach().cpu()))
        history['data'].append(float(data_l.detach().cpu()))
        history['norm'].append(float(norm_l.detach().cpu()))
        history['l2'].append(float(l2))

        # save best
        if save_best and l2 < best_l2:
            best_l2 = l2
            best_state_u = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            if method == 'WAN':
                best_state_v = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}

    # ===== save results =====
    stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    tag = f"{method}_{bc_mode}_d{dim}_ks{'-'.join(map(str,ks))}_{stamp}"
    os.makedirs(save_path, exist_ok=True)

    # save last model
    last_ckpt = os.path.join(save_path, f'{tag}_last.pth')
    payload = {
        'method': method, 'bc_mode': bc_mode,
        'model': model.state_dict(),
        'dim': dim, 'L': L, 'ks': ks,
        'width': width, 'depth': depth,
        'history': history
    }
    if method == 'WAN':
        payload['critic'] = critic.state_dict()
        payload.update({'critic_width':critic_width,'critic_depth':critic_depth,'critic_steps':critic_steps,'wan_reg':wan_reg})
    torch.save(payload, last_ckpt)

    # save best
    best_ckpt_u, best_ckpt_v = None, None
    if save_best and best_state_u is not None:
        best_ckpt_u = os.path.join(save_path, f'{tag}_best_u.pth')
        torch.save(best_state_u, best_ckpt_u)
        if method == 'WAN' and best_state_v is not None:
            best_ckpt_v = os.path.join(save_path, f'{tag}_best_v.pth')
            torch.save(best_state_v, best_ckpt_v)

    # save history npy
    for k, v in history.items():
        np.save(os.path.join(save_path, f'{tag}_{k}.npy'), np.array(v))

    # save results JSON
    results_file = os.path.join(save_path, 'results_poisson_nd.json')
    entry = {
        'tag': tag,
        'timestamp': stamp,
        'method': method, 'bc_mode': bc_mode,
        'dim': dim, 'L': L, 'ks': ks,
        'width': width, 'depth': depth,
        'n_interior': n_interior, 'n_boundary': n_boundary, 'n_data': n_data,
        'epochs': epochs, 'lr': lr,
        'final_l2': history['l2'][-1] if history['l2'] else None,
        'best_l2': best_l2 if best_l2 < float('inf') else None,
        'last_ckpt': last_ckpt,
        'best_ckpt_u': best_ckpt_u,
        'best_ckpt_v': best_ckpt_v
    }
    if method == 'WAN':
        entry.update({'critic_width':critic_width,'critic_depth':critic_depth,'critic_steps':critic_steps,'wan_reg':wan_reg})
    try:
        if os.path.exists(results_file):
            with open(results_file, 'r') as f:
                arr = json.load(f)
            if not isinstance(arr, list): arr = [arr]
        else:
            arr = []
        arr.append(entry)
        with open(results_file, 'w') as f:
            json.dump(arr, f, indent=2)
    except Exception as e:
        print("Could not update results file:", e)
    print(f"Training complete. Results saved to {save_path}, Best L2: {best_l2:.4e} at epoch {history['l2'].index(best_l2) if best_l2 < float('inf') else 'N/A'}")

    return model, history, {'last_ckpt': last_ckpt, 'best_u': best_ckpt_u, 'best_v': best_ckpt_v}

# ======================== main =========================
if __name__ == '__main__':
    # configurations to run
    dims = [2]
    L = 2.0
    methods = ['DRM', 'PINN', 'WAN']       # 'PINN' | 'DRM' | 'WAN'
    bc_mode = 'FBC'      # 'FBC'  | 'RB'
    for method in methods:
        for dim in dims:
            ks = [1]*dim
            n_interior = 20000
            n_boundary = 4000
            n_data = 0

            epochs = 10000
            lr = 1e-3
            width, depth = 64, 5

            critic_width, critic_depth, critic_steps, wan_reg = 64, 3, 5, 1.0
            norm_mode = 'nontrivial'
            weights = {'pde':1.0,
                    'bc': 1e4 if bc_mode=='RB' else 0.0,
                    'data': 1e3 if n_data>0 else 0.0,
                    'norm': 0.0}

            model, history, ckpts = train_poisson_nd(
                dim=dim, L=L, ks=ks,
                method=method, bc_mode=bc_mode,
                n_interior=n_interior, n_boundary=n_boundary, n_data=n_data,
                epochs=epochs, lr=lr, width=width, depth=depth,
                critic_width=critic_width, critic_depth=critic_depth, critic_steps=critic_steps, wan_reg=wan_reg,
                norm_mode=norm_mode, weights=weights,
                seed=0, save_path='results/ND_Poisson',
                save_best=True
            )
            print("Checkpoints:", ckpts)
