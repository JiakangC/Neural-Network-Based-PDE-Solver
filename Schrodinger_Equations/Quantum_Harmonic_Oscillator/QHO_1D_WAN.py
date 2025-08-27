import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import datetime
# torch.manual_seed(0)
# np.random.seed(0)
import json
from time import time
import datetime
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

def function_w(x, L):
    # x is assumed to require_grad=True already
    low, up = -L, L
    I1 = 0.210987
    h = (up - low) / 2.0
    center = (up + low) / 2.0

    # split into D columns
    x_list = torch.split(x, 1, dim=1)
    phi_list = []
    for xi in x_list:
        t = (xi - center) / h
        mask = t.abs() < 1.0
        phi = torch.where(mask,
                          torch.exp(1.0 / (t.pow(2) - 1.0)) / I1,
                          torch.zeros_like(t))
        phi_list.append(phi)

    w = torch.ones_like(phi_list[0])
    for phi in phi_list:
        w = w * phi

    dw = torch.autograd.grad(outputs=w,
                             inputs=x,
                             grad_outputs=torch.ones_like(w),
                             create_graph=True)[0]
    # if use:
    dw = torch.nan_to_num(dw)
    return w, dw
    # else:
    #     # return w = 1 and dw = 0
    #     return torch.ones_like(x), torch.zeros_like(x)

# Define the neural network for the PINN
class FCN(nn.Module):
    def __init__(self, layers, num_states=1, L=10.0, enforce_bc=False):
        super(FCN, self).__init__()
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
            B = (1 - torch.exp(-(x + self.L))) * (1 - torch.exp(x - self.L))
            return  y * B
        return y
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=nn.init.calculate_gain('tanh'))
                nn.init.zeros_(m.bias)

def WAN_loss(u_model, v_model, x, n, L, weight_pde=1.0, weight_norm=1.0):
    # bump w
    w, dw = function_w(x, L)

    # neural network outputs
    u_interior = u_model(x)
    v_interior = v_model(x)

    # residuals
    du_dx = torch.autograd.grad(u_interior, x, grad_outputs=torch.ones_like(u_interior), create_graph=True)[0]
    dv_dx = torch.autograd.grad(v_interior, x, grad_outputs=torch.ones_like(v_interior), create_graph=True)[0]

    # Weak residuals
    phi = w * v_interior
    dphi_dx = dw * v_interior + w * dv_dx
    prefactor = 0.5
    integral = prefactor * du_dx * dphi_dx + Potential(x) * u_interior * phi - u_model.energies * u_interior * phi

    weak_residual = torch.mean(integral)
    norm = torch.mean(phi**2)

    loss_pde = weak_residual**2 / (norm + 1e-8)  # Add small epsilon for stability
    loss_norm = (2 * L * torch.mean(u_interior**2) - 1.0)**2  # Enforce ∫u² dx ≈ 1
    total_loss = weight_pde * loss_pde + weight_norm * loss_norm
    loss_v = -torch.log(loss_pde + 1e-8)  # Avoid log(0)
    return total_loss, loss_v, loss_pde, loss_norm

def Orthogonal_loss(model, x, n, L):
    # Sample points for computing the orthogonal loss
    u_pred = model(x)
    if n == 0:
        return torch.tensor(0.0, device=x.device)  # No orthogonal loss for n=0
    else:
        ortho_loss = 0.0
        for k in range(0, n):
            u_exact = Exact_solution(k, x)
            # Approximate inner product <u_pred, u_exact> ≈ mean(u_pred * u_exact) * 2L
            inner = torch.mean(u_pred * u_exact) * 2 * L
            # Approximate norm squared <u_exact, u_exact> ≈ mean(u_exact**2) * 2L
            norm_sq = torch.mean(u_exact ** 2) * 2 * L
            # Add the squared projection coefficient
            ortho_loss += (inner ** 2) / norm_sq
        return ortho_loss

def train_seperate(n, L=6, epochs=3000, lr=0.001, layers=[1, 200, 200, 200, 1], LBFGS=False, method='WAN', technique='BC'):
    # Define the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)
    start_time = time()
    # sample points
    x_interior = torch.linspace(-L, L, 1000).view(-1, 1).to(device)
    x_interior.requires_grad = True  # Enable gradients for interior points
    # sample data    
    x_full_data = torch.linspace(-L, L, 1000).view(-1, 1).to(device)
    u_full_data = Exact_solution(n, x_full_data)
    # set the percentage of data to use for training
    percentage = 0.25
    # Sample a subset of the data for training
    n_data = int(percentage * x_full_data.shape[0])

    x_data = x_full_data[n_data:2*n_data:10]  # Sample every 10th point for training
    u_data = u_full_data[n_data:2*n_data:10]

    # boundary conditions
    x_bc = torch.tensor([[-L], [L]], dtype=x_interior.dtype, device=device)
    if technique == 'OG':
        weight_orth = 1000.0
    else:
        weight_orth = 0.0    
    # model
    weight_data = 1000.0
    weight_pde = 10.0
    weight_norm = 10.0
    u_layers = layers
    v_layers = [1, 100, 100, 100, 1]
    if technique == 'BC':
        u_model = FCN(u_layers, num_states=n, L=L, enforce_bc=False).to(device)
        v_model = FCN(v_layers, num_states=n, L=L, enforce_bc=False).to(device)
    else:
        u_model = FCN(u_layers, num_states=n, L=L, enforce_bc=True).to(device)
        v_model = FCN(v_layers, num_states=n, L=L, enforce_bc=True).to(device)
    if u_model.enforce_bc:
        weight_bc = 0.0
    else:
        weight_bc = 1000.0
    # Optimizer
    u_optimizer = torch.optim.Adam(u_model.parameters(), lr=lr)
    v_optimizer = torch.optim.Adam(v_model.parameters(), lr=lr)
    if LBFGS:
        u_optimizer_LBFGS = torch.optim.LBFGS(u_model.parameters(), lr=lr, max_iter=200, line_search_fn='strong_wolfe')
    Losses = []
    L2_errors = []
    best_L2 = float('inf')
    best_state_u = None
    best_state_v = None
    best_epoch = -1
    # Training loop
    for epoch in tqdm(range(epochs)):
        # Train v (multiple steps for better adversarial challenge)
        for p in u_model.parameters():
            p.requires_grad = False
        for p in v_model.parameters():
            p.requires_grad = True
        for _ in range(5):  # 5 updates for v per iteration
            total_loss, loss_v, _, _ = WAN_loss(u_model, v_model, x_interior, n, L, weight_pde, weight_norm)
            v_optimizer.zero_grad()
            loss_v.backward()
            v_optimizer.step()

        # Train u
        for p in u_model.parameters():
            p.requires_grad = True
        for p in v_model.parameters():
            p.requires_grad = False
        total_loss, _, loss_pde, loss_norm = WAN_loss(u_model, v_model, x_interior, n, L, weight_pde, weight_norm)
        ortho_loss = Orthogonal_loss(u_model, x_interior, n, L)
        loss_data = torch.mean((u_model(x_data) - u_data)**2)
        loss_bc = torch.mean((u_model(x_bc))**2)
        loss = total_loss + weight_orth * ortho_loss + weight_data * loss_data + weight_bc * loss_bc
        u_optimizer.zero_grad()
        loss.backward()
        u_optimizer.step()
        Losses.append(loss.item())
        # track L2 error
        with torch.no_grad():
            u_test = u_model(x_full_data)
            u_exact = Exact_solution(n, x_full_data)
            l2_pos = torch.mean((u_test - u_exact)**2).item()
            l2_neg = torch.mean((u_test + u_exact)**2).item()
            L2_error = min(l2_pos, l2_neg)
            L2_errors.append(L2_error)
            if L2_error < best_L2:
                best_L2 = L2_error
                best_state_u = {k: v.detach().cpu().clone() for k, v in u_model.state_dict().items()}
                best_state_v = {k: v.detach().cpu().clone() for k, v in v_model.state_dict().items()}
                best_epoch = epoch
    if LBFGS:
        def closure():
            u_optimizer_LBFGS.zero_grad()
            total_loss, _, _, _ = WAN_loss(u_model, v_model, x_interior, n, L)
            ortho_loss = Orthogonal_loss(u_model, x_interior, n, L)
            loss = total_loss + weight_orth * ortho_loss
            loss.backward()
            return loss
        u_optimizer_LBFGS.step(closure)
    end_time = time()
    print(f'Training for {method} completed in {end_time - start_time:.2f} seconds')
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    # Save the best model
    if best_state_u is not None:
        path_u = os.path.join(save_path, f'{method}_best_model_u_n{n}_{technique}_{timestamp}.pth')
        path_v = os.path.join(save_path, f'{method}_best_model_v_n{n}_{technique}_{timestamp}.pth')
        torch.save(best_state_u, path_u)
        torch.save(best_state_v, path_v)
        print(f'Best model saved for {method} n={n} with L2 error {best_L2:.4e} at epoch {best_epoch}')
    # Save Losses and L2_errors as numpy arrays
    np.save(os.path.join(save_path, f'{method}_losses_n{n}_{technique}_{timestamp}.npy'), np.array(Losses))
    np.save(os.path.join(save_path, f'{method}_L2_errors_n{n}_{technique}_{timestamp}.npy'), np.array(L2_errors))

    L2_error = min(L2_errors)
    # find the order of the minimum L2 error
    min_epoch = L2_errors.index(L2_error)
    print(f'Minimum L2 Error for n={n}: {L2_error:.4e} at epoch {min_epoch}')
    # load json file to track the results

    results = {
        'method': method,
        'n': n,
        'epochs': epochs,
        'LBFGS': LBFGS,
        'L2_error': L2_error,
        'min_epoch': min_epoch,
        'best_model_path': os.path.join(save_path, f'{method}_best_model_u_n{n}_{technique}_{timestamp}.pth'),
        'losses': os.path.join(save_path, f'{method}_losses_n{n}_{technique}_{timestamp}.npy'),
        'L2_errors': os.path.join(save_path, f'{method}_L2_errors_n{n}_{technique}_{timestamp}.npy'),
        'time': end_time - start_time,
        'time_of_best_model': best_epoch * (end_time - start_time) / epochs,
        'technique': technique,
        'weight_data': weight_data,
        'weight_pde': weight_pde,
        'weight_norm': weight_norm,
        'weight_bc': weight_bc,
        'weight_orth': weight_orth,
        'percentage': percentage,
        'Comment': 'WAN with w function for QHO',
        'timestamp': timestamp
    }
    
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

    # Evaluate the model
    # Load best model for evaluation
    best_model = FCN(u_layers, num_states=n, L=L, enforce_bc=True).to(device)
    best_model.load_state_dict(torch.load(path_u, map_location=device))
    best_model.eval()
    with torch.no_grad():
        u_test = best_model(x_full_data)
        u_exact = Exact_solution(n, x_full_data)
    # Plot the results
    l2_pos = torch.mean((u_test - u_exact)**2).item()
    l2_neg = torch.mean((u_test + u_exact)**2).item()
    u_plot = u_test.cpu().numpy() if l2_pos <= l2_neg else -u_test.cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x_full_data.cpu().numpy(), u_plot, label='WAN Solution', color='blue')
    plt.plot(x_full_data.cpu().numpy(), u_exact.cpu().numpy(), label='Exact Solution', color='red', linestyle='--')
    plt.scatter(x_data.cpu().numpy(), u_data.cpu().numpy(), color='green', s=10, label='Training Data', alpha=0.5)
    plt.title(f'Solution for n={n} (QHO)')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'WAN_solution_n{n}_seperate_method_{technique}.png'))
    plt.close()
    # plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(Losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Training Loss (QHO)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'WAN_loss_n{n}_seperate_method_{technique}.png'))
    plt.close()
    # plot the L2 error
    plt.figure(figsize=(10, 5))
    plt.plot(L2_errors, label='L2 Error')
    plt.xlabel('Epochs')
    plt.ylabel('L2 Error Value')
    plt.title('L2 Error over Epochs (QHO)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'WAN_L2_error_n{n}_seperate_method_{technique}.png'))
    plt.close()
    return u_model

def run_seperate_method(n_values, epochs=3000, LBFGS=False):
    for n in n_values:
        print(f"Training for n={n}...")
        # model_wan = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='WAN', technique='BC')
        # model_wan_orth = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='WAN', technique='OG')
        model_wan_fbc = train_seperate(n, epochs=epochs, LBFGS=LBFGS, method='WAN', technique='FBC')
        print(f"Training completed for n={n}.")

if __name__ == "__main__":
    n_values = [0]  # You can adjust this list for more eigenstates
    run_seperate_method(n_values, epochs=10000, LBFGS=False)