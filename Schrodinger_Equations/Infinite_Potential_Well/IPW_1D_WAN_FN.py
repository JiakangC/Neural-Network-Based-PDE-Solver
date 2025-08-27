import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
# torch.manual_seed(0)
# np.random.seed(0)
import json
from time import time
import datetime
save_path = 'results/Infinite_Potential_Well_1D_Compare_For_Paper'
import os
if not os.path.exists(save_path):
    os.makedirs(save_path)

results_file = os.path.join(save_path, 'results_IPW_1D_FN.json')
if not os.path.exists(results_file):
    with open(results_file, 'w') as f:
        json.dump([], f)  # Initialize with an empty list
        print(f"Created results file at {results_file}")
# problem 
def Exact_solution(L, n, x):
    return torch.sqrt(torch.tensor(2.0/L, dtype=x.dtype, device=x.device)) * torch.sin(n * torch.pi * x / L)

def Exact_energy(n, L):
    hbar = 1.0
    m = 1.0
    return (n * np.pi * hbar) ** 2 / (2 * m * L**2)

def function_w(x, L):
    # x is assumed to require_grad=True already
    low, up = 0.0, L
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
    dw = torch.nan_to_num(dw)
    return w, dw

# Define the neural network for the PINN
class FCN(nn.Module):
    def __init__(self, layers, num_states=1, L=2.0, enforce_bc=False):
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
        
    def forward(self, x):
        L = self.L
        num_states = self.num_states
        for k in range(num_states):
            n = k + 1
            f =  x * (L-x)
            if n > 1:
                for j in range(1, n):
                    f *= (x - j * L/ n)
        return f * self.net(x)
    
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
    hbar = 1.0
    mass = 1.0
    prefactor = hbar**2 / (2 * mass)
    integral = prefactor * du_dx * dphi_dx - Exact_energy(n, L) * u_interior * phi

    weak_residual = torch.mean(integral)
    norm = torch.mean(phi**2)

    loss_pde = weak_residual**2 / (norm + 1e-8)  # Add small epsilon for stability
    loss_norm = (L * torch.mean(u_interior**2) - 1.0)**2  # Enforce ∫u² dx ≈ 1
    total_loss = weight_pde * loss_pde + weight_norm * loss_norm
    loss_v = -torch.log(loss_pde + 1e-8)  # Avoid log(0)
    return total_loss, loss_v, loss_pde, loss_norm

def Orthogonal_loss(model, x, n, L):
    # Sample points for computing the orthogonal loss
    u_pred = model(x)
    if n == 1:
        return torch.tensor(0.0, device=x.device)  # No orthogonal loss for n=1
    else:
        ortho_loss = 0.0
        for k in range(1, n):
            u_exact = Exact_solution(L, k, x)
            # Approximate inner product <u_pred, u_exact> ≈ mean(u_pred * u_exact) * L
            inner = torch.mean(u_pred * u_exact) * L
            # Approximate norm squared <u_exact, u_exact> ≈ mean(u_exact**2) * L
            norm_sq = torch.mean(u_exact ** 2) * L
            # Add the squared projection coefficient
            ortho_loss += (inner ** 2) / norm_sq
        return ortho_loss

def train_seperate(n, L=2.0, epochs=3000, lr=0.001, layers=[1, 50, 50, 50, 1], v_layers=[1, 20, 20, 20, 1], LBFGS=False, method='WAN'):
    # Define the network
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    torch.manual_seed(0)
    start_time = time()
    # sample points
    x_interior = torch.linspace(0, L, 1000).view(-1, 1).to(device)
    x_interior.requires_grad = True  # Enable gradients for interior points
    # sample data    
    x_full_data = torch.linspace(0, L, 1000).view(-1, 1).to(device)
    u_full_data = Exact_solution(L, n, x_full_data)
    percentage = 0.25
    # Sample a subset of the data for training
    n_data = int(percentage * x_full_data.shape[0])

    x_data = x_full_data[0:n_data:10]  # Sample every 10th point for training
    u_data = u_full_data[0:n_data:10]
    # boundary conditions
    x_bc = torch.tensor([[0.0], [L]], dtype=x_interior.dtype, device=device)
    
    # model
    weight_data = 0
    weight_pde = 10.0
    weight_norm = 1000.0
    weight_bc = 0.0  # Force the boundary condition
    weight_orth = 0.0  # Orthogonality loss
    u_layers = layers
    v_layers = v_layers
    u_model = FCN(u_layers, num_states=n, L=L, enforce_bc=True).to(device)
    v_model = FCN(v_layers, num_states=n, L=L, enforce_bc=False).to(device)
    
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
        loss = total_loss + weight_orth * ortho_loss + weight_data * loss_data 
        u_optimizer.zero_grad()
        loss.backward()
        u_optimizer.step()
        Losses.append(loss.item())
        # track L2 error
        with torch.no_grad():
            u_test = u_model(x_full_data)
            u_exact = Exact_solution(L, n, x_full_data)
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
        model_filename = f'{method}_best_model_n{n}_FN_{timestamp}.pth'
        path_u = os.path.join(save_path, model_filename)
        path_v = os.path.join(save_path, f'{method}_best_model_v_n{n}_FN_{timestamp}.pth')
        torch.save(best_state_u, path_u)
        torch.save(best_state_v, path_v)
        print(f'Best model saved for {method} n={n} with L2 error {best_L2:.4e} at epoch {best_epoch}')
    # Save Losses and L2_errors as numpy arrays
    losses_filename = f'{method}_losses_n{n}_FN_{timestamp}.npy'
    l2_errors_filename = f'{method}_L2_errors_n{n}_FN_{timestamp}.npy'
    np.save(os.path.join(save_path, losses_filename), np.array(Losses))
    np.save(os.path.join(save_path, l2_errors_filename), np.array(L2_errors))

    L2_error = min(L2_errors)
    # find the order of the minimum L2 error
    min_epoch = L2_errors.index(L2_error)
    print(f'Minimum L2 Error for n={n}: {L2_error:.4e} at epoch {min_epoch}')
    # load json file to track the results
    timestamp = time()
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
        'technique': 'FN',
        'weight_pde': weight_pde,
        'weight_drm': 0,
        'weight_data': weight_data,
        'weight_norm': weight_norm,
        'weight_bc': 0,
        'weight_orth': 0,
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

    # Evaluate the model
    # Load best model for evaluation
    best_model = FCN(u_layers, num_states=n, L=L, enforce_bc=True).to(device)
    best_model.load_state_dict(torch.load(path_u, map_location=device))
    best_model.eval()
    with torch.no_grad():
        u_test = best_model(x_full_data)
        u_exact = Exact_solution(L, n, x_full_data)
    # Plot the results
    l2_pos = torch.mean((u_test - u_exact)**2).item()
    l2_neg = torch.mean((u_test + u_exact)**2).item()
    u_plot = u_test.cpu().numpy() if l2_pos <= l2_neg else -u_test.cpu().numpy()
    plt.figure(figsize=(10, 6))
    plt.plot(x_full_data.cpu().numpy(), u_plot, label='WAN Solution', color='blue')
    plt.plot(x_full_data.cpu().numpy(), u_exact.cpu().numpy(), label='Exact Solution', color='red', linestyle='--')
    plt.title(f'Solution for n={n}')
    plt.xlabel('x')
    plt.ylabel('u(x)')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'WAN_solution_n{n}_seperate_method_BC.png'))
    plt.close()
    # plotting loss
    plt.figure(figsize=(10, 5))
    plt.plot(Losses, label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss Value')
    plt.title('Training Loss')
    plt.legend()
    plt.grid()
    plt.savefig(os.path.join(save_path, f'WAN_loss_n{n}_seperate_method_BC.png'))
    plt.close()
    return u_model

def run_seperate_method(n_values, epochs=3000, layers=None, v_layers=None, LBFGS=False):
    for n in n_values:
        print(f"Training for n={n}...")
        model_wan = train_seperate(n, epochs=epochs, layers=layers, v_layers=v_layers, LBFGS=LBFGS, method='WAN')
        print(f"Training completed for n={n}.")

if __name__ == "__main__":
    n_values = [5]  # You can adjust this list for more eigenstates
    layers_lists = [[1, 10, 1], [1, 10, 10, 1], [1, 10, 10, 10, 1], [1, 10, 10, 10, 10, 1], [1, 50, 1], [1, 50, 50, 1], [1, 50, 50, 50, 1], [1, 50, 50, 50, 50, 1], [1, 100, 1], [1, 100, 100, 1], [1, 100, 100, 100, 1], [1, 100, 100, 100, 100, 1]]  # Different layer configurations
    v_layers_lists = [[1, 5, 1], [1, 5, 5, 1], [1, 5, 5, 5, 1], [1, 5, 5, 5, 5, 1], [1, 20, 1], [1, 20, 20, 1], [1, 20, 20, 20, 1], [1, 20, 20, 20, 20, 1], [1, 50, 1], [1, 50, 50, 1], [1, 50, 50, 50, 1], [1, 50, 50, 50, 50, 1]] # Different layer configurations for v_model
    for layers, v_layers in zip(layers_lists, v_layers_lists):
        print(f"Training with layers: {layers} and v_layers: {v_layers}")
        # Run the training for each layer configuration
        run_seperate_method(n_values, epochs=10000, layers=layers, v_layers=v_layers, LBFGS=False)