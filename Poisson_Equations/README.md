# Poisson_Equations — Usage

This folder contains the N-D Poisson example solved with **PINN**, **DRM**, and **WAN** using a manufactured solution.

## Quick start
```bash
python Poisson_Equations/Poisson_ND.py
```

## Main Works

- Samples interior/boundary points on the N-dimensional domain.
- Trains one of: PINN/DRM/WAN
- Saves Loss, L2 history, model and a JSON summary in `results/ND_Poisson/`

## Change hyperparameters

Open `Poisson_ND.py` and edit the `__main__` block near the bottom:

- methods = ['DRM', 'PINN', 'WAN']  &rarr choose methods to run.
- dims = [2] &rarr spatial dimensions
- epochs, lr, width, depth &rarr training schedule and model size.
- n_interior, n_boundary, n_data &rarr sample sizes
- bc_mode = 'FBC' or 'RB' &rarr fixed BC vs. residual-based BC

## Outputs

- `results/ND_Poisson/losses.npy`, `L2_errors.npy`
- `results/ND_Poisson/results_poisson_nd.json` hyper-params, evaluation metrics
- Check points for best models

