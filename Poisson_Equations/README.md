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
