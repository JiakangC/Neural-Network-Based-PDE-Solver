# Neural-Network-Based-PDE-Solver
This project explores how neural networks can be used to approximate solutions of partial differential equations (PDEs). Traditional numerical solvers rely on mesh‑based discretisations and domain‑specific techniques; in contrast, the methods implemented here embed the governing equations and boundary/initial conditions directly into the loss functions of neural networks. By leveraging automatic differentiation, these models learn to satisfy the physics of the problem while remaining flexible and mesh‑free. The repository was originally developed for an MSc Machine Learning final project and implements several modern neural‑PDE approaches, including Physics‑Informed Neural Networks (PINN), the Deep Ritz method (DRM) and Weighted Adversarial Networks (WAN)

## Features and methodologies
### Physics‑Informed Neural Networks (PINN)
PINNs approximate the solution of a PDE using a fully connected neural network. Automatic differentiation is used to evaluate derivatives of the network output; these are substituted into the PDE to form a residual loss. Boundary and initial conditions are enforced either through additional loss terms or by constructing the network output so that the conditions are satisfied identically.

![PINN Structure](Plots/PINN_structure.png)

