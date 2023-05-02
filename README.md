This is project is developed for one of my grad classes at Illinois Tech. It's an ongoing research project, henceforth the codes and notes are part of an ongoing process.

## PINNs

Physics-Informed Neural Networks (PINNs) are a type of neural network that can solve partial differential equations. PINNs combine the flexibility of neural networks with the physical constraints of the underlying partial differential equations.

In this project, we use a PINN to solve the one-dimensional diffusion equation. The PINN takes the initial and boundary conditions as input and predicts the concentration of the diffusing substance at any point in space and time.

# Diffusion Equation Solution using PINNs

This project aims to solve the one-dimensional diffusion equation using Physics-Informed Neural Networks (PINNs) and visualize the results.

## Diffusion Equation

The diffusion equation is a partial differential equation that describes the diffusion of a substance in space. The one-dimensional form of the diffusion equation is given by:

$$
\frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2}
$$

where $u(x, t)$ is the concentration of the diffusing substance at position $x$ and time $t$, and $D$ is the diffusion coefficient.


# Navier-Stokes Equation Solution using PINNs

The flow field around a fluid object, such as an airfoil, is an essential problem in fluid mechanics that has practical applications in various fields, including aerospace, marine, and civil engineering. Solving the Navier-Stokes equations to predict the flow field around a fluid object is a challenging task due to the complex geometry and boundary conditions involved. Recently, physics-informed neural networks (PINNs) have emerged as a promising technique to solve partial differential equations (PDEs) accurately and efficiently. This proposal aims to investigate the application of PINNs in fluid mechanics to predict the flow field around a fluid object by solving the Navier-Stokes equations.

## A more detailed explanation of how PINNs can be used to solve the Navier-Stokes equations for fluid flow around an airfoil. The funndamental Navier-Stokes equations describe the motion of fluids and are given by:

$$\rho\left(\frac{\partial\textbf{v}}{\partial t}+\textbf{v}\cdot\nabla\textbf{v}\right)=-\nabla p+\mu\nabla^2\textbf{v}+\textbf{f}$$
$$\nabla\cdot\textbf{v}=0$$

where $\rho$ is the fluid density, $\textbf{v}$ is the velocity field, $p$ is the pressure field, $\mu$ is the fluid viscosity, and $\textbf{f}$ is the external force field.

## Objective

The main objective of this proposal is to develop a physics-informed neural network model to predict the flow field around a fluid object by solving the Navier-Stokes equations. The specific objectives follows by constructing a dataset of Navier-Stokes solutions for the flow field around a fluid object using a numerical method, such as finite volume or finite element methods. The dataset will then be used to develop the PINN model to predict the flow field around a fluid object by incorporating the Navier-Stokes equations as a constraint. Finally, the PINN model will be trained using the dataset of Navier-Stokes solutions and evaluate its performance in predicting the flow field around a fluid object.

## Results

The PINN successfully solves the one-dimensional diffusion equation and produces the following plots:

- The predicted concentration of the diffusing substance at specific times during the diffusion process.
- The concentration of the diffusing substance at different values of the diffusion coefficient on the same plot, for comparison.
- The heat flux or gradient of the concentration distribution, to gain insights into the behavior of the system.
- A contour plot of the concentration distribution, to better visualize the boundaries and gradients.

## Requirements

The following packages are required to run this project:

- TensorFlow
- Scienceplot
- Latex
- NumPy
- SciPy
- Matplotlib

## Usage

To run the project, simply run the `diffusion_PINN.py` file. The output plots will be saved in the `plots` directory.



## Credits

This project is based on the work of Raissi et al. (2019) and their implementation of PINNs for solving the diffusion equation. 

## References

Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics, 378, 686-707.


### Questions?
The codes I wrote here are purely for eucational purposes, so please take caution when you refer to them. For questions, comments, edit and changes, please contact me. Or tag my username <b> @eyobghiday </b> in the code.
