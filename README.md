# PINNs
This is a <b> two-part </b> project that is being developed for the spring of 2023 at Illinois Tech. It's an ongoing project, henceforth the codes and notes are part of the learning experience and not a final product. The first section of this project attempts to use PINN on diffusion 2D probelm. The main code is the `main_diffussion.py` file. The second project will focus on the Navier-Stokes equations, which I haven't finished yet. Please feel free to suggest me any ideas or inputs. 

Physics-Informed Neural Networks (PINNs) are a type of neural network that can solve partial differential equations. PINNs combine the flexibility of neural networks with the physical constraints of the underlying partial differential equations. In this project, we use a PINN to solve the one-dimensional diffusion equation. The PINN takes the initial and boundary conditions as input and predicts the concentration of the diffusing substance at any point in space and time. This project aims to solve the one-dimensional diffusion equation using Physics-Informed Neural Networks (PINNs) and visualize the results. This project also includes the full derviation of diffusion equation in both polar and carrtesian coordinates. The advantages, limitations, and opportunities of using physics-informed neural networks for data-driven simulations are also discussed in the [a report](Eyob_PINN_2D_Diffusion_Equation_[2023].pdf). 

## 1. Diffusion Equation

The diffusion equation is a partial differential equation that describes the diffusion of a substance in space. The one-dimensional form of the diffusion equation is given by:

$$ \frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} $$

where $u(x, t)$ is the concentration of the diffusing substance at position $x$ and time $t$, and $D$ is the diffusion coefficient.


## 2. Navier-Stokes Equation Solution using PINNs

Depending on the time and progres I will also work on the invisicd solution of the NS equatoin. The funndamental Navier-Stokes equations describe the motion of fluids and are given by:

$$\rho\left(\frac{\partial\textbf{v}}{\partial t}+\textbf{v}\cdot\nabla\textbf{v}\right)=-\nabla p+\mu\nabla^2\textbf{v}+\textbf{f}$$
$$\nabla\cdot\textbf{v}=0$$

where $\rho$ is the fluid density, $\textbf{v}$ is the velocity field, $p$ is the pressure field, $\mu$ is the fluid viscosity, and $\textbf{f}$ is the external force field.

## Objective

The main objective of this projeect is to develop a physics-informed neural network model to solve PDE's compeltely or partially. The Green’s function to the 2D Diffusion equation was solved using inverse Fourier transform first. The model was then trained on synthetic data generated by the exact solution of the diffusion equation, and its accuracy was assessed by comparing predicted data with test data using contour plots and normal graphs. In addition, the PINN model was applied to simulate Brownian motion of a particle in a 2D domain develop the PINN model to predict the flow field around a fluid object by incorporating the BC's as a constraint. Finally, the PINN model will be trained using the dataset based off of those PDE-equation solutions and evaluate its performance in predicting the flow of a fluid object.

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

To run the project, simply run the `main_diffussion.py` file. The output plots will be saved in the `plots` directory. You can also read the report and proposal to get a better understanding of what I'm trying to do. 

## References:
[1] Srivastava, A., & Sukumar, N. (2022). Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks. Computer Methods in Applied Mechanics and Engineering, 389, 114333. doi: 10.1016/j.cma.2021.114333.

[2] Willis, J. R. (1980). Polarization approach to the scattering of elastic waves-I. Scattering by a single inclusion. Journal of the Mechanics and Physics of Solids, 28(5-6), 287-305. doi: 10.1016/0022-5096(80)90021-6.

[3] Fernández de la Mata, F., Gijon, A., Molina-Solana, M., & Gomez-Romero, J. (2023). Physics-informed neural networks for data-driven simulation: Advantages, limitations, and opportunities. Physica A: Statistical Mechanics and its Applications, 610, 128415. doi: 10.1016/j.physa.2022.128415.

[4] Ursell, T. (2005). APh Physics Laboratory. Physics. Retrieved from http://www.physics.nyu.edu/grierlab/methods/node11.html.

[5] Tang, J., Azevedo, V. C., Cordonnier, G., & Solenthaler, B. (2022). Neural Green's function for Laplacian systems. Computers Graphics, 107, 186-196. doi:

[6] Eyob, G. (2023). Full 2D-Diffusion Equation Derivation. Retrieved from https://github.com/eyobghiday/PINNs-in-Fluid-Mechanics/blob/main/Eyob_Ghiday_Difussion_Derivation.pdf.

[7] Eyob, G. (2023). Application of PiNNs in 2D Diffusion Equation. Retrieved from https://github.com/eyobghiday/PINNs-in-Fluid-Mechanics.

[8] Arocha, M. A. (2018). Crank-Nicolson Method. Retrieved from https://matlabgeeks.weebly.com/uploads/8/0/4/8/8048228/crank_nicolson_method_presentation-v5.pdf.

[9] Hassan, A. A. (2017). Green's Function for the Heat Equation. Fluid Mechanics: Open Access, 04(02), 2-7. doi: 10.4172/2476-2296.1000152

[10] Skinner, D. (n.d.). Green's functions for PDEs. Retrieved from http://www.damtp.cam.ac.uk/user/dbs26/1
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

### Questions?
The codes I wrote here are purely for eucational purposes, so please take caution when you refer to them. For questions, comments, edit and changes, please contact me. Or tag my username <b> @eyobghiday </b> in the code.
