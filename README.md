# PINNs
This is project is being developed for one of my projects at Illinois Tech. It's an ongoing research project, henceforth the codes and notes are part of an ongoing process.
Physics-Informed Neural Networks (PINNs) are a type of neural network that can solve partial differential equations. PINNs combine the flexibility of neural networks with the physical constraints of the underlying partial differential equations. In this project, we use a PINN to solve the one-dimensional diffusion equation. The PINN takes the initial and boundary conditions as input and predicts the concentration of the diffusing substance at any point in space and time. This project aims to solve the one-dimensional diffusion equation using Physics-Informed Neural Networks (PINNs) and visualize the results. This project also includes an exact imposition of boundary conditions with distance functions, as proposed by Srivastava and Sukumar (2022), and the advantages, limitations, and opportunities of using physics-informed neural networks for data-driven simulations are discussed. 

## Diffusion Equation

The diffusion equation is a partial differential equation that describes the diffusion of a substance in space. The one-dimensional form of the diffusion equation is given by:

$$ \frac{\partial u}{\partial t} = D \frac{\partial^2 u}{\partial x^2} $$

where $u(x, t)$ is the concentration of the diffusing substance at position $x$ and time $t$, and $D$ is the diffusion coefficient.


## Navier-Stokes Equation Solution using PINNs

The flow field around a fluid object, such as an airfoil, is an essential problem in fluid mechanics that has practical applications in various fields, including aerospace, marine, and civil engineering. Solving the Navier-Stokes equations to predict the flow field around a fluid object is a challenging task due to the complex geometry and boundary conditions involved. Recently, physics-informed neural networks (PINNs) have emerged as a promising technique to solve partial differential equations (PDEs) accurately and efficiently. This proposal aims to investigate the application of PINNs in fluid mechanics to predict the flow field around a fluid object by solving the Navier-Stokes equations.
A more detailed explanation of how PINNs can be used to solve the Navier-Stokes equations for fluid flow around an airfoil. The funndamental Navier-Stokes equations describe the motion of fluids and are given by:

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

To run the project, simply run the `main_diffussion.py` file. The output plots will be saved in the `plots` directory. You can also read report and proposal to get a better understanding of what I'm trying to do. 

## References:
[1] Srivastava, A., & Sukumar, N. (2022). Exact imposition of boundary conditions with distance functions in physics-informed deep neural networks. Computer Methods in Applied Mechanics and Engineering, 389, 114333. doi: 10.1016/j.cma.2021.114333. https://www.researchgate.net/publication/316948785_Green%27s_Function_for_the_Heat_Equation
Retrieved from https://www.sciencedirect.com/science/article/abs/pii/S0045782521006186

[2] Willis, J. R. (1980). Polarization approach to the scattering of elastic waves-I. Scattering by a single inclusion. Journal of the Mechanics and Physics of Solids, 28(5-6), 287-305. doi: 10.1016/0022-5096(80)90021-6. https://www.researchgate.net/publication/316948785_Green%27s_Function_for_the_Heat_Equation
Retrieved from https://www.sciencedirect.com/science/article/abs/pii/0022509680900216

[3] Fernández de la Mata, F., Gijon, A., Molina-Solana, M., & Gomez-Romero, J. (2023). Physics-informed neural networks for data-driven simulation: Advantages, limitations, and opportunities. Physica A: Statistical Mechanics and its Applications, 610, 128415. doi: 10.1016/j.physa.2022.128415. https://www.researchgate.net/publication/316948785_Green%27s_Function_for_the_Heat_Equation
Retrieved from https://www.sciencedirect.com/science/article/pii/S0378437122009736

[4] Ursell, T. (2005). APh Physics Laboratory. Physics. Retrieved from http://www.physics.nyu.edu/grierlab/methods/node11.html.

[5] Tang, J., Azevedo, V. C., Cordonnier, G., & Solenthaler, B. (2022). Neural Green's function for Laplacian systems. Computers Graphics, 107, 186-196. doi: https://www.researchgate.net/publication/316948785_Green%27s_Function_for_the_Heat_Equation
Retrieved from https://doi.org/10.1016/j.cag.2022.07.016.

[6] Eyob, G. (2023). Full 2D-Diffusion Equation Derivation. Retrieved from https://github.com/eyobghiday/PINNs-in-Fluid-Mechanics/blob/main/Eyob_Ghiday_Difussion_Derivation.pdf.

[7] Eyob, G. (2023). Application of PiNNs in 2D Diffusion Equation. Retrieved from https://github.com/eyobghiday/PINNs-in-Fluid-Mechanics.

[8] Arocha, M. A. (2018). Crank-Nicolson Method. Retrieved from https://matlabgeeks.weebly.com/uploads/8/0/4/8/8048228/crank_nicolson_method_presentation-v5.pdf.

[9] Hassan, A. A. (2017). Green's Function for the Heat Equation. Fluid Mechanics: Open Access, 04(02), 2-7. doi: 10.4172/2476-2296.1000152. https://www.researchgate.net/publication/316948785_Green%27s_Function_for_the_Heat_Equation
Retrieved from https://www.researchgate.net/publication/316948785_Green%27s_Function_for_the_Heat_Equation

[10] Skinner, D. (n.d.). Green's functions for PDEs. Retrieved from http://www.damtp.cam.ac.uk/user/dbs26/1
Raissi, M., Perdikaris, P., & Karniadakis, G. E. (2019). Physics-informed neural networks: A deep learning framework for solving forward and inverse problems involving nonlinear partial differential equations. Journal of Computational Physics.

### Questions?
The codes I wrote here are purely for eucational purposes, so please take caution when you refer to them. For questions, comments, edit and changes, please contact me. Or tag my username <b> @eyobghiday </b> in the code.
