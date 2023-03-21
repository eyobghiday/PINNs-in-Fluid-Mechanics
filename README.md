This is project is developed for one of my grad classes at Illinois Tech. It's an ongoing research project, henceforth the codes and notes are part of an ongoing process.

# PINNs-in-Fluid-Mechanics

The flow field around a fluid object, such as an airfoil, is an essential problem in fluid mechanics that has practical applications in various fields, including aerospace, marine, and civil engineering. Solving the Navier-Stokes equations to predict the flow field around a fluid object is a challenging task due to the complex geometry and boundary conditions involved. Recently, physics-informed neural networks (PINNs) have emerged as a promising technique to solve partial differential equations (PDEs) accurately and efficiently. This proposal aims to investigate the application of PINNs in fluid mechanics to predict the flow field around a fluid object by solving the Navier-Stokes equations.

A more detailed explanation of how PINNs can be used to solve the Navier-Stokes equations for fluid flow around an airfoil. The funndamental Navier-Stokes equations describe the motion of fluids and are given by:

$$\rho\left(\frac{\partial\textbf{v}}{\partial t}+\textbf{v}\cdot\nabla\textbf{v}\right)=-\nabla p+\mu\nabla^2\textbf{v}+\textbf{f}$$
$$\nabla\cdot\textbf{v}=0$$

where $\rho$ is the fluid density, $\textbf{v}$ is the velocity field, $p$ is the pressure field, $\mu$ is the fluid viscosity, and $\textbf{f}$ is the external force field.

## Objective

The main objective of this proposal is to develop a physics-informed neural network model to predict the flow field around a fluid object by solving the Navier-Stokes equations. The specific objectives follows by constructing a dataset of Navier-Stokes solutions for the flow field around a fluid object using a numerical method, such as finite volume or finite element methods. The dataset will then be used to develop the PINN model to predict the flow field around a fluid object by incorporating the Navier-Stokes equations as a constraint. Finally, the PINN model will be trained using the dataset of Navier-Stokes solutions and evaluate its performance in predicting the flow field around a fluid object.
