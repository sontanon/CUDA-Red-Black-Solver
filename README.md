# CUDA-Red-Black-Solver
This code solves Brill wave initial data for 3 + 1 Numerical Relativity simulations using a simple Red-Black Gauss-Seidel. Specifically, we work in an axisymmetric space and solve for the conformal factor.

The code has been fully paralelized in CUDA and I have tested it succesfuly for up to grids of 4096 x 4096 points. As a quick benchmark, a 1024 x 1024 grid running on a FORTRAN fully-vectorized code will execute to convergence in 237.2 seconds on an Intel i5-4690K, whereas on this code, it will converge in 15.8 seconds using a GTX 970 (and the same processor).
