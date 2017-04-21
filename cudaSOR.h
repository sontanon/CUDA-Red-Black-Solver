// Main Red-Black Solver and kernels.
void cudaSOR(double *infnorm, double *twonorm,
		double *d_u, double *d_res, const double *d_s, const size_t pitch,
		const double h, const int NrInterior, const int NzInterior, 
		const int maxiter, const double tol, const int verboseCounter);
