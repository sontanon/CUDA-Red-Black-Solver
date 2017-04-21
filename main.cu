/******************************************************************
 ***                           HEADERS                          ***
 ******************************************************************/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

// Inline error checking.
#include "cudaErrorCheck.h"

// Multiple I/O tools.
#include "tools.h"

// Main solver.
#include "cudaSOR.h"

#include "textures.h"

// Macros.
#define MAX(X, Y) ((X) > (Y)) ? (X) : (Y)
#define MIN(X, Y) ((X) < (Y)) ? (X) : (Y)

//#define SOR

/********************************************************************
 ***                             MAIN                             ***
 ********************************************************************/

int main(int argc, char *argv[])
{
	int fileio = (argc > 1) ? atoi(argv[1]) : 0;

	// Grid parameters.
	int NTotal = (argc > 2) ? atoi(argv[2]) + 1 : 257;
	int NrTotal = NTotal;
	int NzTotal = NTotal;
	int NrInterior = NrTotal - 2;
	int NzInterior = NzTotal - 2;
	double h = 10.0/((double)NTotal - 1.0);

	// Brill parameters.
	const double a0 = 1.0;
	const double rs02 = 1.0;
	const double zs02 = 1.0;

    	printf("MAIN: Parameters are NrTotal = %d, NzTotal = %d, h = %3.6E.\n", NrTotal, NzTotal, h);

	// Host arrays.
    	double *h_u = NULL;
    	double *h_res = NULL;
    	double *h_r = NULL;
    	double *h_z = NULL;
    	double *h_rr = NULL;
    	double *h_s = NULL;

    	// Allocate arrays.
   	h_u = new double[NzTotal * NrTotal]();
   	assert(NULL != h_u);

   	h_res = new double[NzTotal * NrTotal]();
   	assert(NULL != h_res);

   	h_r = new double[NzTotal * NrTotal]();
   	assert(NULL != h_r);

   	h_z = new double[NzTotal * NrTotal]();
   	assert(NULL != h_z);

   	h_rr = new double[NzTotal * NrTotal]();
   	assert(NULL != h_rr);

   	h_s = new double[NzTotal * NrTotal]();
   	assert(NULL != h_s);

    	printf("MAIN: Allocated host memory.\n");

    	// Initialize grids.
    	double  temp_r = 0.0,
            	temp_z = 0.0,
            	temp_rr = 0.0,
            	temp_q = 0.0,
            	temp_s = 0.0;

	// Fill loop.
    	for (int i = 0; i < NrTotal; i++) {
		// r value.
		temp_r = ((double)i - 0.5) * h;
        	for (int j = 0; j < NzTotal; j++) {
	    		// z value.
	    		temp_z = ((double)j - 0.5) * h;

            		// Set initial solution to one.
            		h_u[IDX(i,j,NrTotal,NzTotal)] = 1.0;

	    		// Set coordinate grids.
            		h_r[IDX(i,j,NrTotal,NzTotal)] = temp_r;
            		h_z[IDX(i,j,NrTotal,NzTotal)] = temp_z;

	    		// Radial cooridnate.
            		temp_rr = sqrt(temp_r * temp_r + temp_z * temp_z);
            		h_rr[IDX(i,j,NrTotal,NzTotal)] = temp_rr;

            		// Not actually q. Missing r**2 factor.
            		temp_q = a0 * exp(-((temp_r * temp_r)/rs02 
				 + (temp_z * temp_z)/zs02)); 
			
	    		// Linear term is correct, however.
            		temp_s = 0.5 * (1.0 + temp_r * temp_r * (-5.0/rs02 
                                - 1.0/zs02 + 2.0 * temp_z * temp_z/(zs02 * zs02)
                                + 2.0 * temp_r * temp_r/(rs02 * rs02))) * temp_q;

            		h_s[IDX(i,j,NrTotal,NzTotal)] = temp_s;
        	}
    	}

    	printf("MAIN: Initialized grids and solution.\n");

	if (fileio) {
    	/*** FILE I/O ***/
    	FILE *f_r = fopen("r.tsv", "w");
    	FILE *f_z = fopen("z.tsv", "w");
    	FILE *f_rr = fopen("rr.tsv", "w");
    	FILE *f_s = fopen("s.tsv", "w");

    	printf("MAIN: Writing grids and source to files...\n");

    	for (int i = 0; i < NrTotal; i++) {
		for (int j = 0; j < NzTotal; j++) {
		    	fprintf(f_r, (j < NzTotal - 1) ? "%8.16E\t" : "%8.16E\n", h_r[IDX(i,j,NrTotal,NzTotal)]);
		    	fprintf(f_z, (j < NzTotal - 1) ? "%8.16E\t" : "%8.16E\n", h_z[IDX(i,j,NrTotal,NzTotal)]);
		    	fprintf(f_rr, (j < NzTotal - 1) ? "%8.16E\t" : "%8.16E\n", h_rr[IDX(i,j,NrTotal,NzTotal)]);
		    	fprintf(f_s, (j < NzTotal - 1) ? "%8.16E\t" : "%8.16E\n", h_s[IDX(i,j,NrTotal,NzTotal)]);
	    	}
    	}

    	// Close files.
    	fclose(f_r);
    	fclose(f_z);
    	fclose(f_rr);
    	fclose(f_s);

    	printf("MAIN: Finished writing to files.\n");
    	/*** FILE I/O ***/
	}

    	// Device memory.
    	double *d_u = NULL;
    	double *d_res = NULL;
    	double *d_r = NULL;
    	double *d_s = NULL;

    	size_t pitch = 0;
	size_t pitch2 = 0;

	// Allocate memory.
    	cudaSafeCall(cudaMallocPitch((void **)&d_u, &pitch, sizeof(double) * NzTotal, NrTotal));
    	cudaSafeCall(cudaMallocPitch((void **)&d_res, &pitch2, sizeof(double) * NzTotal, NrTotal));
	assert(pitch == pitch2);
    	cudaSafeCall(cudaMallocPitch((void **)&d_r, &pitch, sizeof(double) * NzTotal, NrTotal));
	assert(pitch == pitch2);
    	cudaSafeCall(cudaMallocPitch((void **)&d_s, &pitch2, sizeof(double) * NzTotal, NrTotal));
	assert(pitch == pitch2);
    	printf("MAIN: Allocated device memory. Pitch is %zd elements.\n", pitch/sizeof(double));

    	// Copy to device.
    	cudaSafeCall(cudaMemcpy2D(d_u, pitch, h_u, NzTotal * sizeof(double), NzTotal * sizeof(double), NrTotal, cudaMemcpyHostToDevice));
    	cudaSafeCall(cudaMemcpy2D(d_res, pitch, h_res, NzTotal * sizeof(double), NzTotal * sizeof(double), NrTotal, cudaMemcpyHostToDevice));
    	cudaSafeCall(cudaMemcpy2D(d_r, pitch, h_r, NzTotal * sizeof(double), NzTotal * sizeof(double),NrTotal, cudaMemcpyHostToDevice));
    	cudaSafeCall(cudaMemcpy2D(d_s, pitch, h_s, NzTotal * sizeof(double), NzTotal * sizeof(double), NrTotal, cudaMemcpyHostToDevice));
    	printf("MAIN: Copied memory to device.\n");

    	// Initialize and bind textures.
    	cudaChannelFormatDesc channelDesc = cudaCreateChannelDesc<int2>();
    	printf("MAIN: Created channel description.\n");

    	cudaSafeCall(cudaBindTexture2D(NULL, tex_u, d_u, channelDesc, NzTotal, NrTotal, pitch));
    	printf("MAIN: Bound texture.\n");

    	// Main algorithm.
    	double infnorm = 1.0, twonorm = 1.0;
	int maxiter = (argc > 3) ? atoi(argv[3]) : 1000;
	int verboseCounter = (argc > 4) ? atoi(argv[4]) : 100;
	double tol = h * h;

	printf("MAIN: Solver parameters are maxiter = %d, verboseCounter = %d.\n", maxiter, verboseCounter);

	// Subroutine call.
	cudaSOR(&infnorm, &twonorm, d_u, d_res, d_s, pitch, h, NrInterior, NzInterior, maxiter, tol, verboseCounter);

	printf("MAIN: Solver done!\n");

	cudaSafeCall(cudaMemcpy2D(h_u, NzTotal * sizeof(double), d_u, pitch, NzTotal * sizeof(double), NrTotal, cudaMemcpyDeviceToHost));
    	cudaSafeCall(cudaMemcpy2D(h_res, NzTotal * sizeof(double), d_res, pitch, NzTotal * sizeof(double), NrTotal, cudaMemcpyDeviceToHost));
    	printf("MAIN: Copied solution and residual back to host.\n");

	if (fileio) {
    	/*** FILE I/O ***/
	FILE *f_u = fopen("u.tsv", "w");
    	FILE *f_res = fopen("res.tsv", "w");

    	printf("MAIN: Writing solution and residual to files...\n");

    	for (int i = 0; i < NrTotal; i++) { 
	    	for (int j = 0; j < NzTotal; j++) {
		    	fprintf(f_u, (j < NzTotal - 1) ? "%8.16E\t" : "%8.16E\n", h_u[IDX(i,j,NrTotal,NzTotal)]);
		    	fprintf(f_res, (j < NzTotal - 1) ? "%8.16E\t" : "%8.16E\n", h_res[IDX(i,j,NrTotal,NzTotal)]);
		}
	}

    	// Close files.
	fclose(f_u);
    	fclose(f_res);

    	printf("MAIN: Finished writing to files.\n");
    	/*** FILE I/O ***/
	}

    	// Unbind textures.
    	cudaSafeCall(cudaUnbindTexture(tex_u));
    	printf("MAIN: Unbound textures.\n");

    	// Deallocate memory.
    	if (d_u) {
        	cudaSafeCall(cudaFree(d_u));
        	d_u = NULL;
    	}
    	if (d_res) {
        	cudaSafeCall(cudaFree(d_res));
        	d_res = NULL;
    	}
    	if (d_r) {
        	cudaSafeCall(cudaFree(d_r));
        	d_r = NULL;
    	}
    	if (d_s) {
        	cudaSafeCall(cudaFree(d_s));
        	d_s = NULL;
    	}
    	printf("MAIN: Cleared device memory.\n");

    	if (h_u) {
        	delete[] h_u;
        	h_u = NULL;
    	}
    	if (h_res) {
        	delete[] h_res;
        	h_res = NULL;
    	}
    	if (h_r) {
        	delete[] h_r;
        	h_r = NULL;
    	}
    	if (h_z) {
        	delete[] h_z;
        	h_z= NULL;
    	}
    	if (h_rr) {
        	delete[] h_rr;
        	h_rr = NULL;
    	}
    	if (h_s) {
        	delete[] h_s;
        	h_s = NULL;
    	}
    	printf("MAIN: Cleared host memory.\n");
    	printf("MAIN: All done!\n");

    	return 0;
}
