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

// Macros.
#define MAX(X, Y) ((X) > (Y)) ? (X) : (Y)
#define MIN(X, Y) ((X) < (Y)) ? (X) : (Y)

//#define SOR

/********************************************************************
 ***                      CUDA ERROR CHECKING                     ***
 ********************************************************************/

#define CUDA_ERROR_CHECK

#define cudaSafeCall( err ) __cudaSafeCall( err, __FILE__, __LINE__ )
#define cudaCheckError() __cudaCheckError( __FILE__, __LINE__ )

inline void __cudaSafeCall(cudaError_t err, const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s: %i: %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}

inline void __cudaCheckError(const char *file, const int line)
{
#ifdef CUDA_ERROR_CHECK
	cudaError_t err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s: %i: %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}

	// Sync devices and check again.
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() wiith sync failed at %s: %i: %s.\n", file, line, cudaGetErrorString(err));
		exit(-1);
	}
#endif
	return;
}



/********************************************************************
 ***                      FORWARD DECLARATIONS                    ***
 ********************************************************************/

// Indexing macro/function.
int IDX(const int r, const int z, const int NrTotal, const int NzTotal);

// Write to 2D CUDA array.
__device__ void write2D(double *d_u, const int r, const int z,
	       		const size_t pitch, const double value);

// Read from 2D CUDA array.
__device__ double read2D(const double *d_u, const int r, const int z,
	       		 const size_t pitch);

// Fetch double texture values.
__device__ double doubleTex2D(const texture<int2,cudaTextureType2D,cudaReadModeElementType>tex,
			      const int r, const int z);

// Main Red-Black Solver and kernels.
void cudaSOR(double *infnorm, double *twonorm,
		double *d_u, double *d_res, const double *d_s, const size_t pitch,
		const double h, const int NrInterior, const int NzInterior, 
		const int maxiter, const double tol, const int verboseCounter);

__global__ void cudaRed(double *d_u, double *d_res, const double *d_s, const size_t pitch,
			double *infnorm_array, double *twonorm2_array,
			const double h, const int NrInterior, const int NzInterior,
			const double omega);

__global__ void cudaBlack(double *d_u, double *d_res, const double *d_s, const size_t pitch,
			double *infnorm_array, double *twonorm2_array,
			const double h, const int NrInterior, const int NzInterior,
			const double omega);

// Boundary update routine.
void boundary_update(double *d_u, const size_t pitch, 
		     const int NrInterior, const int NzInterior, 
		     const double h, const double u_inf);

__global__ void boundary_kernel(double *d_u, const size_t pitch, 
				const int NrInterior, const int NzInterior,
			  	const double h, const double u_inf);

/******************************************************************
 ***                          TEXTURES                          ***
 ******************************************************************/

// Textures.
texture<int2, cudaTextureType2D, cudaReadModeElementType> tex_u;


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




/********************************************************************
 ***                          OTHER TOOLS                         ***
 ********************************************************************/
 // Indexing macro.
int IDX(const int r, const int z, const int NrTotal, const int NzTotal) {
    assert(r < NrTotal);
    assert(z < NzTotal);
    return r * NzTotal + z;
}

// Write to a 2D CUDA array.
__device__ void write2D(double *d_u, const int r, const int z, 
			const size_t pitch, const double u)
{
	((double *)((char *)d_u + r * pitch))[z] = u;
}

// Read from a 2D CUDA array.
__device__ double read2D(const double *d_u, const int r, const int z, const size_t pitch)
{
	return ((double *)((char *)d_u + r * pitch))[z];
}

// Read a double texture.
__device__ double doubleTex2D(const texture<int2, cudaTextureType2D, cudaReadModeElementType> tex_Ref,
			      const int r, const int z)
{
	int2 temp = tex2D(tex_Ref, z, r);
	return __hiloint2double(temp.y, temp.x);
}





void cudaSOR(double *infnorm, double *twonorm,
		double *d_u, double *d_res, const double *d_s, const size_t pitch,
		const double h, const int NrInterior, const int NzInterior, 
		const int maxiter, const double tol, const int verboseCounter)
{
	// We form thread-blocks of 16 x 16 = 256 threads.
	const int threadDim = 16;
	//const int numThreads = 256;

	// Since we need four colors for each subgrid, the number of subgrids in the block
	// will be numThreads/4 = 64.
	//const int numSubgridsPerBlock = 64;

	// Each subgrid is 8 by 8 elements in size.
	//const int subgridDim = 8;

	// The block dimension will therefore be the square root of numSubgridsPerBlock (to
	// divide them evenly between r and z) multiplied by the subgrid dimension subgridDim.
	// In this case, sqrt(numSubgridsPerBlock) = 8 and subgridDim = 8, thus...
	//const int blockDimension = 64;

	// The main point is that each thread-block will extend 64 squares into r and z in
	// this manner and thus we can divide our complete domain as:
	int rBlocks = (NrInterior+1)/64;
	int zBlocks = (NzInterior+1)/64;
	int numBlocks = rBlocks * zBlocks;
	// Recall that NrInterior and NzInterior are both of the form 2**n - 1. We require
	// that our grid size is at least 63 by 63 interior points.

	// With these constants we can form the appropriate thread and grid-blocks.
	dim3 threadBlock(threadDim, threadDim);
	dim3 gridBlock(rBlocks, zBlocks);

	// Allocate memory for the buffers for the norms reduction on host.
	// Since each threadBlock will write one value, we need numBlocks amount of double
	// memory. We need both host and device memory.
	double *h_redInfnorm 	= (double *)malloc(sizeof(double) * numBlocks);
	assert(NULL != h_redInfnorm);

	double *h_blackInfnorm 	= (double *)malloc(sizeof(double) * numBlocks);
	assert(NULL != h_blackInfnorm);

	double *h_redTwonorm2 	= (double *)malloc(sizeof(double) * numBlocks);
	assert(NULL != h_redTwonorm2);

	double *h_blackTwonorm2 = (double *)malloc(sizeof(double) * numBlocks);
	assert(NULL != h_blackTwonorm2);

	double *d_redInfnorm, *d_redTwonorm2, *d_blackInfnorm, *d_blackTwonorm2;
	cudaSafeCall(cudaMalloc((void **)&d_redInfnorm,    sizeof(double) * numBlocks));
	cudaSafeCall(cudaMalloc((void **)&d_redTwonorm2,   sizeof(double) * numBlocks));
	cudaSafeCall(cudaMalloc((void **)&d_blackInfnorm,  sizeof(double) * numBlocks));
	cudaSafeCall(cudaMalloc((void **)&d_blackTwonorm2, sizeof(double) * numBlocks));

	// Convergence flag.
	int convergenceFlag = 0;

	// Auxiliary variables.
	double redInfnorm, blackInfnorm, redTwonorm2, blackTwonorm2;

#ifdef SOR
	const double pi = 4.0 * atan(1.0);
	double rjac = 0.5 * (cos(pi/(double)NrInterior) + cos(pi/(double)NzInterior));
#endif
	double omega = 1.0;

	// The main SOR loop begins here and does up to maxiter iterations.
	clock_t start, stop;
       	start = clock();
	for (int k = 1; k < maxiter+1; k++) {
		// Now we will launch a kernel which will SOR the red squares of the entire domain.
		// Each thread-block will have two buffers of shared memory each with up to numThreads
		// elements. Each thread will store in its shared memory bin the value of the infnorm
		// and on the other buffer the value of the twonorm2 (two norm squared since while the
		// square root has not been calculated, this quantity is additive).
		// After all threads have stored onto the buffer, a reduction is done so that we obtain
		// the infnorm and twonorm of the entire subgrid associated to a specific thread-block.
		// These values are written onto global memory as a final step of the kernel.
		cudaRed<<<gridBlock, threadBlock>>>(d_u, d_res, d_s, pitch, d_redInfnorm, d_redTwonorm2, h, NrInterior, NzInterior, omega);

		// After running the red SOR, we will SOR the black squares. A synchronization step is
		// required so that no block starts SOR before all red blocks have finished. This is
		// done by calling cudaDeviceSynchronize() in the inline function cudaCheckError().
		cudaCheckError();

#ifdef SOR
		// Recalculate omega.
		omega (k != 1) ? 1.0/(1.0 - 0.25 * rjac * rjac * omega) : 1.0/(1.0 - 0.5 * rjac * rjac);
#endif

		// Now that we have synced, we do SOR on the black squares in analogous manner. This
		// kernel will also write onto global memory the norms of each thread-block.
		cudaBlack<<<gridBlock, threadBlock>>>(d_u, d_res, d_s, pitch, d_blackInfnorm, d_blackTwonorm2, h, NrInterior, NzInterior, omega);

		// Obviously, after the kernel has executed, a sync is also necessary.
		cudaCheckError();
		
#ifdef SOR
		// Recalculate omega.
		omega = 1.0/(1.0 - 0.25 * rjac * rjac * omega);
#endif

		// Now we will fix the boundary.
		boundary_update(d_u, pitch, NrInterior, NzInterior, h, 1.0);

		// What remains is to reduce to a single value the norms. We have to put together the
		// red and black norms into one final value. This reduction is done on the host.
		cudaSafeCall(cudaMemcpy(h_redInfnorm,    d_redInfnorm,    sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));
		cudaSafeCall(cudaMemcpy(h_blackInfnorm,  d_blackInfnorm,  sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));
		cudaSafeCall(cudaMemcpy(h_redTwonorm2,   d_redTwonorm2,   sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));
		cudaSafeCall(cudaMemcpy(h_blackTwonorm2, d_blackTwonorm2, sizeof(double) * numBlocks, cudaMemcpyDeviceToHost));

		redInfnorm = h_redInfnorm[0];
		blackInfnorm = h_blackInfnorm[0];
		redTwonorm2 = h_redTwonorm2[0];
		blackTwonorm2 = h_blackTwonorm2[0];

		for (int l = 1; l < numBlocks; l++) {
			redInfnorm = MAX(redInfnorm, h_redInfnorm[l]);
			blackInfnorm = MAX(blackInfnorm, h_blackInfnorm[l]);
			redTwonorm2 += h_redTwonorm2[l];
			blackTwonorm2 += h_blackTwonorm2[l];
		}
		// Put values at pointers.
		*infnorm = MAX(redInfnorm, blackInfnorm);
		*twonorm = sqrt(redTwonorm2 + blackTwonorm2);


		// It is a good custom to display the initial infnorm and two norm. This obviously is
		// only the case if the iteration k = 1.
		if (k == 1) {
			// Print the initial norms.
			printf("RED-BLACK SOLVER: Initial infnorm = %3.6E and twonorm = %3.6E.\n", *infnorm, *twonorm);
		}

		// Now that we have a reduced norm, we can assert if our tolerance has been met, and if
		// so we will exit the loop.
		if (*infnorm < tol) {
			// Print that convergence has been achieved at iteration number.
			printf("RED-BLACK SOLVER: Convergece has been achieved at iteration k = %d.\n", k);
			convergenceFlag = 1;
			break;
		}

		// If we have not converged, we will print out to the screen the current progress every
		// verboseCounter iterations.
		if (k % verboseCounter == 0) {
			// Print current progress.
			printf("RED-BLACK SOLVER: Iteration = %d, infnorm = %3.6E, twonorm = %3.6E, omega = %3.6E.\n", k, *infnorm, *twonorm, omega);
		}
	}
	stop = clock();
	// Execution time. 
	printf("RED-BLACK SOLVER: Loop execution time was %lf seconds.\n", (stop-start)/(double)CLOCKS_PER_SEC);

	// Main loop has finished. Check if converged.
	if (!convergenceFlag) {
		printf("RED-BLACK SOLVER: Failed to converge to tol = %3.6E after maxiter = %d iterations.\n", tol, maxiter);
	}
	printf("RED-BLACK SOLVER: Final infnorm = %3.6E and twonorm = %3.6E.\n", *infnorm, *twonorm);

	// This concludes the subroutine. What only remains is to clear memory.
	cudaSafeCall(cudaFree(d_redInfnorm));
	cudaSafeCall(cudaFree(d_redTwonorm2));
	cudaSafeCall(cudaFree(d_blackInfnorm));
	cudaSafeCall(cudaFree(d_blackTwonorm2));

	free(h_redInfnorm);
	free(h_redTwonorm2);
	free(h_blackInfnorm);
	free(h_blackTwonorm2);

	// All done.
	printf("RED-BLACK SOLVER: All done!\n");
       	return;	
}


__global__ void cudaRed(double *d_u, double *d_res, const double *d_s, const size_t pitch,
			double *infnorm_array, double *twonorm2_array,
			const double h, const int NrInterior, const int NzInterior, 
			const double omega)
{
	// The first thing we need to do is to find out in which block we are.
	int bidr = blockIdx.x;
	int bidz = blockIdx.y;

	// Block dimension is 64 in each direction.
	int block_r_base = 1 + bidr * 64;
	int block_z_base = 1 + bidz * 64;

	// Now we need to find in which subgrid we are executing.
	int tidr = threadIdx.x;
	int tidz = threadIdx.y;

	// BlockDim.y is 16.
	int threadOffset = tidr * 16 + tidz;
	int blockOffset = bidr * gridDim.y + bidz;
	int r_bin = tidr/2;
	int z_bin = tidz/2;

	int color;
	// Determine color.
	if (tidr % 2 == 0) {
		if (tidz % 2 == 0) {
			color = 0;
		} else {
			color = 1;
		}
	} else {
		if (tidz % 2 == 0) {
			color = 2;
		} else {
			color = 3;
		}
	}

	// Since every four threads execute on the same grid.
	int subgrid_r_base = block_r_base + r_bin * 8;
	int subgrid_z_base = block_z_base + z_bin * 8;

	// Shared memory buffer.
	__shared__ double buffer[512];
	double *block_infnorm = buffer;
	double *block_twonorm2 = (double *)&buffer[256];

	// Integer coordinates.
	int r_coord0, r_coord1, r_coord2, r_coord3, r_coord4, r_coord5, r_coord6, r_coord7; 
	int z_coord0, z_coord1, z_coord2, z_coord3, z_coord4, z_coord5, z_coord6, z_coord7; 

	// Now we start our update. It differs according to the color.
	switch (color) {
		case 0:
			// This is the frist color that is clamped to the base.
			// The eight coordinates are:
			r_coord0 = subgrid_r_base;
			z_coord0 = subgrid_z_base;
			r_coord1 = subgrid_r_base;
			z_coord1 = subgrid_z_base + 4;		
			r_coord2 = subgrid_r_base + 2;
			z_coord2 = subgrid_z_base + 2;
			r_coord3 = subgrid_r_base + 2;
			z_coord3 = subgrid_z_base + 6;
			r_coord4 = subgrid_r_base + 4;
			z_coord4 = subgrid_z_base;
			r_coord5 = subgrid_r_base + 4;
			z_coord5 = subgrid_z_base + 4;
			r_coord6 = subgrid_r_base + 6;
			z_coord6 = subgrid_z_base + 2;
			r_coord7 = subgrid_r_base + 6;
			z_coord7 = subgrid_z_base + 6;
			break;

		case 1:
			r_coord0 = subgrid_r_base + 1;
			z_coord0 = subgrid_z_base + 1;
			r_coord1 = subgrid_r_base + 1;
			z_coord1 = subgrid_z_base + 5;		
			r_coord2 = subgrid_r_base + 3;
			z_coord2 = subgrid_z_base + 3;
			r_coord3 = subgrid_r_base + 3;
			z_coord3 = subgrid_z_base + 7;
			r_coord4 = subgrid_r_base + 5;
			z_coord4 = subgrid_z_base + 1;
			r_coord5 = subgrid_r_base + 5;
			z_coord5 = subgrid_z_base + 5;
			r_coord6 = subgrid_r_base + 7;
			z_coord6 = subgrid_z_base + 3;
			r_coord7 = subgrid_r_base + 7;
			z_coord7 = subgrid_z_base + 7;
			break;

		case 2:
			r_coord0 = subgrid_r_base;
			z_coord0 = subgrid_z_base + 2;
			r_coord1 = subgrid_r_base;
			z_coord1 = subgrid_z_base + 6;		
			r_coord2 = subgrid_r_base + 2;
			z_coord2 = subgrid_z_base;
			r_coord3 = subgrid_r_base + 2;
			z_coord3 = subgrid_z_base + 4;
			r_coord4 = subgrid_r_base + 4;
			z_coord4 = subgrid_z_base + 2;
			r_coord5 = subgrid_r_base + 4;
			z_coord5 = subgrid_z_base + 6;
			r_coord6 = subgrid_r_base + 6;
			z_coord6 = subgrid_z_base;
			r_coord7 = subgrid_r_base + 6;
			z_coord7 = subgrid_z_base + 4;
			break;

		case 3:
			r_coord0 = subgrid_r_base + 1;
			z_coord0 = subgrid_z_base + 3;
			r_coord1 = subgrid_r_base + 1;
			z_coord1 = subgrid_z_base + 7;		
			r_coord2 = subgrid_r_base + 3;
			z_coord2 = subgrid_z_base + 1;
			r_coord3 = subgrid_r_base + 3;
			z_coord3 = subgrid_z_base + 5;
			r_coord4 = subgrid_r_base + 5;
			z_coord4 = subgrid_z_base + 3;
			r_coord5 = subgrid_r_base + 5;
			z_coord5 = subgrid_z_base + 7;
			r_coord6 = subgrid_r_base + 7;
			z_coord6 = subgrid_z_base + 1;
			r_coord7 = subgrid_r_base + 7;
			z_coord7 = subgrid_z_base + 5;
			break;
	}

	// Now fetch values.
	double u0, u0_rplus, u0_rminus, u0_zplus, u0_zminus;
	double u1, u1_rplus, u1_rminus, u1_zplus, u1_zminus;
	double u2, u2_rplus, u2_rminus, u2_zplus, u2_zminus;
	double u3, u3_rplus, u3_rminus, u3_zplus, u3_zminus;
	double u4, u4_rplus, u4_rminus, u4_zplus, u4_zminus;
	double u5, u5_rplus, u5_rminus, u5_zplus, u5_zminus;
	double u6, u6_rplus, u6_rminus, u6_zplus, u6_zminus;
	double u7, u7_rplus, u7_rminus, u7_zplus, u7_zminus;

	u0 = doubleTex2D(tex_u, r_coord0, z_coord0);
	u0_rplus = doubleTex2D(tex_u, r_coord0 + 1, z_coord0);
	u0_rminus = doubleTex2D(tex_u, r_coord0 - 1, z_coord0);
	u0_zplus = doubleTex2D(tex_u, r_coord0, z_coord0 + 1);
	u0_zminus = doubleTex2D(tex_u, r_coord0, z_coord0 - 1);
	
	u1 = doubleTex2D(tex_u, r_coord1, z_coord1);
	u1_rplus = doubleTex2D(tex_u, r_coord1 + 1, z_coord1);
	u1_rminus = doubleTex2D(tex_u, r_coord1 - 1, z_coord1);
	u1_zplus = doubleTex2D(tex_u, r_coord1, z_coord1 + 1);
	u1_zminus = doubleTex2D(tex_u, r_coord1, z_coord1 - 1);
	
	u2 = doubleTex2D(tex_u, r_coord2, z_coord2);
	u2_rplus = doubleTex2D(tex_u, r_coord2 + 1, z_coord2);
	u2_rminus = doubleTex2D(tex_u, r_coord2 - 1, z_coord2);
	u2_zplus = doubleTex2D(tex_u, r_coord2, z_coord2 + 1);
	u2_zminus = doubleTex2D(tex_u, r_coord2, z_coord2 - 1);
	
	u3 = doubleTex2D(tex_u, r_coord3, z_coord3);
	u3_rplus = doubleTex2D(tex_u, r_coord3 + 1, z_coord3);
	u3_rminus = doubleTex2D(tex_u, r_coord3 - 1, z_coord3);
	u3_zplus = doubleTex2D(tex_u, r_coord3, z_coord3 + 1);
	u3_zminus = doubleTex2D(tex_u, r_coord3, z_coord3 - 1);

	u4 = doubleTex2D(tex_u, r_coord4, z_coord4);
	u4_rplus = doubleTex2D(tex_u, r_coord4 + 1, z_coord4);
	u4_rminus = doubleTex2D(tex_u, r_coord4 - 1, z_coord4);
	u4_zplus = doubleTex2D(tex_u, r_coord4, z_coord4 + 1);
	u4_zminus = doubleTex2D(tex_u, r_coord4, z_coord4 - 1);

	u5 = doubleTex2D(tex_u, r_coord5, z_coord5);
	u5_rplus = doubleTex2D(tex_u, r_coord5 + 1, z_coord5);
	u5_rminus = doubleTex2D(tex_u, r_coord5 - 1, z_coord5);
	u5_zplus = doubleTex2D(tex_u, r_coord5, z_coord5 + 1);
	u5_zminus = doubleTex2D(tex_u, r_coord5, z_coord5 - 1);
	
	u6 = doubleTex2D(tex_u, r_coord6, z_coord6);
	u6_rplus = doubleTex2D(tex_u, r_coord6 + 1, z_coord6);
	u6_rminus = doubleTex2D(tex_u, r_coord6 - 1, z_coord6);
	u6_zplus = doubleTex2D(tex_u, r_coord6, z_coord6 + 1);
	u6_zminus = doubleTex2D(tex_u, r_coord6, z_coord6 - 1);
		
	u7 = doubleTex2D(tex_u, r_coord7, z_coord7);
	u7_rplus = doubleTex2D(tex_u, r_coord7 + 1, z_coord7);
	u7_rminus = doubleTex2D(tex_u, r_coord7 - 1, z_coord7);
	u7_zplus = doubleTex2D(tex_u, r_coord7, z_coord7 + 1);
	u7_zminus = doubleTex2D(tex_u, r_coord7, z_coord7 - 1);

	// Perform calculation and update.
	// We need to fetch a value from global memory, so we need to be careful
	// not to overflow.
	// Auxiliary variables.
    	double h2 = h * h;
    	double one_over_h2 = 1.0/h2;

	double s, diag, res;
	double u0_new, u1_new, u2_new, u3_new, u4_new, u5_new, u6_new, u7_new; 
	double thread_infnorm = 0.0;
	double thread_twonorm2 = 0.0;

	if ((r_coord0 < NrInterior + 1) && (z_coord0 < NzInterior + 1)) {
		s = read2D(d_s, r_coord0, z_coord0, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u0 + (u0_zplus + u0_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord0 - 1)) * u0_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord0 - 1)) * u0_rminus);

		// Calculate update.
		u0_new = u0 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord0, z_coord0, pitch, u0_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord0, z_coord0, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord1 < NrInterior + 1) && (z_coord1 < NzInterior + 1)) {
		s = read2D(d_s, r_coord1, z_coord1, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u1 + (u1_zplus + u1_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord1 - 1)) * u1_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord1 - 1)) * u1_rminus);

		// Calculate update.
		u1_new = u1 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord1, z_coord1, pitch, u1_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord1, z_coord1, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord2 < NrInterior + 1) && (z_coord2 < NzInterior + 1)) {
		s = read2D(d_s, r_coord2, z_coord2, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u2 + (u2_zplus + u2_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord2 - 1)) * u2_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord2 - 1)) * u2_rminus);

		// Calculate update.
		u2_new = u2 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord2, z_coord2, pitch, u2_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord2, z_coord2, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord3 < NrInterior + 1) && (z_coord3 < NzInterior + 1)) {
		s = read2D(d_s, r_coord3, z_coord3, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u3 + (u3_zplus + u3_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord3 - 1)) * u3_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord3 - 1)) * u3_rminus);

		// Calculate update.
		u3_new = u3 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord3, z_coord3, pitch, u3_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord3, z_coord3, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord4 < NrInterior + 1) && (z_coord4 < NzInterior + 1)) {
		s = read2D(d_s, r_coord4, z_coord4, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u4 + (u4_zplus + u4_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord4 - 1)) * u4_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord4 - 1)) * u4_rminus);

		// Calculate update.
		u4_new = u4 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord4, z_coord4, pitch, u4_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord4, z_coord4, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord5 < NrInterior + 1) && (z_coord5 < NzInterior + 1)) {
		s = read2D(d_s, r_coord5, z_coord5, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u5 + (u5_zplus + u5_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord5 - 1)) * u5_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord5 - 1)) * u5_rminus);

		// Calculate update.
		u5_new = u5 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord5, z_coord5, pitch, u5_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord5, z_coord5, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord6 < NrInterior + 1) && (z_coord6 < NzInterior + 1)) {
		s = read2D(d_s, r_coord6, z_coord6, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u6 + (u6_zplus + u6_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord6 - 1)) * u6_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord6 - 1)) * u6_rminus);

		// Calculate update.
		u6_new = u6 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord6, z_coord6, pitch, u6_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord6, z_coord6, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord7 < NrInterior + 1) && (z_coord7 < NzInterior + 1)) {
		s = read2D(d_s, r_coord7, z_coord7, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u7 + (u7_zplus + u7_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord7 - 1)) * u7_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord7 - 1)) * u7_rminus);

		// Calculate update.
		u7_new = u7 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord7, z_coord7, pitch, u7_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord7, z_coord7, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}

	// Write to shared memory.
	block_infnorm[threadOffset] = thread_infnorm;
	block_twonorm2[threadOffset] = thread_twonorm2;

	__syncthreads();

	// Perform reductions.
	for (int i = 128; i > 0; i /= 2) {
		if (threadOffset < i) {
			block_infnorm[threadOffset] = MAX(block_infnorm[threadOffset], block_infnorm[threadOffset + i]);
			block_twonorm2[threadOffset] += block_twonorm2[threadOffset + i];
		}
		__syncthreads();
	}

	// Write to global memory.
	if (threadOffset == 0) {
		infnorm_array[blockOffset] = block_infnorm[0];
		twonorm2_array[blockOffset] = block_twonorm2[0];
	}

	// All done.
}

__global__ void cudaBlack(double *d_u, double *d_res, const double *d_s, const size_t pitch,
			double *infnorm_array, double *twonorm2_array,
			const double h, const int NrInterior, const int NzInterior,
			const double omega)
{
	// The first thing we need to do is to find out in which block we are.
	int bidr = blockIdx.x;
	int bidz = blockIdx.y;

	// Block dimension is 64 in each direction.
	int block_r_base = 1 + bidr * 64;
	int block_z_base = 1 + bidz * 64;

	// Now we need to find in which subgrid we are executing.
	int tidr = threadIdx.x;
	int tidz = threadIdx.y;

	// BlockDim.y is 16.
	int threadOffset = tidr * 16 + tidz;
	int blockOffset = bidr * gridDim.y + bidz;
	int r_bin = tidr/2;
	int z_bin = tidz/2;

	int color;
	// Determine color.
	if (tidr % 2 == 0) {
		if (tidz % 2 == 0) {
			color = 0;
		} else {
			color = 1;
		}
	} else {
		if (tidz % 2 == 0) {
			color = 2;
		} else {
			color = 3;
		}
	}

	// Since every four threads execute on the same grid.
	int subgrid_r_base = block_r_base + r_bin * 8;
	int subgrid_z_base = block_z_base + z_bin * 8;

	// Shared memory buffer.
	__shared__ double buffer[512];
	double *block_infnorm = buffer;
	double *block_twonorm2 = (double *)&buffer[256];

	// Integer coordinates.
	int r_coord0, r_coord1, r_coord2, r_coord3, r_coord4, r_coord5, r_coord6, r_coord7; 
	int z_coord0, z_coord1, z_coord2, z_coord3, z_coord4, z_coord5, z_coord6, z_coord7; 

	// Now we start our update. It differs according to the color.
	switch (color) {
		case 0:
			// This is the frist color that is clamped to the base.
			// The eight coordinates are:
			r_coord0 = subgrid_r_base + 1;
			z_coord0 = subgrid_z_base;
			r_coord1 = subgrid_r_base + 1;
			z_coord1 = subgrid_z_base + 4;		
			r_coord2 = subgrid_r_base + 3;
			z_coord2 = subgrid_z_base + 2;
			r_coord3 = subgrid_r_base + 3;
			z_coord3 = subgrid_z_base + 6;
			r_coord4 = subgrid_r_base + 5;
			z_coord4 = subgrid_z_base;
			r_coord5 = subgrid_r_base + 5;
			z_coord5 = subgrid_z_base + 4;
			r_coord6 = subgrid_r_base + 7;
			z_coord6 = subgrid_z_base + 2;
			r_coord7 = subgrid_r_base + 7;
			z_coord7 = subgrid_z_base + 6;
			break;

		case 1:
			r_coord0 = subgrid_r_base;
			z_coord0 = subgrid_z_base + 3;
			r_coord1 = subgrid_r_base;
			z_coord1 = subgrid_z_base + 7;
			r_coord2 = subgrid_r_base + 2;
			z_coord2 = subgrid_z_base + 1;
			r_coord3 = subgrid_r_base + 2;
			z_coord3 = subgrid_z_base + 5;		
			r_coord4 = subgrid_r_base + 4;
			z_coord4 = subgrid_z_base + 3;
			r_coord5 = subgrid_r_base + 4;
			z_coord5 = subgrid_z_base + 7;
			r_coord6 = subgrid_r_base + 6;
			z_coord6 = subgrid_z_base + 1;
			r_coord7 = subgrid_r_base + 6;
			z_coord7 = subgrid_z_base + 5;
			break;

		case 2:
			r_coord0 = subgrid_r_base + 1;
			z_coord0 = subgrid_z_base + 2;
			r_coord1 = subgrid_r_base + 1;
			z_coord1 = subgrid_z_base + 6;		
			r_coord2 = subgrid_r_base + 3;
			z_coord2 = subgrid_z_base;
			r_coord3 = subgrid_r_base + 3;
			z_coord3 = subgrid_z_base + 4;
			r_coord4 = subgrid_r_base + 5;
			z_coord4 = subgrid_z_base + 2;
			r_coord5 = subgrid_r_base + 5;
			z_coord5 = subgrid_z_base + 6;
			r_coord6 = subgrid_r_base + 7;
			z_coord6 = subgrid_z_base;
			r_coord7 = subgrid_r_base + 7;
			z_coord7 = subgrid_z_base + 4;
			break;

		case 3:
			r_coord0 = subgrid_r_base;
			z_coord0 = subgrid_z_base + 1;
			r_coord1 = subgrid_r_base;
			z_coord1 = subgrid_z_base + 5;
			r_coord2 = subgrid_r_base + 2;
			z_coord2 = subgrid_z_base + 3;
			r_coord3 = subgrid_r_base + 2;
			z_coord3 = subgrid_z_base + 7;		
			r_coord4 = subgrid_r_base + 4;
			z_coord4 = subgrid_z_base + 1;
			r_coord5 = subgrid_r_base + 4;
			z_coord5 = subgrid_z_base + 5;
			r_coord6 = subgrid_r_base + 6;
			z_coord6 = subgrid_z_base + 3;
			r_coord7 = subgrid_r_base + 6;
			z_coord7 = subgrid_z_base + 7;
			break;
	}

	// Now fetch values.
	double u0, u0_rplus, u0_rminus, u0_zplus, u0_zminus;
	double u1, u1_rplus, u1_rminus, u1_zplus, u1_zminus;
	double u2, u2_rplus, u2_rminus, u2_zplus, u2_zminus;
	double u3, u3_rplus, u3_rminus, u3_zplus, u3_zminus;
	double u4, u4_rplus, u4_rminus, u4_zplus, u4_zminus;
	double u5, u5_rplus, u5_rminus, u5_zplus, u5_zminus;
	double u6, u6_rplus, u6_rminus, u6_zplus, u6_zminus;
	double u7, u7_rplus, u7_rminus, u7_zplus, u7_zminus;

	u0 = doubleTex2D(tex_u, r_coord0, z_coord0);
	u0_rplus = doubleTex2D(tex_u, r_coord0 + 1, z_coord0);
	u0_rminus = doubleTex2D(tex_u, r_coord0 - 1, z_coord0);
	u0_zplus = doubleTex2D(tex_u, r_coord0, z_coord0 + 1);
	u0_zminus = doubleTex2D(tex_u, r_coord0, z_coord0 - 1);
	
	u1 = doubleTex2D(tex_u, r_coord1, z_coord1);
	u1_rplus = doubleTex2D(tex_u, r_coord1 + 1, z_coord1);
	u1_rminus = doubleTex2D(tex_u, r_coord1 - 1, z_coord1);
	u1_zplus = doubleTex2D(tex_u, r_coord1, z_coord1 + 1);
	u1_zminus = doubleTex2D(tex_u, r_coord1, z_coord1 - 1);
	
	u2 = doubleTex2D(tex_u, r_coord2, z_coord2);
	u2_rplus = doubleTex2D(tex_u, r_coord2 + 1, z_coord2);
	u2_rminus = doubleTex2D(tex_u, r_coord2 - 1, z_coord2);
	u2_zplus = doubleTex2D(tex_u, r_coord2, z_coord2 + 1);
	u2_zminus = doubleTex2D(tex_u, r_coord2, z_coord2 - 1);
	
	u3 = doubleTex2D(tex_u, r_coord3, z_coord3);
	u3_rplus = doubleTex2D(tex_u, r_coord3 + 1, z_coord3);
	u3_rminus = doubleTex2D(tex_u, r_coord3 - 1, z_coord3);
	u3_zplus = doubleTex2D(tex_u, r_coord3, z_coord3 + 1);
	u3_zminus = doubleTex2D(tex_u, r_coord3, z_coord3 - 1);

	u4 = doubleTex2D(tex_u, r_coord4, z_coord4);
	u4_rplus = doubleTex2D(tex_u, r_coord4 + 1, z_coord4);
	u4_rminus = doubleTex2D(tex_u, r_coord4 - 1, z_coord4);
	u4_zplus = doubleTex2D(tex_u, r_coord4, z_coord4 + 1);
	u4_zminus = doubleTex2D(tex_u, r_coord4, z_coord4 - 1);

	u5 = doubleTex2D(tex_u, r_coord5, z_coord5);
	u5_rplus = doubleTex2D(tex_u, r_coord5 + 1, z_coord5);
	u5_rminus = doubleTex2D(tex_u, r_coord5 - 1, z_coord5);
	u5_zplus = doubleTex2D(tex_u, r_coord5, z_coord5 + 1);
	u5_zminus = doubleTex2D(tex_u, r_coord5, z_coord5 - 1);
	
	u6 = doubleTex2D(tex_u, r_coord6, z_coord6);
	u6_rplus = doubleTex2D(tex_u, r_coord6 + 1, z_coord6);
	u6_rminus = doubleTex2D(tex_u, r_coord6 - 1, z_coord6);
	u6_zplus = doubleTex2D(tex_u, r_coord6, z_coord6 + 1);
	u6_zminus = doubleTex2D(tex_u, r_coord6, z_coord6 - 1);
		
	u7 = doubleTex2D(tex_u, r_coord7, z_coord7);
	u7_rplus = doubleTex2D(tex_u, r_coord7 + 1, z_coord7);
	u7_rminus = doubleTex2D(tex_u, r_coord7 - 1, z_coord7);
	u7_zplus = doubleTex2D(tex_u, r_coord7, z_coord7 + 1);
	u7_zminus = doubleTex2D(tex_u, r_coord7, z_coord7 - 1);

	// Perform calculation and update.
	// We need to fetch a value from global memory, so we need to be careful
	// not to overflow.
	// Auxiliary variables.
    	double h2 = h * h;
    	double one_over_h2 = 1.0/h2;

	double s, diag, res;
	double u0_new, u1_new, u2_new, u3_new, u4_new, u5_new, u6_new, u7_new; 
	double thread_infnorm = 0.0;
	double thread_twonorm2 = 0.0;

	if ((r_coord0 < NrInterior + 1) && (z_coord0 < NzInterior + 1)) {
		s = read2D(d_s, r_coord0, z_coord0, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u0 + (u0_zplus + u0_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord0 - 1)) * u0_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord0 - 1)) * u0_rminus);

		// Calculate update.
		u0_new = u0 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord0, z_coord0, pitch, u0_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord0, z_coord0, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord1 < NrInterior + 1) && (z_coord1 < NzInterior + 1)) {
		s = read2D(d_s, r_coord1, z_coord1, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u1 + (u1_zplus + u1_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord1 - 1)) * u1_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord1 - 1)) * u1_rminus);

		// Calculate update.
		u1_new = u1 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord1, z_coord1, pitch, u1_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord1, z_coord1, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord2 < NrInterior + 1) && (z_coord2 < NzInterior + 1)) {
		s = read2D(d_s, r_coord2, z_coord2, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u2 + (u2_zplus + u2_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord2 - 1)) * u2_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord2 - 1)) * u2_rminus);

		// Calculate update.
		u2_new = u2 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord2, z_coord2, pitch, u2_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord2, z_coord2, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord3 < NrInterior + 1) && (z_coord3 < NzInterior + 1)) {
		s = read2D(d_s, r_coord3, z_coord3, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u3 + (u3_zplus + u3_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord3 - 1)) * u3_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord3 - 1)) * u3_rminus);

		// Calculate update.
		u3_new = u3 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord3, z_coord3, pitch, u3_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord3, z_coord3, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord4 < NrInterior + 1) && (z_coord4 < NzInterior + 1)) {
		s = read2D(d_s, r_coord4, z_coord4, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u4 + (u4_zplus + u4_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord4 - 1)) * u4_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord4 - 1)) * u4_rminus);

		// Calculate update.
		u4_new = u4 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord4, z_coord4, pitch, u4_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord4, z_coord4, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord5 < NrInterior + 1) && (z_coord5 < NzInterior + 1)) {
		s = read2D(d_s, r_coord5, z_coord5, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u5 + (u5_zplus + u5_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord5 - 1)) * u5_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord5 - 1)) * u5_rminus);

		// Calculate update.
		u5_new = u5 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord5, z_coord5, pitch, u5_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord5, z_coord5, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord6 < NrInterior + 1) && (z_coord6 < NzInterior + 1)) {
		s = read2D(d_s, r_coord6, z_coord6, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u6 + (u6_zplus + u6_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord6 - 1)) * u6_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord6 - 1)) * u6_rminus);

		// Calculate update.
		u6_new = u6 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord6, z_coord6, pitch, u6_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord6, z_coord6, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}
	if ((r_coord7 < NrInterior + 1) && (z_coord7 < NzInterior + 1)) {
		s = read2D(d_s, r_coord7, z_coord7, pitch);
		diag = (-4.0 + s * h2);
		res = -(diag * u7 + (u7_zplus + u7_zminus)
				+ (1.0 + 1.0/(double)(2 * r_coord7 - 1)) * u7_rplus
				+ (1.0 - 1.0/(double)(2 * r_coord7 - 1)) * u7_rminus);

		// Calculate update.
		u7_new = u7 + omega * res/diag;
		// Write to global memory.
		write2D(d_u, r_coord7, z_coord7, pitch, u7_new);

		// Calculate "real" residual.
		res *= one_over_h2;
		// Write to global memory.
		write2D(d_res, r_coord7, z_coord7, pitch, res);

		thread_infnorm = MAX(thread_infnorm, abs(res));
		thread_twonorm2 += res * res;
	}

	// Write to shared memory.
	block_infnorm[threadOffset] = thread_infnorm;
	block_twonorm2[threadOffset] = thread_twonorm2;

	__syncthreads();

	// Perform reductions.
	for (int i = 128; i > 0; i /= 2) {
		if (threadOffset < i) {
			block_infnorm[threadOffset] = MAX(block_infnorm[threadOffset], block_infnorm[threadOffset + i]);
			block_twonorm2[threadOffset] += block_twonorm2[threadOffset + i];
		}
		__syncthreads();
	}

	// Write to global memory.
	if (threadOffset == 0) {
		infnorm_array[blockOffset] = block_infnorm[0];
		twonorm2_array[blockOffset] = block_twonorm2[0];
	}

	// All done.
}


/********************************************************************
 ***                        BOUNDARY UPDATE                       ***
 ********************************************************************/

__global__ void boundary_kernel(double *d_u, const size_t pitch, 
				const int NrInterior, const int NzInterior,
			  	const double h, const double u_inf) 
{
	int tid = threadIdx.x;
	int bid = blockIdx.x;

	// Global offset goes from 0, 1, ..., NzInterior-1, NzInterior, ..., 
	// NzInterior + NrInterior - 1.
	// As always, possibly overextends but there will be an out-of-bounds check.

	int offset = tid + bid * blockDim.x;

	// Variables to utilize.
	int r_coord, z_coord;
	double u1, u2, u3, r, z, rr2, temp;

	// Out-of bounds check.
	// Check if we are doing r or z boundary.
	if (offset < (NrInterior + NzInterior)) {
		// R boundary.
		if (offset < NzInterior) {
			// Axis boundary: parity condition.
			z_coord = offset + 1;
			// Fetch value.
			u1 = doubleTex2D(tex_u, 1, z_coord);
			 
			// Write to global memory.
			write2D(d_u, 0, z_coord, pitch, u1);

			// Now do Robin boundary.
			// Fetch values.
			u1 = doubleTex2D(tex_u, NrInterior, z_coord);
			u2 = doubleTex2D(tex_u, NrInterior-1, z_coord);
			u3 = doubleTex2D(tex_u, NrInterior-2, z_coord);

			r = ((double)NrInterior - 0.5) * h;
			z = ((double)z_coord - 0.5) * h;
			rr2 = r * r + z * z;

			// Calculate using fourth-order Robin.
			temp = 3.0 * (r * h/rr2) * (u_inf - u1) 
				- 1.5 * u1 + 3.0 * u2 - 0.5 * u3;

			// Write to global memory.
			write2D(d_u, NrInterior+1, z_coord, pitch, temp);
		}
		// Z boundary.
		else {
			offset -= NzInterior;
			r_coord = offset + 1;

			// Fetch value.
			u1 = doubleTex2D(tex_u, r_coord, 1);

			// Write to global memory.
			write2D(d_u, r_coord, 0, pitch, u1);

			// On the opposite edge we always have Robin.
			// Fetch values.
			u1 = doubleTex2D(tex_u, r_coord, NzInterior);
			u2 = doubleTex2D(tex_u, r_coord, NzInterior-1);
			u3 = doubleTex2D(tex_u, r_coord, NzInterior-2);

			r = ((double)r_coord - 0.5) * h;
			z = ((double)NzInterior - 0.5) * h;
			rr2 = r * r + z * z;

			// Calculate using Robin.
			temp = 3.0 * (z * h/rr2) * (u_inf - u1) 
				- 1.5 * u1 + 3.0 * u2 - 0.5 * u3;

			// Write to global memory.
			write2D(d_u, r_coord, NzInterior+1, pitch, temp);
		}
	}
}

void boundary_update(double *d_u, const size_t pitch, 
		     const int NrInterior, const int NzInterior, 
		     const double h, const double u_inf) 
{
	// We have to update a total of 2 * NrInterior + 2 * NzIterior points 
	// (the corners are unphyisical).
	// All boundary updates are idependent of each other, so they can be easily overlapped.
	// Launch therefore NrInterior + NzInterior total threads distributed as wished.
	dim3 threadBlock(256);
	dim3 gridBlock((NrInterior + NzInterior + threadBlock.x - 1)/threadBlock.x);

	boundary_kernel<<<gridBlock, threadBlock>>>(d_u, pitch, NrInterior, NzInterior, h, u_inf);

	cudaCheckError();
	// Finished.
}


