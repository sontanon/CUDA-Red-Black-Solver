#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <time.h>

#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include "cudaErrorCheck.h"
#include "tools.h"
#include "textures.h"

// Kernel.

__global__ void boundary_kernel(double *d_u, const size_t pitch, 
				const int NrInterior, 
				const int NzInterior, 
				const double h, const double u_inf);

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
