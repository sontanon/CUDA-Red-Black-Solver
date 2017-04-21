/********************************************************************
 ***                          OTHER TOOLS                         ***
 ********************************************************************/

 // Indexing macro.
int IDX(const int r, const int z, const int NrTotal, const int NzTotal);

// Write to a 2D CUDA array.
__device__ void write2D(double *d_u, const int r, const int z, 
			const size_t pitch, const double u);

// Read from a 2D CUDA array.
__device__ double read2D(const double *d_u, const int r, const int z, const size_t pitch);

// Read a double texture.
__device__ double doubleTex2D(const texture<int2, cudaTextureType2D, cudaReadModeElementType> tex_Ref,
			      const int r, const int z);
