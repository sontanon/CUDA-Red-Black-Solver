#include <assert.h>
/******************************************************************
 ***                          TEXTURES                          ***
 ******************************************************************/

 // Indexing macro.
int IDX(const int r, const int z, const int NrTotal, const int NzTotal)
 {
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
