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

