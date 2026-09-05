#ifndef HELPER_CUDA_H
#define HELPER_CUDA_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

static inline void checkCudaErrors(cudaError_t err) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error: %s at line %d\n", cudaGetErrorString(err), __LINE__);
        exit(EXIT_FAILURE);
    }
}

#endif // HELPER_CUDA_H
