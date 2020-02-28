#if !defined(SINE_H_)
#define SINE_H_

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <math.h>

__global__ void arrSine(float *A, float* B, float* C, int N) {

    int i = blockDim.x * blockIdx.x + threadIdx.x;

    if (i < N) {
    }
}

#endif // SINE_H_
