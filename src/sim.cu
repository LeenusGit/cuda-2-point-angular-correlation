#include "sim.cuh"

#include <stdlib.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include <math_constants.h>
#include <device_atomic_functions.h>
#include <time.h>

#include <iostream>
#include <string>
#include <vector>

#include "./cudaDmy.cuh"
#include "./filereader.h"

using std::cout;
using std::string;

#define RAD_2_DEGREE 57.2957795f
#define THREADS_IN_BLOCK 512
#define THREADS_IN_SUM_BLOCK 512
#define BIN_COUNT 360

__host__ SimResult kernelWrapper(const string &r_path, const string &s_path,
                                 void *output, int N) {

    int grid_width = (N + THREADS_IN_BLOCK - 1) / THREADS_IN_BLOCK;
    int grid_height = grid_width;

    dim3 grid(grid_width, grid_height);
    dim3 block(THREADS_IN_BLOCK, 1);

    uint64_t blocks_in_grid = grid.x * grid.y;
    int blocks_for_sum =
        (blocks_in_grid + THREADS_IN_SUM_BLOCK - 1) / THREADS_IN_SUM_BLOCK;

    dim3 sum_grid(blocks_for_sum, 1);
    dim3 sum_block(THREADS_IN_SUM_BLOCK, 1);

    // grid.x * grid.y;

    int data_size = sizeof(Vec3D) * N;

    uint64_t sub_h_grams_size = BIN_COUNT * sizeof(int) * blocks_in_grid;
    int output_hist_size = BIN_COUNT * (sizeof(int));

    SimResult ret;

    int *DR = reinterpret_cast<int *>(malloc(BIN_COUNT * sizeof(int)));
    int *RR = reinterpret_cast<int *>(malloc(BIN_COUNT * sizeof(int)));
    int *DD = reinterpret_cast<int *>(malloc(BIN_COUNT * sizeof(int)));

    vector<Vec3D> r_points;
    vector<Vec3D> s_points;

    Vec3D *d_r_points;
    Vec3D *d_s_points;
    int *d_sub_h_grams;
    int *d_output_hist;

    clock_t start = clock();
    clock_t stop;
    double duration;

    r_points = readSphericalCoords(r_path, N);
    // s_points = readSphericalCoords(s_path, N);
    stop = clock() - start;

    duration = ((double)stop / CLOCKS_PER_SEC);

    cout << "read duration: " << duration * 1000 << " milliseconds\n";

    // The first cuda call will eat the time for initialization
    cudaFree(0);
    // cudaMemsetAsync(0, 0, 0);

    stop = clock() - stop;
    duration = ((double)stop / CLOCKS_PER_SEC);
    cout << "Cuda initialization done in: " << duration * 1000
         << " milliseconds\n";

    // Allocate and copy real angles
    cudaMalloc(&d_r_points, data_size);
    cudaMemcpy(d_r_points, r_points.data(), data_size, cudaMemcpyHostToDevice);

    // Allocate memory for each subhistogram
    cudaMalloc(&d_sub_h_grams, sub_h_grams_size);
    cudaMemset(d_sub_h_grams, 0, sub_h_grams_size);

    // Allocate memory for the result histogram;
    cudaMalloc(&d_output_hist, output_hist_size);
    cudaMemset(d_output_hist, 0, output_hist_size);

    // DD - Non blocking call
    cmpEqualAngles LAUNCH_KERNEL_ARG3(grid, block, sizeof(Vec3D) * block.x)(
        d_r_points, d_r_points, d_sub_h_grams, N);

    // Read the synthetic data while previous call is running
    s_points = readSphericalCoords(s_path, N);

    // Allocate and copy synthetic angle data
    cudaMalloc(&d_s_points, data_size);
    cudaMemcpy(d_s_points, s_points.data(), data_size, cudaMemcpyHostToDevice);

    // cudaDeviceSynchronize();

    // Sum the DD angles
    sumAngles LAUNCH_KERNEL_ARG2(sum_grid, sum_block)(
        d_sub_h_grams, d_output_hist, blocks_in_grid);

    cudaMemcpy(DD, d_output_hist, output_hist_size, cudaMemcpyDeviceToHost);
    cudaMemset(d_output_hist, 0, output_hist_size);
    cudaMemset(d_sub_h_grams, 0, sub_h_grams_size);

    // DR
    cmpAngles LAUNCH_KERNEL_ARG3(grid, block, sizeof(Vec3D) * block.x)(
        d_r_points, d_s_points, d_sub_h_grams, N);

    // cudaDeviceSynchronize();

    sumAngles LAUNCH_KERNEL_ARG2(sum_grid, sum_block)(
        d_sub_h_grams, d_output_hist, blocks_in_grid);

    cudaMemcpy(DR, d_output_hist, output_hist_size, cudaMemcpyDeviceToHost);
    cudaMemset(d_output_hist, 0, output_hist_size);
    cudaMemset(d_sub_h_grams, 0, sub_h_grams_size);

    // RR
    cmpEqualAngles LAUNCH_KERNEL_ARG3(grid, block, sizeof(Vec3D) * block.x)(
        d_s_points, d_s_points, d_sub_h_grams, N);

    // cudaDeviceSynchronize();

    sumAngles LAUNCH_KERNEL_ARG2(sum_grid, sum_block)(
        d_sub_h_grams, d_output_hist, blocks_in_grid);
    cudaMemcpy(RR, d_output_hist, output_hist_size, cudaMemcpyDeviceToHost);

    cout << "Total angles in DR: " << totalAngles(DR) << "\n";
    cout << "Total angles in DD: " << totalAngles(DD) << "\n";
    cout << "Total angles in RR: " << totalAngles(RR) << "\n";
    cout << "--------------------------\n";

    ret.DR = DR;
    ret.DD = DD;
    ret.RR = RR;

    cudaFree(d_r_points);
    cudaFree(d_s_points);
    cudaFree(d_sub_h_grams);
    cudaFree(d_output_hist);

    return ret;
}

__global__ void cmpAngles(Vec3D *p_self, Vec3D *p_other, int *hists, int N) {

    uint64_t blockId = blockIdx.x + blockIdx.y * gridDim.x +
                       gridDim.x * gridDim.y * blockIdx.z;

    __shared__ int sub_hist[BIN_COUNT];
    for (int j = 0; j < BIN_COUNT; j++) {
        sub_hist[j] = 0;
    }

    extern __shared__ Vec3D s_mem[];

    int global_x = blockDim.x * blockIdx.x + threadIdx.x;
    int global_y = blockDim.x * blockIdx.y + threadIdx.y;

    if (global_x >= N || global_y >= N) {
        // printf("exit thread when g_x: %d and g_y: %d\n", global_x, global_y);
        return;
    }

    // // Load angles to compare against into shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            s_mem[i] = p_other[global_y + i];
        }
        // for (int i = 0; i < blockDim.x; i++) {
        //     s_mem2[i] = p_self[global_x + i];
        // }
    }
    __syncthreads();

    // float x1 = s_mem2[threadIdx.x].x;
    // float y1 = s_mem2[threadIdx.x].y;
    // float z1 = s_mem2[threadIdx.x].z;

    float x1 = p_self[global_x].x;
    float y1 = p_self[global_x].y;
    float z1 = p_self[global_x].z;

    int y_stop = blockDim.x;

    // Adjust stop when we would go over N
    if (global_y + blockDim.x >= N) {
        y_stop = N - global_y;
    }

    for (int i = 0; i < y_stop; i++) {

        float angle = RAD_2_DEGREE * acosf(x1 * s_mem[i].x + y1 * s_mem[i].y +
                                           z1 * s_mem[i].z);

        unsigned int bin_nr = (unsigned)(angle * 4);

        if (bin_nr < 360) {

            atomicAdd(&sub_hist[bin_nr], 1);

        } else {
            // printf("Bin nr > 359 found! Nr: %d; ", bin_nr);
            atomicAdd(&sub_hist[359], 1);
        }
    }
    __syncthreads();

    // Should execute only for one thread per block
    if (threadIdx.x == 0 && threadIdx.y == 0) {

        // Add an offset according to block id
        hists += blockId * BIN_COUNT;

        // Write subhistogram to global memory
        for (int j = 0; j < BIN_COUNT; j++) {
            hists[j] = sub_hist[j];
        }
    }
}

__global__ void fillAnglesN(Vec3D *p_self, Vec3D *p_other, int *hists, int N) {

    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;

    __shared__ int s_hist[BIN_COUNT];
    for (int i = 0; i < BIN_COUNT; i++) {
        s_hist[i] = 0;
    }

    __syncthreads();

    if (g_idx < N) {

        float x1 = p_self[g_idx].x;
        float y1 = p_self[g_idx].y;
        float z1 = p_self[g_idx].z;

        // if (g_idx == 0) {
        //     printf("%f %f %f; %f %f %f\n", x1, y1, z1, p_other[g_idx].x,
        //            p_other[g_idx].y, p_other[g_idx].z);
        // }

        for (int i = 0; i < N; i++) {

            if (i > g_idx) {
                continue;
            }

            float angle =
                RAD_2_DEGREE * acosf(x1 * p_other[i].x + y1 * p_other[i].y +
                                     z1 * p_other[i].z);

            unsigned int bin_nr = (unsigned)(angle / 0.25f);

            if (bin_nr < 360) {
                atomicAdd(&s_hist[bin_nr], 1);
            } else {
                printf("Bin nr > 359 found! Nr: %d\n", bin_nr);
                atomicAdd(&s_hist[359], 1);
            }

            // if (i == 0 && g_idx == 0) {
            //     printf("Angle[0][0] = %f\n", angle);
            //     printf("%f %f %f; %f %f %f\n", x1, y1, z1, p_other[i].x,
            //            p_other[i].y, p_other[i].z);
            // }
        }

        // Wait for all threads to finish
        __syncthreads();

        if (threadIdx.x == 0) {

            // Add an offset according to block id
            hists += blockIdx.x * BIN_COUNT;

            for (int j = 0; j < BIN_COUNT; j++) {
                hists[j] = s_hist[j];
            }
        }
    }
}

__global__ void cmpEqualAngles(Vec3D *p_self, Vec3D *p_other, int *hists,
                               int N) {

    uint64_t blockId = blockIdx.x + blockIdx.y * gridDim.x +
                       gridDim.x * gridDim.y * blockIdx.z;

    __shared__ int sub_hist[BIN_COUNT];
    for (int j = 0; j < BIN_COUNT; j++) {
        sub_hist[j] = 0;
    }

    extern __shared__ Vec3D s_mem[];

    int global_x = blockDim.x * blockIdx.x + threadIdx.x;
    int global_y = blockDim.x * blockIdx.y + threadIdx.y;

    // Do nothing if our thread is outside of N x N
    if (global_x >= N || global_y >= N) {
        // printf("exit thread when g_x: %d and g_y: %d\n", global_x, global_y);
        return;
    }

    if (blockIdx.x > blockIdx.y) {
        return;
    }

    // // Load angles to compare against into shared memory
    if (threadIdx.x == 0 && threadIdx.y == 0) {
        for (int i = 0; i < blockDim.x; i++) {
            s_mem[i] = p_other[global_y + i];
        }
    }
    __syncthreads();

    float x1 = p_self[global_x].x;
    float y1 = p_self[global_x].y;
    float z1 = p_self[global_x].z;

    int y_start = 0;
    int y_stop = blockDim.x;
    bool count_twice = true;

    // Count once only when we are on the diagonal
    if (global_x == global_y) {
        count_twice = false;
    }

    // Offset y_start if block is on the diagonal
    if (global_x > global_y && blockIdx.x == blockIdx.y) {
        y_start = global_x - global_y;
    }

    // Stop comparing when we go outside of N in y-direction
    if (global_y + blockDim.x >= N) {
        y_stop = N - global_y;
    }

    for (int i = y_start; i < y_stop; i++) {

        float angle = RAD_2_DEGREE * acosf(x1 * s_mem[i].x + y1 * s_mem[i].y +
                                           z1 * s_mem[i].z);

        unsigned int bin_nr = (unsigned)(angle * 4);

        if (bin_nr < 360) {

            if (!count_twice) {
                atomicAdd(&sub_hist[bin_nr], 1);
            } else {
                atomicAdd(&sub_hist[bin_nr], 2);
            }

        } else {
            printf("Bin nr > 359 found! Nr: %d; ", bin_nr);
            printf("Global x: %d, global y: %d\n", bin_nr);
            atomicAdd(&sub_hist[359], 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0 && threadIdx.y == 0) {

        // Add an offset according to block id
        hists += blockId * BIN_COUNT;

        for (int j = 0; j < BIN_COUNT; j++) {
            hists[j] = sub_hist[j];
        }
    }
}

__global__ void sumAngles(int *global_hist, int *out_hist, int N) {

    int g_idx = blockDim.x * blockIdx.x + threadIdx.x;

    if (g_idx >= N) {
        return;
    }

    global_hist += BIN_COUNT * g_idx;

    for (size_t j = 0; j < BIN_COUNT; j++) {
        atomicAdd(&out_hist[j], global_hist[j]);
        // out_hist[j] += global_hist[j];
    }
}

__host__ unsigned long totalAngles(int *hist) {

    unsigned long total_angles = 0;
    for (int i = 0; i < BIN_COUNT; i++) {

        int val = hist[i];
        total_angles += val;

        // cout << val << "\n";
    }
    return total_angles;
}