#pragma once
#ifdef __INTELLISENSE__
void __syncthreads(); // workaround __syncthreads warning
int atomicAdd(void *address, int val);
// int atomicAdd_block(int *address, int val);
#define __CUDACC__
#define LAUNCH_KERNEL_ARG2(grid, block)
#define LAUNCH_KERNEL_ARG3(grid, block, sh_mem)
#define LAUNCH_KERNEL_ARG4(grid, block, sh_mem, stream)
#else
#define LAUNCH_KERNEL_ARG2(grid, block) <<< grid, block >>>
#define LAUNCH_KERNEL_ARG3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define LAUNCH_KERNEL_ARG4(grid, block, sh_mem, stream)                        \
< < < grid, block, sh_mem, stream >>>
#endif
#define gpuErrChk(ans)                                                         \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line,
                      bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file,
                line);
        if (abort)
            exit(code);
    }
}
