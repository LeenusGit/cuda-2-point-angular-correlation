#ifndef SIM_CUH_
#define SIM_CUH_

#include <cuda_runtime.h>
#include <vector>
#include <string>
#include "./anglepair.h"

using std::string;
using std::vector;

struct SimResult {
    int *DR;
    int *DD;
    int *RR;
};

__host__ SimResult kernelWrapper(const string &r_path, const string &s_path,
                                  void *output, int N);

__global__ void cmpAngles(Vec3D *p_self, Vec3D *p_other, int *hists, int N);

__global__ void fillAnglesN(Vec3D *r_points, Vec3D *s_points, int *hists,
                            int N);

__global__ void sumAngles(int *global_hist, int *out_hist, int N);

__global__ void cmpEqualAngles(Vec3D *p_self, Vec3D *p_other, int *hists,
                               int N);

__host__ unsigned long totalAngles(int *hist);

#endif // SIM_CUH_
