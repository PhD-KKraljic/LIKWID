/*
 * =======================================================================================
 *
 *      Filename:  triad.cu
 *
 *      Description:  Triad kernel in CUDA to test NvMarkerAPI
 *
 *      Version:   <VERSION>
 *      Released:  <DATE>
 *
 *      Author:   Dominik Ernst (de) dominik.ernst@fau.de
 *      Project:  likwid
 *
 *      Copyright (C) 2019 RRZE, University Erlangen-Nuremberg
 *
 *      This program is free software: you can redistribute it and/or modify it under
 *      the terms of the GNU General Public License as published by the Free Software
 *      Foundation, either version 3 of the License, or (at your option) any later
 *      version.
 *
 *      This program is distributed in the hope that it will be useful, but WITHOUT ANY
 *      WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
 *      PARTICULAR PURPOSE.  See the GNU General Public License for more details.
 *
 *      You should have received a copy of the GNU General Public License along with
 *      this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * =======================================================================================
 */


#include <iomanip>
#include <iostream>
#include <sys/time.h>

#include <hip/hip_runtime.h>
#include <likwid-marker.h>

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

#define GPU_ERROR(ans)                                                         \
  { gpuAssert((ans), __FILE__, __LINE__); }

inline void gpuAssert(hipError_t code, const char *file, int line,
                      bool abort = true) {
  if (code != hipSuccess) {
    std::cerr << "GPUassert: \"" << hipGetErrorString(code) << "\"  in "
              << file << ": " << line << "\n";
    if (abort)
      exit(code);
  }
}

using namespace std;

template <typename T>
__global__ void init_kernel(T *A, const double value, const size_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        A[i] = value;
    }
}

template <typename T>
__global__ void stream_copy_kernel(const T *__restrict__ A,
                                  T *C,
                                  const int64_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        C[i] = A[i];
    }
}

template <typename T>
__global__ void stream_scale_kernel(T *B,
                                  const T *__restrict__ C,
                                  const T *__restrict__ scalar,
                                  const int64_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        B[i] = scalar[0] * C[i];
    }
}

template <typename T>
__global__ void stream_add_kernel(const T *__restrict__ A,
                                const T *__restrict__ B,
                                T *C,
                                const int64_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        C[i] = A[i] + B[i];
    }
}

template <typename T>
__global__ void stream_triad_kernel(T *A,
                                 const T *__restrict__ B,
                                 const T *__restrict__ C,
                                 const T *__restrict__ scalar,
                                 const int64_t N) {
    int tidx = threadIdx.x + blockIdx.x * blockDim.x;
    for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
        A[i] = B[i] + scalar[0] * C[i];
    }
}

#ifndef STREAM_TYPE
#define STREAM_TYPE double
#endif
STREAM_TYPE *a, *b, *c, *scalar;
char region_tag[80];

int main(int argc, char **argv) {
  size_t buffer_size = 128 * 1024 * 1024;
  if (argc == 2)
  {
    buffer_size = atoi(argv[1]);
  }
  cout << "Buffer size: " << buffer_size << endl;

  // Get start time
  double tstart = dtime();

  // Marker init
  ROCMON_MARKER_INIT;
  ROCMON_MARKER_REGISTER("init");

  GPU_ERROR(hipMalloc(&a, buffer_size * sizeof(STREAM_TYPE)));
  GPU_ERROR(hipMalloc(&b, buffer_size * sizeof(STREAM_TYPE)));
  GPU_ERROR(hipMalloc(&c, buffer_size * sizeof(STREAM_TYPE)));
  GPU_ERROR(hipMalloc(&scalar, sizeof(STREAM_TYPE)));

  ROCMON_MARKER_START("init");
  hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, a, 1.0, buffer_size);
  hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, b, 2.0, buffer_size);
  hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, c, 0.0, buffer_size);
  hipLaunchKernelGGL((init_kernel<STREAM_TYPE>), dim3(256), dim3(400), 0, 0, scalar, 3.0, 1);
  ROCMON_MARKER_STOP("init");

  GPU_ERROR(hipDeviceSynchronize());
  const int iters = 100;
  cout << "Iterations: " << iters << endl;

  const int block_size = 512;
  hipDeviceProp_t prop;
  int deviceId = 0;
  GPU_ERROR(hipGetDevice(&deviceId));
  GPU_ERROR(hipGetDeviceProperties(&prop, deviceId));
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, stream_triad_kernel<STREAM_TYPE>, block_size, 0));

  int max_blocks = maxActiveBlocks * smCount;
  cout << "GPU Mapping: " << endl;
  cout << "Workgroups " << max_blocks << " consists of " << maxActiveBlocks << "(max active blocks per cu) * " << smCount << " (amount of Compute Units)" << endl;
  cout << "Threads " << block_size << endl;

  GPU_ERROR(hipDeviceSynchronize());
  double t1 = dtime();
  for (int i = 0; i < iters; i++) {
    sprintf(region_tag, "copy-%ld", buffer_size);
    ROCMON_MARKER_REGISTER(region_tag);
    ROCMON_MARKER_START(region_tag);
    hipLaunchKernelGGL((stream_copy_kernel<STREAM_TYPE>), dim3(max_blocks), dim3(block_size), 0, 0, a, c, buffer_size);
    ROCMON_MARKER_STOP(region_tag);
    sprintf(region_tag, "scale-%ld", buffer_size);
    ROCMON_MARKER_REGISTER(region_tag);
    ROCMON_MARKER_START(region_tag);
    hipLaunchKernelGGL((stream_scale_kernel<STREAM_TYPE>), dim3(max_blocks), dim3(block_size), 0, 0, b, c, scalar, buffer_size);
    ROCMON_MARKER_STOP(region_tag);
    sprintf(region_tag, "sum-%ld", buffer_size);
    ROCMON_MARKER_REGISTER(region_tag);
    ROCMON_MARKER_START(region_tag);
    hipLaunchKernelGGL((stream_add_kernel<STREAM_TYPE>), dim3(max_blocks), dim3(block_size), 0, 0, a, b, c, buffer_size);
    ROCMON_MARKER_STOP(region_tag);
    sprintf(region_tag, "triad-%ld", buffer_size);
    ROCMON_MARKER_REGISTER(region_tag);
    ROCMON_MARKER_START(region_tag);
    hipLaunchKernelGGL((stream_triad_kernel<STREAM_TYPE>), dim3(max_blocks), dim3(block_size), 0, 0, a, b, c, scalar, buffer_size);
    ROCMON_MARKER_STOP(region_tag);
  }
  GPU_ERROR(hipGetLastError());
  GPU_ERROR(hipDeviceSynchronize());
  double t2 = dtime();

  // Marker stop
  ROCMON_MARKER_CLOSE;

  GPU_ERROR(hipFree(a));
  GPU_ERROR(hipFree(b));
  GPU_ERROR(hipFree(c));
  GPU_ERROR(hipFree(scalar));

  // Get start time
  double tstop = dtime();

  cout << "Total time: " << (tstop - tstart) * 1000 << " ms" << endl;
  cout << "Iteration time: " << (t2 - t1) * 1000 << " ms" << endl;

  double dt = (t2 - t1) / iters;
  cout << fixed << setprecision(2) << setw(6) << dt * 1000 << "ms " << setw(5)
       << 4 * buffer_size * sizeof(double) / dt * 1e-9 << "GB/s \n";

  return 0;
}
