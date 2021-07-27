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
#include <likwid.h>

double dtime() {
  double tseconds = 0;
  struct timeval t;
  gettimeofday(&t, NULL);
  tseconds = (double)t.tv_sec + (double)t.tv_usec * 1.0e-6;
  return tseconds;
}

#define LIKWID_TEST(ans) { int err = (ans); if (err < 0) { std::cerr << "Error " << err << " in `" << #ans << "` (" << __FILE__ << ": " << __LINE__ << ")" << std::endl; return -err; } }
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
__global__ void init_kernel(T *A, const T *__restrict__ B,
                            const T *__restrict__ C, const T *__restrict__ D,
                            const size_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (size_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = 0.1;
  }
}

template <typename T>
__global__ void sch_triad_kernel(T *A, const T *__restrict__ B,
                                 const T *__restrict__ C,
                                 const T *__restrict__ D, const int64_t N) {
  int tidx = threadIdx.x + blockIdx.x * blockDim.x;
  for (int64_t i = tidx; i < N; i += blockDim.x * gridDim.x) {
    A[i] = B[i] + C[i] * D[i];
  }
}

void print_results(int gid)
{
  for (int j = 0; j < rocmon_getNumberOfEvents(gid); j++)
  {
    char* name = rocmon_getEventName(gid, j);
    double lastValue = rocmon_getLastResult(0, gid, j);
    double fullValue = rocmon_getResult(0, gid, j);

    printf("%s: %.2f / %.2f\n", name, fullValue, lastValue);
  }
}

int main(int argc, char **argv) {
  const size_t buffer_size = 248 * 1024 * 1024;

  double *dA, *dB, *dC, *dD;

  GPU_ERROR(hipMalloc(&dA, buffer_size * sizeof(double)));
  GPU_ERROR(hipMalloc(&dB, buffer_size * sizeof(double)));
  GPU_ERROR(hipMalloc(&dC, buffer_size * sizeof(double)));
  GPU_ERROR(hipMalloc(&dD, buffer_size * sizeof(double)));

  hipLaunchKernelGGL((init_kernel<double>), dim3(256), dim3(400), 0, 0, dA, dA, dA, dA, buffer_size);
  hipLaunchKernelGGL((init_kernel<double>), dim3(256), dim3(400), 0, 0, dB, dB, dB, dB, buffer_size);
  hipLaunchKernelGGL((init_kernel<double>), dim3(256), dim3(400), 0, 0, dC, dC, dC, dC, buffer_size);
  hipLaunchKernelGGL((init_kernel<double>), dim3(256), dim3(400), 0, 0, dD, dD, dD, dD, buffer_size);

  GPU_ERROR(hipDeviceSynchronize());
  const int iters = 10;

  const int block_size = 512;
  hipDeviceProp_t prop;
  int deviceId = 0;
  GPU_ERROR(hipGetDevice(&deviceId));
  GPU_ERROR(hipGetDeviceProperties(&prop, deviceId));
  int smCount = prop.multiProcessorCount;
  int maxActiveBlocks = 0;
  GPU_ERROR(hipOccupancyMaxActiveBlocksPerMultiprocessor(
      &maxActiveBlocks, sch_triad_kernel<double>, block_size, 0));

  int max_blocks = maxActiveBlocks * smCount;

  int gpuIds[1] = {deviceId}; 
  if (rocmon_init(1, gpuIds) < 0)
  {
      printf("Failed to initialize Rocmon\n");
      return -1;
  }

  EventList_rocm_t* list;
  if (rocmon_getEventsOfGpu(deviceId, &list) < 0)
  {
    printf("Failed to fetch event list\n");
    return -1;
  }
  for (int i = 0; i < list->numEvents; i++)
  {
    printf("-- %s [%d]\n", list->events[i].name, list->events[i].instances);
  }
  rocmon_freeEventsOfGpu(list);

  int gid = -1;
  if (rocmon_addEventSet("RSMI_POWER_AVE[0]:_,RSMI_VOLT_VDDGFX:_", &gid) < 0 || gid < 0)
  {
      printf("Failed to add event set\n");
      return -1;
  }
  LIKWID_TEST(rocmon_setupCounters(gid));
  LIKWID_TEST(rocmon_startCounters());

  hipLaunchKernelGGL((sch_triad_kernel<double>), dim3(max_blocks), dim3(block_size), 0, 0, dA, dB, dC, dD, buffer_size);

  GPU_ERROR(hipDeviceSynchronize());
  double t1 = dtime();
  for (int i = 0; i < iters; i++) {
    hipLaunchKernelGGL((sch_triad_kernel<double>), dim3(max_blocks), dim3(block_size), 0, 0, dA, dB, dC, dD, buffer_size);

    LIKWID_TEST(rocmon_readCounters());
    std::cout << i << ":" << std::endl; 
    print_results(gid);
    std::cout << std::endl;
  }
  GPU_ERROR(hipGetLastError());
  GPU_ERROR(hipDeviceSynchronize());
  double t2 = dtime();

  LIKWID_TEST(rocmon_stopCounters());
  print_results(gid);
  printf("Time of group: %f / %f\n", rocmon_getTimeOfGroup(0), rocmon_getLastTimeOfGroup(0));

  double dt = (t2 - t1) / iters;

  cout << fixed << setprecision(2) << setw(6) << dt * 1000 << "ms " << setw(5)
       << 4 * buffer_size * sizeof(double) / dt * 1e-9 << "GB/s \n";

  rocmon_finalize();
  GPU_ERROR(hipFree(dA));
  GPU_ERROR(hipFree(dB));
  GPU_ERROR(hipFree(dC));
  GPU_ERROR(hipFree(dD));
  return 0;
}
