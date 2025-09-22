#include <cuda.h>
#include <iostream>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

unsigned SM_NUMBER;           // number of SMs
unsigned WARP_SIZE;           // max threads per warp
unsigned MAX_THREADS_PER_SM;  // max threads / sm
unsigned MAX_SHARED_MEM_SIZE; // Max configerable shared memory size in bytes
unsigned MAX_WARPS_PER_SM;    // max warps / sm
unsigned MAX_REG_PER_SM;      // max warps / sm

unsigned MAX_THREAD_BLOCK_SIZE;         // max threads per threadblock
unsigned MAX_SHARED_MEM_SIZE_PER_BLOCK; // Max configerable shared memory size
                                        // per block in bytes
unsigned
    MAX_REG_PER_BLOCK; // Max configerable shared memory size per block in bytes

size_t L2_SIZE; // L2 size in bytes

size_t MEM_SIZE;            // Memory size in bytes
unsigned MEM_CLK_FREQUENCY; // Memory clock freq in MHZ
unsigned MEM_BITWIDTH;      // Memory bit width

// launched threadblocks
unsigned THREADS_PER_BLOCK;
unsigned BLOCKS_PER_SM;
unsigned THREADS_PER_SM;
unsigned BLOCKS_NUM;
unsigned TOTAL_THREADS;

cudaDeviceProp deviceProp;

unsigned intilizeDeviceProp(unsigned deviceID) {
  cudaSetDevice(deviceID);
  cudaGetDeviceProperties(&deviceProp, deviceID);

  // core stats
  SM_NUMBER = deviceProp.multiProcessorCount;
  MAX_THREADS_PER_SM = deviceProp.maxThreadsPerMultiProcessor;
  MAX_SHARED_MEM_SIZE = deviceProp.sharedMemPerMultiprocessor;
  WARP_SIZE = deviceProp.warpSize;
  MAX_WARPS_PER_SM =
      deviceProp.maxThreadsPerMultiProcessor / deviceProp.warpSize;
  MAX_REG_PER_SM = deviceProp.regsPerMultiprocessor;

  // threadblock stats
  MAX_THREAD_BLOCK_SIZE = deviceProp.maxThreadsPerBlock;
  MAX_SHARED_MEM_SIZE_PER_BLOCK = deviceProp.sharedMemPerBlock;
  MAX_REG_PER_BLOCK = deviceProp.regsPerBlock;

  // launched thread blocks to ensure GPU is fully occupied as much as possible
  THREADS_PER_BLOCK = deviceProp.maxThreadsPerBlock;
  BLOCKS_PER_SM =
      deviceProp.maxThreadsPerMultiProcessor / deviceProp.maxThreadsPerBlock;
  THREADS_PER_SM = BLOCKS_PER_SM * THREADS_PER_BLOCK;
  BLOCKS_NUM = BLOCKS_PER_SM * SM_NUMBER;
  TOTAL_THREADS = THREADS_PER_BLOCK * BLOCKS_NUM;

  // L2 cache
  L2_SIZE = deviceProp.l2CacheSize;

  // memory
  MEM_SIZE = deviceProp.totalGlobalMem;
  MEM_CLK_FREQUENCY = deviceProp.memoryClockRate * 1e-3f;
  MEM_BITWIDTH = deviceProp.memoryBusWidth;

  return 1;
}

#define CLK_FREQUENCY 1410 // frequency in MHz
__global__ void empty_kernel() {
  // 空 kernel，只做一次 __syncthreads()
}

float measure_latency(dim3 grid, dim3 block) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  cudaEventRecord(start);
  empty_kernel<<<grid, block>>>();
  cudaEventRecord(stop);

  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);

  return milliseconds * 1000 * CLK_FREQUENCY;  // 转换为 GPU cycles
}

int main() {
  intilizeDeviceProp(0);

  // 定义不同的 grid / block 配置
  std::vector<int> grid_sizes   = {32, 64, 128, 256, 512, 1024, 2048};
  std::vector<int> block_sizes  = {128, 256, 512, 1024};

  std::cout << "Grid Size, Block Size, Latency (cycles)\n";
  for (int b : block_sizes) {
    for (int g : grid_sizes) {
        for (int i = 0; i < 10; i++) { // 重复测量以获得平均值
            float lat = measure_latency(dim3(g), dim3(b));
            std::cout << g << "," << b << "," << lat << "\n";
        }
    }
  }

  return 0;
}
