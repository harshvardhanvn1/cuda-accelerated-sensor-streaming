#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) \
if((call) != cudaSuccess) { \
    std::cerr << "CUDA error" << std::endl; \
    exit(1); \
}

__global__ void computeRMS_naive(const float* input, float* output, 
                                 int num_samples, int num_channels) {
    int ch = blockIdx.x;
    int tid = threadIdx.x;
    if (ch >= num_channels) return;
    
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;
    
    for (int i = tid; i < num_samples; i += blockDim.x) {
        float val = input[i * num_channels + ch];
        sdata[tid] += val * val;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) output[ch] = sqrtf(sdata[0] / num_samples);
}

int main() {
    const int NUM_CHANNELS = 8;
    const int SAMPLES_PER_FILE = 20480;
    const int NUM_FILES = 20; // Profile first 20 files
    
    std::cout << "=== DETAILED TIMING BREAKDOWN: Naive CUDA ===" << std::endl;
    std::cout << "Profiling to show H2D, Kernel, D2H timing\n" << std::endl;
    
    float *h_input = new float[SAMPLES_PER_FILE * NUM_CHANNELS];
    float *h_output = new float[NUM_CHANNELS];
    float *d_input, *d_output;
    
    CHECK_CUDA(cudaMalloc(&d_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, NUM_CHANNELS * sizeof(float)));
    
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    
    cudaEvent_t start_h2d, stop_h2d, start_kernel, stop_kernel, start_d2h, stop_d2h;
    CHECK_CUDA(cudaEventCreate(&start_h2d));
    CHECK_CUDA(cudaEventCreate(&stop_h2d));
    CHECK_CUDA(cudaEventCreate(&start_kernel));
    CHECK_CUDA(cudaEventCreate(&stop_kernel));
    CHECK_CUDA(cudaEventCreate(&start_d2h));
    CHECK_CUDA(cudaEventCreate(&stop_d2h));
    
    float total_h2d = 0, total_kernel = 0, total_d2h = 0;
    
    for (int f = 0; f < NUM_FILES; ++f) {
        file.read(reinterpret_cast<char*>(h_input), SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float));
        
        // Time H2D transfer
        CHECK_CUDA(cudaEventRecord(start_h2d));
        CHECK_CUDA(cudaMemcpy(d_input, h_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float),
                              cudaMemcpyHostToDevice));
        CHECK_CUDA(cudaEventRecord(stop_h2d));
        CHECK_CUDA(cudaEventSynchronize(stop_h2d));
        
        // Time kernel
        CHECK_CUDA(cudaEventRecord(start_kernel));
        computeRMS_naive<<<NUM_CHANNELS, 256>>>(d_input, d_output, SAMPLES_PER_FILE, NUM_CHANNELS);
        CHECK_CUDA(cudaEventRecord(stop_kernel));
        CHECK_CUDA(cudaEventSynchronize(stop_kernel));
        
        // Time D2H transfer
        CHECK_CUDA(cudaEventRecord(start_d2h));
        CHECK_CUDA(cudaMemcpy(h_output, d_output, NUM_CHANNELS * sizeof(float),
                              cudaMemcpyDeviceToHost));
        CHECK_CUDA(cudaEventRecord(stop_d2h));
        CHECK_CUDA(cudaEventSynchronize(stop_d2h));
        
        float h2d_ms, kernel_ms, d2h_ms;
        CHECK_CUDA(cudaEventElapsedTime(&h2d_ms, start_h2d, stop_h2d));
        CHECK_CUDA(cudaEventElapsedTime(&kernel_ms, start_kernel, stop_kernel));
        CHECK_CUDA(cudaEventElapsedTime(&d2h_ms, start_d2h, stop_d2h));
        
        total_h2d += h2d_ms;
        total_kernel += kernel_ms;
        total_d2h += d2h_ms;
    }
    
    float avg_h2d = total_h2d / NUM_FILES;
    float avg_kernel = total_kernel / NUM_FILES;
    float avg_d2h = total_d2h / NUM_FILES;
    float avg_total = avg_h2d + avg_kernel + avg_d2h;
    
    std::cout << "=== NAIVE CUDA BREAKDOWN (RTX A4000) ===" << std::endl;
    printf("%-20s %10.4f ms  (%5.1f%%)\n", "H2D Transfer:", avg_h2d, (avg_h2d/avg_total*100));
    printf("%-20s %10.4f ms  (%5.1f%%)\n", "Kernel Execution:", avg_kernel, (avg_kernel/avg_total*100));
    printf("%-20s %10.4f ms  (%5.1f%%)\n", "D2H Transfer:", avg_d2h, (avg_d2h/avg_total*100));
    printf("%-20s %10.4f ms\n", "TOTAL:", avg_total);
    
    std::cout << "\n❌ Problem: All operations are SERIAL (no overlap)" << std::endl;
    std::cout << "   H2D → Kernel → D2H (blocked execution)" << std::endl;
    
    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    file.close();
    
    return 0;
}
