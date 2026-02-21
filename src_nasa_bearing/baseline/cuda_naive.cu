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
    const int NUM_FILES = 100;
    
    float *h_input = new float[SAMPLES_PER_FILE * NUM_CHANNELS];
    float *h_output = new float[NUM_CHANNELS];
    float *d_input, *d_output;
    
    CHECK_CUDA(cudaMalloc(&d_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, NUM_CHANNELS * sizeof(float)));
    
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        file.read(reinterpret_cast<char*>(h_input), SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float));
        CHECK_CUDA(cudaMemcpy(d_input, h_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float),
                              cudaMemcpyHostToDevice));
        computeRMS_naive<<<NUM_CHANNELS, 256>>>(d_input, d_output, SAMPLES_PER_FILE, NUM_CHANNELS);
        CHECK_CUDA(cudaMemcpy(h_output, d_output, NUM_CHANNELS * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    file.seekg(0);
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int f = 0; f < NUM_FILES; ++f) {
        file.read(reinterpret_cast<char*>(h_input), SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float));
        CHECK_CUDA(cudaMemcpy(d_input, h_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float),
                              cudaMemcpyHostToDevice));
        computeRMS_naive<<<NUM_CHANNELS, 256>>>(d_input, d_output, SAMPLES_PER_FILE, NUM_CHANNELS);
        CHECK_CUDA(cudaMemcpy(h_output, d_output, NUM_CHANNELS * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / NUM_FILES;
    float throughput = SAMPLES_PER_FILE / (avg_time_ms / 1000.0f);
    
    // Output in parseable format
    std::cout << "NAIVE_CUDA," << avg_time_ms << "," << (int)throughput << std::endl;
    
    delete[] h_input;
    delete[] h_output;
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    file.close();
    
    return 0;
}
