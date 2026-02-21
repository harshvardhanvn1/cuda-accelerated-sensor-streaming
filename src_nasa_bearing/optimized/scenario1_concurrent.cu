#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) \
if((call) != cudaSuccess) { \
    std::cerr << "CUDA error" << std::endl; \
    exit(1); \
}

__global__ void computeRMS(const float* input, float* output, 
                           int num_samples, int num_channels, int bearing_id) {
    int ch_base = bearing_id * 2;
    __shared__ float sdata[256 * 2];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    sdata[tid] = 0.0f;
    sdata[tid + 256] = 0.0f;
    
    if (idx < num_samples) {
        float val0 = input[idx * num_channels + ch_base];
        float val1 = input[idx * num_channels + ch_base + 1];
        sdata[tid] = val0 * val0;
        sdata[tid + 256] = val1 * val1;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
            sdata[tid + 256] += sdata[tid + s + 256];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(&output[ch_base], sdata[0]);
        atomicAdd(&output[ch_base + 1], sdata[256]);
    }
}

int main() {
    const int NUM_CHANNELS = 8;
    const int NUM_BEARINGS = 4;
    const int SAMPLES_PER_FILE = 20480;
    const int NUM_FILES = 100;
    const int NUM_STREAMS = 4;
    
    float *h_input, *h_output;
    CHECK_CUDA(cudaMallocHost(&h_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output, NUM_CHANNELS * sizeof(float)));
    
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, NUM_CHANNELS * sizeof(float)));
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        file.read(reinterpret_cast<char*>(h_input), SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float));
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        CHECK_CUDA(cudaMemset(d_output, 0, NUM_CHANNELS * sizeof(float)));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (SAMPLES_PER_FILE + threadsPerBlock - 1) / threadsPerBlock;
        
        for (int b = 0; b < NUM_BEARINGS; ++b) {
            computeRMS<<<blocksPerGrid, threadsPerBlock, 0, streams[b]>>>(
                d_input, d_output, SAMPLES_PER_FILE, NUM_CHANNELS, b);
        }
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    file.seekg(0);
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int f = 0; f < NUM_FILES; ++f) {
        file.read(reinterpret_cast<char*>(h_input), SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float));
        
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        CHECK_CUDA(cudaMemset(d_output, 0, NUM_CHANNELS * sizeof(float)));
        
        int threadsPerBlock = 256;
        int blocksPerGrid = (SAMPLES_PER_FILE + threadsPerBlock - 1) / threadsPerBlock;
        
        for (int b = 0; b < NUM_BEARINGS; ++b) {
            computeRMS<<<blocksPerGrid, threadsPerBlock, 0, streams[b]>>>(
                d_input, d_output, SAMPLES_PER_FILE, NUM_CHANNELS, b);
        }
        
        CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[0]));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / NUM_FILES;
    float throughput = SAMPLES_PER_FILE / (avg_time_ms / 1000.0f);
    
    // Output in parseable format
    std::cout << "SCENARIO1_CONCURRENT," << avg_time_ms << "," << (int)throughput << std::endl;
    
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    file.close();
    
    return 0;
}
