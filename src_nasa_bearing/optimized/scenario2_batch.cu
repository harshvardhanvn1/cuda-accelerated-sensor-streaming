#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) \
if((call) != cudaSuccess) { \
    std::cerr << "CUDA error" << std::endl; \
    exit(1); \
}

__global__ void computeRMS_batch(const float* input, float* output, 
                                 int samples_per_file, int num_channels, int num_files) {
    int file_id = blockIdx.y;
    int ch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (file_id >= num_files || ch >= num_channels) return;
    
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;
    
    int file_offset = file_id * samples_per_file * num_channels;
    
    for (int i = tid; i < samples_per_file; i += blockDim.x) {
        float val = input[file_offset + i * num_channels + ch];
        sdata[tid] += val * val;
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] += sdata[tid + s];
        __syncthreads();
    }
    
    if (tid == 0) {
        output[file_id * num_channels + ch] = sqrtf(sdata[0] / samples_per_file);
    }
}

int main() {
    const int NUM_CHANNELS = 8;
    const int SAMPLES_PER_FILE = 20480;
    const int TOTAL_FILES = 100;
    const int BATCH_SIZE = 5; // Optimal from testing
    
    int batches = TOTAL_FILES / BATCH_SIZE;
    int batch_samples = BATCH_SIZE * SAMPLES_PER_FILE;
    
    float *h_input, *h_output;
    CHECK_CUDA(cudaMallocHost(&h_input, batch_samples * NUM_CHANNELS * sizeof(float)));
    CHECK_CUDA(cudaMallocHost(&h_output, BATCH_SIZE * NUM_CHANNELS * sizeof(float)));
    
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, batch_samples * NUM_CHANNELS * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, BATCH_SIZE * NUM_CHANNELS * sizeof(float)));
    
    cudaStream_t stream;
    CHECK_CUDA(cudaStreamCreate(&stream));
    
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    
    // Warm-up
    for (int i = 0; i < 5; ++i) {
        file.read(reinterpret_cast<char*>(h_input), batch_samples * NUM_CHANNELS * sizeof(float));
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, batch_samples * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        
        dim3 grid(NUM_CHANNELS, BATCH_SIZE);
        computeRMS_batch<<<grid, 256, 0, stream>>>(d_input, d_output, SAMPLES_PER_FILE, NUM_CHANNELS, BATCH_SIZE);
        
        CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, BATCH_SIZE * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    CHECK_CUDA(cudaStreamSynchronize(stream));
    file.seekg(0);
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start, stream));
    
    for (int b = 0; b < batches; ++b) {
        file.read(reinterpret_cast<char*>(h_input), batch_samples * NUM_CHANNELS * sizeof(float));
        
        CHECK_CUDA(cudaMemcpyAsync(d_input, h_input, batch_samples * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, stream));
        
        dim3 grid(NUM_CHANNELS, BATCH_SIZE);
        computeRMS_batch<<<grid, 256, 0, stream>>>(d_input, d_output, SAMPLES_PER_FILE, NUM_CHANNELS, BATCH_SIZE);
        
        CHECK_CUDA(cudaMemcpyAsync(h_output, d_output, BATCH_SIZE * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyDeviceToHost, stream));
    }
    
    CHECK_CUDA(cudaEventRecord(stop, stream));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / TOTAL_FILES;
    float throughput = SAMPLES_PER_FILE / (avg_time_ms / 1000.0f);
    
    // Output in parseable format
    std::cout << "SCENARIO2_BATCH," << avg_time_ms << "," << (int)throughput << std::endl;
    
    CHECK_CUDA(cudaFreeHost(h_input));
    CHECK_CUDA(cudaFreeHost(h_output));
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaStreamDestroy(stream));
    file.close();
    
    return 0;
}
