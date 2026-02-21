#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) \
if((call) != cudaSuccess) { \
    std::cerr << "CUDA error" << std::endl; \
    exit(1); \
}

__global__ void normalize(const float* input, float* output, int num_samples, int num_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples * num_channels) return;
    output[idx] = input[idx];
}

__global__ void compute_bearing_rms(const float* input, float* output, 
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
        atomicAdd(&output[ch_base], sqrtf(sdata[0] / num_samples));
        atomicAdd(&output[ch_base + 1], sqrtf(sdata[256] / num_samples));
    }
}

__global__ void detect_peaks(const float* input, float* output, int num_samples, int num_channels) {
    int ch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (ch >= num_channels) return;
    
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;
    
    for (int i = tid; i < num_samples; i += blockDim.x) {
        sdata[tid] = fmaxf(sdata[tid], fabsf(input[i * num_channels + ch]));
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    
    if (tid == 0) output[ch] = sdata[0];
}

int main() {
    const int NUM_CHANNELS = 8;
    const int NUM_BEARINGS = 4;
    const int SAMPLES_PER_FILE = 20480;
    const int NUM_FILES = 100;
    const int BATCH_SIZE = 5;
    const int NUM_STREAMS = 12;
    
    int batches = NUM_FILES / BATCH_SIZE;
    int batch_samples = BATCH_SIZE * SAMPLES_PER_FILE;
    
    float *h_input[3], *h_rms[3], *h_peak[3];
    float *d_input[3], *d_normalized[3], *d_rms[3], *d_peak[3];
    
    for (int i = 0; i < 3; ++i) {
        CHECK_CUDA(cudaMallocHost(&h_input[i], batch_samples * NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_rms[i], NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_peak[i], NUM_CHANNELS * sizeof(float)));
        
        CHECK_CUDA(cudaMalloc(&d_input[i], batch_samples * NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_normalized[i], batch_samples * NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_rms[i], NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_peak[i], NUM_CHANNELS * sizeof(float)));
    }
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    
    // Warm-up
    for (int i = 0; i < 5; ++i) {
        int s = i % 3;
        file.read(reinterpret_cast<char*>(h_input[s]), batch_samples * NUM_CHANNELS * sizeof(float));
        
        CHECK_CUDA(cudaMemcpyAsync(d_input[s], h_input[s], 
                                   batch_samples * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        
        normalize<<<(batch_samples * NUM_CHANNELS + 255) / 256, 256, 0, streams[0]>>>(
            d_input[s], d_normalized[s], batch_samples, NUM_CHANNELS);
        
        CHECK_CUDA(cudaMemsetAsync(d_rms[s], 0, NUM_CHANNELS * sizeof(float), streams[1]));
        
        for (int b = 0; b < NUM_BEARINGS; ++b) {
            int stream_id = 1 + b;
            compute_bearing_rms<<<(batch_samples + 255) / 256, 256, 0, streams[stream_id]>>>(
                d_normalized[s], d_rms[s], batch_samples, NUM_CHANNELS, b);
        }
        
        detect_peaks<<<NUM_CHANNELS, 256, 0, streams[5]>>>(
            d_normalized[s], d_peak[s], batch_samples, NUM_CHANNELS);
        
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    file.seekg(0);
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int batch = 0; batch < batches; ++batch) {
        int s = batch % 3;
        
        file.read(reinterpret_cast<char*>(h_input[s]), batch_samples * NUM_CHANNELS * sizeof(float));
        
        CHECK_CUDA(cudaMemcpyAsync(d_input[s], h_input[s], 
                                   batch_samples * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        
        normalize<<<(batch_samples * NUM_CHANNELS + 255) / 256, 256, 0, streams[0]>>>(
            d_input[s], d_normalized[s], batch_samples, NUM_CHANNELS);
        
        if (batch > 0) {
            int prev = (batch - 1) % 3;
            CHECK_CUDA(cudaMemsetAsync(d_rms[prev], 0, NUM_CHANNELS * sizeof(float), streams[1]));
            
            for (int b = 0; b < NUM_BEARINGS; ++b) {
                int stream_id = 1 + b;
                compute_bearing_rms<<<(batch_samples + 255) / 256, 256, 0, streams[stream_id]>>>(
                    d_normalized[prev], d_rms[prev], batch_samples, NUM_CHANNELS, b);
            }
        }
        
        if (batch > 1) {
            int prev2 = (batch - 2) % 3;
            detect_peaks<<<NUM_CHANNELS, 256, 0, streams[5]>>>(
                d_normalized[prev2], d_peak[prev2], batch_samples, NUM_CHANNELS);
            
            CHECK_CUDA(cudaMemcpyAsync(h_rms[prev2], d_rms[prev2], NUM_CHANNELS * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[5]));
            CHECK_CUDA(cudaMemcpyAsync(h_peak[prev2], d_peak[prev2], NUM_CHANNELS * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[5]));
        }
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    float avg_time_ms = milliseconds / NUM_FILES;
    float throughput = SAMPLES_PER_FILE / (avg_time_ms / 1000.0f);
    
    // Output in parseable format
    std::cout << "COMBINED_ALL," << avg_time_ms << "," << (int)throughput << std::endl;
    
    for (int i = 0; i < 3; ++i) {
        CHECK_CUDA(cudaFreeHost(h_input[i]));
        CHECK_CUDA(cudaFreeHost(h_rms[i]));
        CHECK_CUDA(cudaFreeHost(h_peak[i]));
        CHECK_CUDA(cudaFree(d_input[i]));
        CHECK_CUDA(cudaFree(d_normalized[i]));
        CHECK_CUDA(cudaFree(d_rms[i]));
        CHECK_CUDA(cudaFree(d_peak[i]));
    }
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    file.close();
    
    return 0;
}
