#include <iostream>
#include <fstream>
#include <cuda_runtime.h>
#include <cmath>

#define CHECK_CUDA(call) \
if((call) != cudaSuccess) { \
    std::cerr << "CUDA error" << std::endl; \
    exit(1); \
}

// Stage 1: Normalization
__global__ void stage1_normalize(const float* input, float* output, int num_samples, int num_channels) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= num_samples * num_channels) return;
    output[idx] = input[idx];
}

// Stage 2: RMS
__global__ void stage2_rms(const float* input, float* output, int num_samples, int num_channels) {
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
    
    if (tid == 0) {
        output[ch] = sqrtf(sdata[0] / num_samples);
    }
}

// Stage 3: Peak detection
__global__ void stage3_peak(const float* input, float* output, int num_samples, int num_channels) {
    int ch = blockIdx.x;
    int tid = threadIdx.x;
    
    if (ch >= num_channels) return;
    
    __shared__ float sdata[256];
    sdata[tid] = 0.0f;
    
    for (int i = tid; i < num_samples; i += blockDim.x) {
        float val = fabsf(input[i * num_channels + ch]);
        sdata[tid] = fmaxf(sdata[tid], val);
    }
    __syncthreads();
    
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) sdata[tid] = fmaxf(sdata[tid], sdata[tid + s]);
        __syncthreads();
    }
    
    if (tid == 0) {
        output[ch] = sdata[0];
    }
}

int main() {
    const int NUM_CHANNELS = 8;
    const int SAMPLES_PER_FILE = 20480;
    const int NUM_FILES = 100;
    const int NUM_STREAMS = 3;
    
    float *h_input[3], *h_rms[3], *h_peak[3];
    float *d_input[3], *d_normalized[3], *d_rms[3], *d_peak[3];
    
    for (int i = 0; i < 3; ++i) {
        CHECK_CUDA(cudaMallocHost(&h_input[i], SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_rms[i], NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_peak[i], NUM_CHANNELS * sizeof(float)));
        
        CHECK_CUDA(cudaMalloc(&d_input[i], SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_normalized[i], SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_rms[i], NUM_CHANNELS * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_peak[i], NUM_CHANNELS * sizeof(float)));
    }
    
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; ++i) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        int s = i % 3;
        file.read(reinterpret_cast<char*>(h_input[s]), SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float));
        
        CHECK_CUDA(cudaMemcpyAsync(d_input[s], h_input[s], SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        
        stage1_normalize<<<(SAMPLES_PER_FILE * NUM_CHANNELS + 255) / 256, 256, 0, streams[0]>>>(
            d_input[s], d_normalized[s], SAMPLES_PER_FILE, NUM_CHANNELS);
        
        stage2_rms<<<NUM_CHANNELS, 256, 0, streams[1]>>>(
            d_normalized[s], d_rms[s], SAMPLES_PER_FILE, NUM_CHANNELS);
        
        stage3_peak<<<NUM_CHANNELS, 256, 0, streams[2]>>>(
            d_normalized[s], d_peak[s], SAMPLES_PER_FILE, NUM_CHANNELS);
        
        CHECK_CUDA(cudaDeviceSynchronize());
    }
    file.seekg(0);
    
    // Benchmark
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    CHECK_CUDA(cudaEventRecord(start));
    
    for (int f = 0; f < NUM_FILES; ++f) {
        int s = f % 3;
        
        file.read(reinterpret_cast<char*>(h_input[s]), SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float));
        
        CHECK_CUDA(cudaMemcpyAsync(d_input[s], h_input[s], 
                                   SAMPLES_PER_FILE * NUM_CHANNELS * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[0]));
        
        stage1_normalize<<<(SAMPLES_PER_FILE * NUM_CHANNELS + 255) / 256, 256, 0, streams[0]>>>(
            d_input[s], d_normalized[s], SAMPLES_PER_FILE, NUM_CHANNELS);
        
        if (f > 0) {
            int prev = (f - 1) % 3;
            stage2_rms<<<NUM_CHANNELS, 256, 0, streams[1]>>>(
                d_normalized[prev], d_rms[prev], SAMPLES_PER_FILE, NUM_CHANNELS);
        }
        
        if (f > 1) {
            int prev2 = (f - 2) % 3;
            stage3_peak<<<NUM_CHANNELS, 256, 0, streams[2]>>>(
                d_normalized[prev2], d_peak[prev2], SAMPLES_PER_FILE, NUM_CHANNELS);
            
            CHECK_CUDA(cudaMemcpyAsync(h_rms[prev2], d_rms[prev2], NUM_CHANNELS * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[2]));
            CHECK_CUDA(cudaMemcpyAsync(h_peak[prev2], d_peak[prev2], NUM_CHANNELS * sizeof(float),
                                       cudaMemcpyDeviceToHost, streams[2]));
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
    std::cout << "SCENARIO3_PIPELINE," << avg_time_ms << "," << (int)throughput << std::endl;
    
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
