#include <iostream>
#include <fstream>
#include <vector>
#include <cuda_runtime.h>

// CUDA error checking macro
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ << std::endl; \
            std::cerr << cudaGetErrorString(err) << std::endl; \
            exit(1); \
        } \
    } while(0)

// Naive CUDA kernel - simple parallelization
__global__ void preprocessNaive(const float* input, float* output, 
                                int num_samples, int num_features) {
    // Calculate global thread index
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Boundary check
    if (idx >= num_samples) {
        return;
    }
    
    // Step 1: Normalization (simplified - just copy for now)
    // Each thread processes one sample (6 features)
    for (int f = 0; f < num_features; f++) {
        int pos = idx * num_features + f;
        // Simplified normalization: (x - 0) / 1 = x
        output[pos] = input[pos];
    }
    
    // Step 2: Moving average (simplified - window size 5)
    if (idx < num_samples - 5) {
        for (int f = 0; f < num_features; f++) {
            float sum = 0.0f;
            // Sum 5 consecutive samples
            for (int w = 0; w < 5; w++) {
                sum += output[(idx + w) * num_features + f];
            }
            // Store average
            output[idx * num_features + f] = sum / 5.0f;
        }
    }
}

int main() {
    // Configuration
    const int BATCH_SIZE = 1000;
    const int NUM_FEATURES = 6;
    const int NUM_BATCHES = 500;
    
    std::cout << "=== Naive CUDA Benchmark ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Features: " << NUM_FEATURES << std::endl;
    std::cout << "  Number of batches: " << NUM_BATCHES << std::endl;
    
    // Allocate host memory
    std::vector<float> h_input(BATCH_SIZE * NUM_FEATURES);
    std::vector<float> h_output(BATCH_SIZE * NUM_FEATURES);
    
    // Allocate device memory
    float *d_input, *d_output;
    CHECK_CUDA(cudaMalloc(&d_input, BATCH_SIZE * NUM_FEATURES * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_output, BATCH_SIZE * NUM_FEATURES * sizeof(float)));
    
    // Open data file
    std::ifstream file("data/sensor_batches_full.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data/sensor_batches_full.bin" << std::endl;
        return 1;
    }
    
    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "\nGPU Launch Configuration:" << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "  Blocks per grid: " << blocksPerGrid << std::endl;
    std::cout << "  Total threads: " << blocksPerGrid * threadsPerBlock << std::endl;
    
    // Warm-up phase
    std::cout << "\nWarming up GPU..." << std::endl;
    for (int i = 0; i < 10; i++) {
        file.read(reinterpret_cast<char*>(h_input.data()), 
                  h_input.size() * sizeof(float));
        
        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), 
                              BATCH_SIZE * NUM_FEATURES * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        preprocessNaive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, 
                                                             BATCH_SIZE, NUM_FEATURES);
        
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, 
                              BATCH_SIZE * NUM_FEATURES * sizeof(float),
                              cudaMemcpyDeviceToHost));
    }
    
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Reset file
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Benchmark
    std::cout << "Running benchmark..." << std::endl;
    
    CHECK_CUDA(cudaEventRecord(start));
    
    float checksum = 0.0f;
    
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        // Read batch from file
        file.read(reinterpret_cast<char*>(h_input.data()), 
                  h_input.size() * sizeof(float));
        
        // H2D transfer
        CHECK_CUDA(cudaMemcpy(d_input, h_input.data(), 
                              BATCH_SIZE * NUM_FEATURES * sizeof(float),
                              cudaMemcpyHostToDevice));
        
        // Launch kernel
        preprocessNaive<<<blocksPerGrid, threadsPerBlock>>>(d_input, d_output, 
                                                             BATCH_SIZE, NUM_FEATURES);
        
        // D2H transfer
        CHECK_CUDA(cudaMemcpy(h_output.data(), d_output, 
                              BATCH_SIZE * NUM_FEATURES * sizeof(float),
                              cudaMemcpyDeviceToHost));
        
        // Prevent optimization
        checksum += h_output[0];
    }
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Calculate timing
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avg_batch_time_ms = milliseconds / NUM_BATCHES;
    float throughput = BATCH_SIZE / (avg_batch_time_ms / 1000.0f);
    float per_sample_time_us = (avg_batch_time_ms * 1000.0f) / BATCH_SIZE;
    
    // Print results
    std::cout << "\nVerification:" << std::endl;
    std::cout << "  output[0] = " << h_output[0] << std::endl;
    std::cout << "  Checksum: " << checksum << std::endl;
    
    std::cout << "\n=== Naive CUDA Results ===" << std::endl;
    std::cout << "Total time: " << milliseconds << " ms" << std::endl;
    std::cout << "Average batch time: " << avg_batch_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (int)throughput << " samples/sec" << std::endl;
    std::cout << "Per-sample time: " << per_sample_time_us << " μs" << std::endl;
    
    std::cout << "\n Speedup vs CPU (0.073ms baseline): " 
              << (0.073f / avg_batch_time_ms) << "x" << std::endl;
    
    // Cleanup
    CHECK_CUDA(cudaFree(d_input));
    CHECK_CUDA(cudaFree(d_output));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    file.close();
    
    return 0;
}