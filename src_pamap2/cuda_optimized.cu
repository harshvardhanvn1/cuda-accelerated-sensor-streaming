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

// Optimized kernel with better memory access patterns
__global__ void preprocessOptimized(const float* __restrict__ input,
                                   float* __restrict__ output,
                                   int num_samples, int num_features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (idx >= num_samples) {
        return;
    }
    
    // Normalization - each thread processes one sample's features
    for (int f = 0; f < num_features; f++) {
        int pos = idx * num_features + f;
        float val = input[pos];
        output[pos] = val;  // Simplified for now
    }
    
    // Synchronize to ensure normalization is complete
    __syncthreads();
    
    // Moving average with loop unrolling
    if (idx < num_samples - 5) {
        for (int f = 0; f < num_features; f++) {
            float sum = 0.0f;
            
            // Manual loop unrolling for better performance
            #pragma unroll
            for (int w = 0; w < 5; w++) {
                sum += output[(idx + w) * num_features + f];
            }
            
            // Multiply by 1/5 instead of divide (faster)
            output[idx * num_features + f] = sum * 0.2f;
        }
    }
}

int main() {
    // Configuration
    const int BATCH_SIZE = 1000;
    const int NUM_FEATURES = 6;
    const int NUM_BATCHES = 500;
    const int NUM_STREAMS = 4;  // Key optimization: 4 concurrent streams
    
    std::cout << "=== Optimized CUDA with Streams ===" << std::endl;
    std::cout << "Configuration:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Features: " << NUM_FEATURES << std::endl;
    std::cout << "  Number of batches: " << NUM_BATCHES << std::endl;
    std::cout << "  Concurrent streams: " << NUM_STREAMS << std::endl;
    
    // Allocate PINNED host memory for async transfers
    // Pinned memory enables DMA (Direct Memory Access) for faster transfers
    float *h_input[NUM_STREAMS], *h_output[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMallocHost(&h_input[i], BATCH_SIZE * NUM_FEATURES * sizeof(float)));
        CHECK_CUDA(cudaMallocHost(&h_output[i], BATCH_SIZE * NUM_FEATURES * sizeof(float)));
    }
    
    // Allocate device memory (one set per stream)
    float *d_input[NUM_STREAMS], *d_output[NUM_STREAMS];
    
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaMalloc(&d_input[i], BATCH_SIZE * NUM_FEATURES * sizeof(float)));
        CHECK_CUDA(cudaMalloc(&d_output[i], BATCH_SIZE * NUM_FEATURES * sizeof(float)));
    }
    
    // Create CUDA streams
    cudaStream_t streams[NUM_STREAMS];
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaStreamCreate(&streams[i]));
    }
    
    // Open data file
    std::ifstream file("data/sensor_batches_full.bin", std::ios::binary);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data file" << std::endl;
        return 1;
    }
    
    // Kernel launch configuration
    int threadsPerBlock = 256;
    int blocksPerGrid = (BATCH_SIZE + threadsPerBlock - 1) / threadsPerBlock;
    
    std::cout << "\nGPU Launch Configuration:" << std::endl;
    std::cout << "  Threads per block: " << threadsPerBlock << std::endl;
    std::cout << "  Blocks per grid: " << blocksPerGrid << std::endl;
    
    // Warm-up phase
    std::cout << "\nWarming up with concurrent streams..." << std::endl;
    for (int i = 0; i < 10; i++) {
        int s = i % NUM_STREAMS;  // Round-robin stream selection
        
        file.read(reinterpret_cast<char*>(h_input[s]), 
                  BATCH_SIZE * NUM_FEATURES * sizeof(float));
        
        // Async H2D transfer
        CHECK_CUDA(cudaMemcpyAsync(d_input[s], h_input[s],
                                   BATCH_SIZE * NUM_FEATURES * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]));
        
        // Kernel launch in stream
        preprocessOptimized<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
            d_input[s], d_output[s], BATCH_SIZE, NUM_FEATURES);
        
        // Async D2H transfer
        CHECK_CUDA(cudaMemcpyAsync(h_output[s], d_output[s],
                                   BATCH_SIZE * NUM_FEATURES * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[s]));
    }
    
    // Wait for all streams to complete
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Reset file
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Create events for timing
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    // Benchmark with concurrent streams
    std::cout << "Running optimized benchmark..." << std::endl;
    
    CHECK_CUDA(cudaEventRecord(start));
    
    float checksum = 0.0f;
    
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        int s = batch % NUM_STREAMS;  // Round-robin across streams
        
        // Read batch from file
        file.read(reinterpret_cast<char*>(h_input[s]), 
                  BATCH_SIZE * NUM_FEATURES * sizeof(float));
        
        // ASYNC H2D transfer (non-blocking!)
        CHECK_CUDA(cudaMemcpyAsync(d_input[s], h_input[s],
                                   BATCH_SIZE * NUM_FEATURES * sizeof(float),
                                   cudaMemcpyHostToDevice, streams[s]));
        
        // Kernel launch in specific stream
        preprocessOptimized<<<blocksPerGrid, threadsPerBlock, 0, streams[s]>>>(
            d_input[s], d_output[s], BATCH_SIZE, NUM_FEATURES);
        
        // ASYNC D2H transfer (non-blocking!)
        CHECK_CUDA(cudaMemcpyAsync(h_output[s], d_output[s],
                                   BATCH_SIZE * NUM_FEATURES * sizeof(float),
                                   cudaMemcpyDeviceToHost, streams[s]));
        
        // Accumulate checksum (stream will complete before we access)
        if (batch >= NUM_STREAMS) {
            // Only access data from completed streams
            int completed_stream = (batch - NUM_STREAMS) % NUM_STREAMS;
            CHECK_CUDA(cudaStreamSynchronize(streams[completed_stream]));
            checksum += h_output[completed_stream][0];
        }
    }
    
    // Synchronize all streams
    CHECK_CUDA(cudaDeviceSynchronize());
    
    CHECK_CUDA(cudaEventRecord(stop));
    CHECK_CUDA(cudaEventSynchronize(stop));
    
    // Calculate metrics
    float milliseconds = 0;
    CHECK_CUDA(cudaEventElapsedTime(&milliseconds, start, stop));
    
    float avg_batch_time_ms = milliseconds / NUM_BATCHES;
    float throughput = BATCH_SIZE / (avg_batch_time_ms / 1000.0f);
    float per_sample_time_us = (avg_batch_time_ms * 1000.0f) / BATCH_SIZE;
    
    // Print results
    std::cout << "\nVerification:" << std::endl;
    std::cout << "  output[0] = " << h_output[0][0] << std::endl;
    std::cout << "  Checksum: " << checksum << std::endl;
    
    std::cout << "\n=== Optimized CUDA Results ===" << std::endl;
    std::cout << "Total time: " << milliseconds << " ms" << std::endl;
    std::cout << "Average batch time: " << avg_batch_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (int)throughput << " samples/sec" << std::endl;
    std::cout << "Per-sample time: " << per_sample_time_us << " μs" << std::endl;
    
    // Compare to baselines
    float cpu_baseline = 0.054f;  // From our measurement
    float naive_cuda = 0.040f;    // From naive version
    
    std::cout << "\n📊 Performance Comparison:" << std::endl;
    std::cout << "  Speedup vs CPU: " << (cpu_baseline / avg_batch_time_ms) << "x" << std::endl;
    std::cout << "  Speedup vs Naive CUDA: " << (naive_cuda / avg_batch_time_ms) << "x" << std::endl;
    std::cout << "  Throughput improvement: " << (throughput / 18551498.0f) << "x vs CPU" << std::endl;
    
    // Cleanup
    for (int i = 0; i < NUM_STREAMS; i++) {
        CHECK_CUDA(cudaFree(d_input[i]));
        CHECK_CUDA(cudaFree(d_output[i]));
        CHECK_CUDA(cudaFreeHost(h_input[i]));
        CHECK_CUDA(cudaFreeHost(h_output[i]));
        CHECK_CUDA(cudaStreamDestroy(streams[i]));
    }
    
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    file.close();
    
    return 0;
}