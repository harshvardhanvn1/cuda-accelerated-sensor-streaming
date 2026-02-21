#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>
#include <algorithm>

// Optimized preprocessing with proper algorithms
void preprocessCPU(const float* input, float* output, int num_samples, int num_features) {
    // Step 1: Compute actual mean and std for each feature
    std::vector<float> mean(num_features, 0.0f);
    std::vector<float> stddev(num_features, 0.0f);
    
    // Calculate mean
    for (int f = 0; f < num_features; f++) {
        for (int i = 0; i < num_samples; i++) {
            mean[f] += input[i * num_features + f];
        }
        mean[f] /= num_samples;
    }
    
    // Calculate standard deviation
    for (int f = 0; f < num_features; f++) {
        for (int i = 0; i < num_samples; i++) {
            float diff = input[i * num_features + f] - mean[f];
            stddev[f] += diff * diff;
        }
        stddev[f] = std::sqrt(stddev[f] / num_samples);
        // Avoid division by zero
        if (stddev[f] < 1e-6f) stddev[f] = 1.0f;
    }
    
    // Normalize using computed statistics
    for (int i = 0; i < num_samples; i++) {
        for (int f = 0; f < num_features; f++) {
            int idx = i * num_features + f;
            output[idx] = (input[idx] - mean[f]) / stddev[f];
        }
    }
    
    // Step 2: Moving average with sliding window optimization
    const int WINDOW_SIZE = 5;
    
    for (int f = 0; f < num_features; f++) {
        // Compute initial window sum
        float window_sum = 0.0f;
        for (int w = 0; w < WINDOW_SIZE; w++) {
            window_sum += output[w * num_features + f];
        }
        
        // Store first average
        float temp = output[0 * num_features + f];
        output[0 * num_features + f] = window_sum / WINDOW_SIZE;
        
        // Slide the window through the data
        for (int i = 1; i < num_samples - WINDOW_SIZE + 1; i++) {
            // Remove oldest value, add newest value
            window_sum = window_sum 
                       - output[(i - 1) * num_features + f]
                       + output[(i + WINDOW_SIZE - 1) * num_features + f];
            
            float next_temp = output[i * num_features + f];
            output[i * num_features + f] = window_sum / WINDOW_SIZE;
        }
    }
}

int main() {
    // Configuration
    const int BATCH_SIZE = 1000;
    const int NUM_FEATURES = 6;
    const int NUM_BATCHES = 500; 
    
    std::cout << "=== Optimized CPU Baseline Benchmark ===" << std::endl;
    std::cout << "Optimizations:" << std::endl;
    std::cout << "  - Sliding window for moving average (O(n) vs O(n*w))" << std::endl;
    std::cout << "  - Proper z-score normalization" << std::endl;
    std::cout << "  - Cache-friendly memory access patterns" << std::endl;
    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Batch size: " << BATCH_SIZE << std::endl;
    std::cout << "  Features: " << NUM_FEATURES << std::endl;
    std::cout << "  Number of batches: " << NUM_BATCHES << std::endl;
    
    // Allocate memory
    std::vector<float> input(BATCH_SIZE * NUM_FEATURES);
    std::vector<float> output(BATCH_SIZE * NUM_FEATURES);
    
    // Open binary data file
    std::ifstream file("data/sensor_batches_full.bin", std::ios::binary);  // Changed filename
    if (!file.is_open()) {
        std::cerr << "Error: Could not open data/sensor_batches_full.bin" << std::endl;
        return 1;
    }
    
    // Warm-up phase
    std::cout << "\nWarming up CPU caches..." << std::endl;
    for (int i = 0; i < 10; i++) {
        file.read(reinterpret_cast<char*>(input.data()), 
                  input.size() * sizeof(float));
        preprocessCPU(input.data(), output.data(), BATCH_SIZE, NUM_FEATURES);
    }
    
    // Reset file pointer
    file.clear();
    file.seekg(0, std::ios::beg);
    
    // Actual benchmark
    std::cout << "Running benchmark..." << std::endl;
    
    auto start = std::chrono::high_resolution_clock::now();
    
    float checksum = 0.0f;  // Prevent compiler optimization
    
    for (int batch = 0; batch < NUM_BATCHES; batch++) {
        // Read one batch
        file.read(reinterpret_cast<char*>(input.data()), 
                  input.size() * sizeof(float));
        
        // Process the batch
        preprocessCPU(input.data(), output.data(), BATCH_SIZE, NUM_FEATURES);
        
        // Use output to prevent dead code elimination
        checksum += output[0] + output[BATCH_SIZE * NUM_FEATURES - 1];
    }
    
    auto end = std::chrono::high_resolution_clock::now();

    // Verify computation actually happened
    std::cout << "\nVerification (first 5 output values):" << std::endl;
    for (int i = 0; i < 5; i++) {
        std::cout << "  output[" << i << "] = " << output[i] << std::endl;
    }
    std::cout << "Checksum: " << checksum << std::endl;
    
    // Calculate metrics
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    float total_time_ms = duration.count() / 1000.0f;
    float avg_batch_time_ms = total_time_ms / NUM_BATCHES;
    float throughput = BATCH_SIZE / (avg_batch_time_ms / 1000.0f);
    float per_sample_time_us = (avg_batch_time_ms * 1000.0f) / BATCH_SIZE;
    
    // Print results
    std::cout << "\n=== Optimized CPU Baseline Results ===" << std::endl;
    std::cout << "Total time: " << total_time_ms << " ms" << std::endl;
    std::cout << "Average batch time: " << avg_batch_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (int)throughput << " samples/sec" << std::endl;
    std::cout << "Per-sample time: " << per_sample_time_us << " μs" << std::endl;
    std::cout << "\n📊 This is our HONEST baseline to beat with GPU!" << std::endl;
    
    // Prevent optimization of checksum
    if (checksum > 1e10f) {
        std::cout << "Checksum: " << checksum << std::endl;
    }
    
    file.close();
    
    return 0;
}