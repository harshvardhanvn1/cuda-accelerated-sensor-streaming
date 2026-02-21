#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>

// Simple preprocessing: compute RMS (Root Mean Square) for each channel
void preprocessCPU(const float* input, float* output, int num_samples, int num_channels) {
    // Compute RMS for each channel
    for (int ch = 0; ch < num_channels; ++ch) {
        float sum = 0.0f;
        for (int i = 0; i < num_samples; ++i) {
            float val = input[i * num_channels + ch];
            sum += val * val;
        }
        output[ch] = std::sqrt(sum / num_samples);
    }
}

int main() {
    // Configuration
    const int NUM_CHANNELS = 8;
    const int SAMPLES_PER_FILE = 20480;
    const int NUM_FILES = 100; // Process first 100 files for benchmark
    const int BATCH_SIZE = SAMPLES_PER_FILE;
    
    // Allocate memory
    std::vector<float> input(BATCH_SIZE * NUM_CHANNELS);
    std::vector<float> output(NUM_CHANNELS);
    
    // Open binary file
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open bearing_data.bin" << std::endl;
        return 1;
    }
    
    // Warm-up
    std::cout << "Warming up..." << std::endl;
    for (int i = 0; i < 10; ++i) {
        file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        preprocessCPU(input.data(), output.data(), BATCH_SIZE, NUM_CHANNELS);
    }
    file.seekg(0);
    
    // Benchmark
    std::cout << "\nBenchmarking CPU baseline..." << std::endl;
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int batch = 0; batch < NUM_FILES; ++batch) {
        file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        preprocessCPU(input.data(), output.data(), BATCH_SIZE, NUM_CHANNELS);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time_ms = duration.count() / (float)NUM_FILES / 1000.0f;
    float throughput = BATCH_SIZE / (avg_time_ms / 1000.0f);
    
    std::cout << "\n=== CPU BASELINE RESULTS ===" << std::endl;
    std::cout << "Files processed: " << NUM_FILES << std::endl;
    std::cout << "Avg time per file: " << avg_time_ms << " ms" << std::endl;
    std::cout << "Throughput: " << (int)throughput << " samples/sec" << std::endl;
    std::cout << "Per-sample time: " << avg_time_ms / BATCH_SIZE << " ms" << std::endl;
    
    std::cout << "\nSample RMS values (first file):" << std::endl;
    for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
        std::cout << "  Channel " << ch << ": " << output[ch] << std::endl;
    }
    
    file.close();
    return 0;
}
