#include <iostream>
#include <fstream>
#include <vector>
#include <chrono>
#include <cmath>

void preprocessCPU(const float* input, float* output, int num_samples, int num_channels) {
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
    const int NUM_CHANNELS = 8;
    const int SAMPLES_PER_FILE = 20480;
    const int NUM_FILES = 100;
    
    std::vector<float> input(SAMPLES_PER_FILE * NUM_CHANNELS);
    std::vector<float> output(NUM_CHANNELS);
    
    std::ifstream file("data_nasa_bearing/processed/bearing_data.bin", std::ios::binary);
    if (!file) {
        std::cerr << "Error: Cannot open bearing_data.bin" << std::endl;
        return 1;
    }
    
    // Warm-up
    for (int i = 0; i < 10; ++i) {
        file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        preprocessCPU(input.data(), output.data(), SAMPLES_PER_FILE, NUM_CHANNELS);
    }
    file.seekg(0);
    
    // Benchmark
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int batch = 0; batch < NUM_FILES; ++batch) {
        file.read(reinterpret_cast<char*>(input.data()), input.size() * sizeof(float));
        preprocessCPU(input.data(), output.data(), SAMPLES_PER_FILE, NUM_CHANNELS);
    }
    
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
    
    float avg_time_ms = duration.count() / (float)NUM_FILES / 1000.0f;
    float throughput = SAMPLES_PER_FILE / (avg_time_ms / 1000.0f);
    
    // Output in parseable format
    std::cout << "CPU_BASELINE," << avg_time_ms << "," << (int)throughput << std::endl;
    
    file.close();
    return 0;
}
