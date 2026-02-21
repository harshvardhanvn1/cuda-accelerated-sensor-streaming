#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

struct DataStats {
    int original_samples = 0;
    int valid_samples = 0;
};

// Process a single subject file
DataStats processSubjectFile(const std::string& filepath, std::vector<float>& all_data) {
    DataStats stats;
    
    std::ifstream infile(filepath);
    if (!infile.is_open()) {
        std::cerr << "Warning: Could not open " << filepath << std::endl;
        return stats;
    }
    
    std::string line;
    while (std::getline(infile, line)) {
        stats.original_samples++;
        
        // Parse line
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;
        
        while (iss >> token) {
            tokens.push_back(token);
        }
        
        // Need at least 13 columns
        if (tokens.size() < 13) {
            continue;
        }
        
        // Extract columns: 4,5,6 (accel) and 10,11,12 (gyro)
        int indices[6] = {4, 5, 6, 10, 11, 12};
        float values[6];
        bool has_nan = false;
        
        for (int i = 0; i < 6; i++) {
            if (tokens[indices[i]] == "NaN") {
                has_nan = true;
                break;
            }
            values[i] = std::stof(tokens[indices[i]]);
        }
        
        if (has_nan) {
            continue;
        }
        
        // Add valid data
        for (int i = 0; i < 6; i++) {
            all_data.push_back(values[i]);
        }
        stats.valid_samples++;
    }
    
    infile.close();
    return stats;
}

int main() {
    const std::string protocol_dir = "data/Protocol";
    const std::string output_file = "data/sensor_batches_full.bin";
    const int batch_size = 1000;
    const int num_features = 6;
    
    std::cout << "=== Preparing FULL PAMAP2 Dataset ===" << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Features: " << num_features << std::endl;
    
    // Find all .dat files
    std::vector<std::string> subject_files;
    
    for (const auto& entry : fs::directory_iterator(protocol_dir)) {
        if (entry.path().extension() == ".dat") {
            subject_files.push_back(entry.path().string());
        }
    }
    
    // Sort files for consistent order
    std::sort(subject_files.begin(), subject_files.end());
    
    std::cout << "\nFound " << subject_files.size() << " subject files:" << std::endl;
    for (const auto& file : subject_files) {
        std::cout << "  - " << fs::path(file).filename().string() << std::endl;
    }
    
    // Storage for all data
    std::vector<float> all_sensor_data;
    all_sensor_data.reserve(4000000 * num_features);  // Pre-allocate ~4M samples
    
    int total_original = 0;
    int total_valid = 0;
    
    // Process each subject file
    for (const auto& filepath : subject_files) {
        std::cout << "\nLoading " << fs::path(filepath).filename().string() << "..." << std::endl;
        
        DataStats stats = processSubjectFile(filepath, all_sensor_data);
        
        std::cout << "  Original samples: " << stats.original_samples << std::endl;
        std::cout << "  After removing NaN: " << stats.valid_samples << std::endl;
        
        total_original += stats.original_samples;
        total_valid += stats.valid_samples;
    }
    
    std::cout << "\nCombining all subjects..." << std::endl;
    std::cout << "Total combined samples: " << total_valid << std::endl;
    
    // Calculate batches
    int num_batches = total_valid / batch_size;
    int total_samples = num_batches * batch_size;
    
    // Trim to exact batches
    all_sensor_data.resize(total_samples * num_features);
    
    // Write binary file
    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create " << output_file << std::endl;
        return 1;
    }
    
    outfile.write(reinterpret_cast<char*>(all_sensor_data.data()),
                  all_sensor_data.size() * sizeof(float));
    outfile.close();
    
    // Get file size
    auto file_size = fs::file_size(output_file);
    
    std::cout << "\n Full dataset prepared successfully!" << std::endl;
    std::cout << "   Total subjects: " << subject_files.size() << std::endl;
    std::cout << "   Total batches: " << num_batches << std::endl;
    std::cout << "   Total samples: " << total_samples << std::endl;
    std::cout << "   Features per sample: " << num_features << std::endl;
    std::cout << "   File size: " << (file_size / (1024.0 * 1024.0)) << " MB" << std::endl;
    std::cout << "   Saved to: " << output_file << std::endl;
    
    return 0;
}