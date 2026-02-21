#include <iostream>
#include <fstream>
#include <vector>
#include <string>
#include <sstream>
#include <filesystem>
#include <algorithm>

namespace fs = std::filesystem;

int main() {
    std::cout << "=== NASA Bearing Data Preprocessing ===" << std::endl;
    
    const std::string input_dir = "data_nasa_bearing/raw/1st_test/1st_test/";
    const std::string output_file = "data_nasa_bearing/processed/bearing_data.bin";
    const int NUM_CHANNELS = 8;
    const int SAMPLES_PER_FILE = 20480;
    
    // Create output directory
    fs::create_directories("data_nasa_bearing/processed");
    
    // Get all files and sort them
    std::vector<std::string> files;
    for (const auto& entry : fs::directory_iterator(input_dir)) {
        if (entry.is_regular_file()) {
            files.push_back(entry.path().string());
        }
    }
    std::sort(files.begin(), files.end());
    
    std::cout << "Found " << files.size() << " files" << std::endl;
    
    // Open output file
    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile) {
        std::cerr << "Error: Cannot create output file" << std::endl;
        return 1;
    }
    
    // Process each file
    std::vector<float> buffer(SAMPLES_PER_FILE * NUM_CHANNELS);
    int processed = 0;
    
    for (const auto& filepath : files) {
        std::ifstream infile(filepath);
        if (!infile) {
            std::cerr << "Warning: Cannot open " << filepath << std::endl;
            continue;
        }
        
        // Read all values
        int idx = 0;
        std::string line;
        while (std::getline(infile, line) && idx < SAMPLES_PER_FILE) {
            std::istringstream iss(line);
            for (int ch = 0; ch < NUM_CHANNELS; ++ch) {
                float val;
                if (iss >> val) {
                    buffer[idx * NUM_CHANNELS + ch] = val;
                }
            }
            idx++;
        }
        
        // Write to binary file
        outfile.write(reinterpret_cast<const char*>(buffer.data()), 
                      buffer.size() * sizeof(float));
        
        processed++;
        if (processed % 100 == 0) {
            std::cout << "Processed " << processed << " files..." << std::endl;
        }
    }
    
    outfile.close();
    
    std::cout << "\n✓ Preprocessing complete!" << std::endl;
    std::cout << "Files processed: " << processed << std::endl;
    std::cout << "Total samples: " << processed * SAMPLES_PER_FILE << std::endl;
    std::cout << "Output: " << output_file << std::endl;
    
    // Get file size
    std::cout << "File size: " << (fs::file_size(output_file) / (1024.0 * 1024.0)) 
              << " MB" << std::endl;
    
    return 0;
}
