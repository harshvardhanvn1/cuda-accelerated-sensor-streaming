#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>
#include <string>
#include <cmath>

int main() {
    //Configuration
    const std::string input_file = "data/subject101.dat";
    const std::string output_file = "data/sensor_batches_cpp.bin";
    const int batch_size = 1000;
    const int num_features = 6;

    //Storage for sensor data
    std::vector<float> sensor_data;

    //Open the input file
    std::ifstream infile(input_file);
    if(!infile.is_open()) {
        std::cerr << "Error: courl not open" << input_file << std::endl;
        return 1;
    }

    std::cout << " Loading" << input_file << "..." << std::endl;

    // Read file line by line
    std::string line;
    int original_count = 0;
    int valid_count = 0;
    while (std::getline(infile, line)) {
        original_count++ ;

        //Parse the line
        std::istringstream iss(line);
        std::vector<std::string> tokens;
        std::string token;

        //Split line by spaces
        while (iss >> token) {
            tokens.push_back(token);
        }

        //We need atleast 13 columns (indices 0 - 12)
        if (tokens.size() < 13) {
            continue;
        }

        //Extract columns : 4,5,6 (accel) and 10,11,12 (gyro) - 0-indexed
        int indices[6] = {4,5,6,10,11,12};
        float values[6];
        bool has_nan = false;

        // Parse the 6 values we nees
        for (int i = 0; i<6; i++) {
            if (tokens[indices[i]] == "NaN") {
                has_nan = true;
                break;
            }
            values[i] = std::stof(tokens[indices[i]]);
        }

        // Skip rows with NaN
        if (has_nan) {
            continue;
        }

        //Add valid data
        for (int i = 0; i < 6; i++) {
            sensor_data.push_back(values[i]);
        }
        valid_count ++;
    }
    infile.close();
    std::cout << "Original samples: " << original_count << std::endl;
    std::cout << "After removing NaN: " << valid_count << std::endl;

    //Calculate batches
    int num_batches = valid_count / batch_size;
    int total_samples = num_batches * batch_size;

    //Trim to exact batches
    sensor_data.resize(total_samples * num_features);
    
    //Write binary file
    std::ofstream outfile(output_file, std::ios::binary);
    if (!outfile.is_open()) {
        std::cerr << "Error: Could not create " << output_file << std::endl;
        return 1;
    }

    outfile.write(reinterpret_cast<char*>(sensor_data.data()),
                sensor_data.size() * sizeof(float));
    outfile.close();

    std::cout << "\n Data prepared successfully!" << std::endl;
    std::cout << "   Batches: " << num_batches << std::endl;
    std::cout << "   Total samples: " << total_samples << std::endl;
    std::cout << "   Features per sample: " << num_features << std::endl;
    std::cout << "   Saved to: " << output_file << std::endl;
    
    return 0;


}