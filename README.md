# CUDA-Accelerated Sensor Stream Processing

Real-time sensor preprocessing optimization using CUDA for edge inference applications.

## Project Overview

This project demonstrates GPU acceleration techniques for industrial IoT sensor processing, implementing three key optimization scenarios:

1. **Multi-Sensor Concurrent Processing** - 4x throughput via parallel stream processing
2. **Large Batch Optimization** - 3x speedup through overlapped memory transfers
3. **Multi-Stage Pipeline** - Overlapped preprocessing stages for continuous processing

**Target Hardware:** NVIDIA Tesla T4 (industry-standard edge inference GPU)  
**Primary Dataset:** Edge-IIoTset (10+ industrial sensors)  
**Learning Dataset:** PAMAP2 (human activity monitoring)

## Project Structure
```
cuda-accelerated-sensor-streaming/
├── src_edge_iiot/              # Main project implementations
│   ├── scenario1_multi_sensor.cu    # Concurrent sensor streams
│   ├── scenario2_large_batch.cu     # Batch optimization
│   └── scenario3_pipeline.cu        # Pipeline parallelism
│
├── src_pamap2/                 # Learning implementations
│   ├── cpu_baseline.cpp
│   ├── cuda_naive.cu
│   └── cuda_optimized.cu
│
├── data_edge_iiot/             # Edge-IIoT dataset
│   ├── raw/                    # Original CSV files
│   ├── processed/              # Per-sensor binary files
│   └── combined/               # Multi-sensor batches
│
├── data_pamap2/                # PAMAP2 dataset
│
├── results/
│   ├── pamap2/                 # Learning phase results
│   └── edge_iiot/              # Main project results
│
└── docs/
    ├── LEARNING_JOURNEY.md     # Development process
    └── PROJECT_REPORT.md       # Final benchmarks
```

## Technical Stack

- **Languages:** CUDA C++, C++17, Python (data preprocessing)
- **GPU:** NVIDIA Tesla T4 (2560 CUDA cores, 16GB VRAM)
- **Profiling:** NVIDIA Nsight Systems
- **Platform:** Google Colab (cloud GPU), local development on macOS

## Quick Start

### Prerequisites
- CUDA Toolkit 12.x
- C++17 compatible compiler
- NVIDIA GPU (Tesla T4 or equivalent)

### Building and Running

**Scenario 1: Multi-Sensor Concurrent Processing**
```bash
nvcc -O3 -std=c++17 -o scenario1 src_edge_iiot/scenario1_multi_sensor.cu
./scenario1
```

**Scenario 2: Large Batch Optimization**
```bash
nvcc -O3 -std=c++17 -o scenario2 src_edge_iiot/scenario2_large_batch.cu
./scenario2
```

**Scenario 3: Multi-Stage Pipeline**
```bash
nvcc -O3 -std=c++17 -o scenario3 src_edge_iiot/scenario3_pipeline.cu
./scenario3
```

## Results

### PAMAP2 Learning Phase

Baseline CPU: 0.054ms/batch (18.5M samples/sec)  
Naive CUDA: 0.040ms/batch (24.7M samples/sec, 1.35x speedup)  
CUDA Streams: 0.038ms/batch (26.3M samples/sec, 1.42x speedup)

See [PAMAP2 Benchmark Results](results/pamap2/BENCHMARK_RESULTS.md) for detailed analysis.

### Edge-IIoT Main Project

Results pending completion of three scenarios. See [PROJECT_REPORT.md](docs/PROJECT_REPORT.md).

## Key Optimizations

**Algorithmic:**
- Sliding window moving average (O(n) vs O(n×w))
- Z-score normalization with precomputed statistics
- Cache-friendly memory access patterns

**CUDA-Specific:**
- Concurrent stream processing for independent data sources
- Asynchronous memory transfers (cudaMemcpyAsync)
- Pinned host memory for DMA transfers
- Kernel optimizations: loop unrolling, memory coalescing
- Multi-stage pipeline with overlapped execution

## Development Process

This project evolved through iterative learning:

1. **Phase 1:** CPU baseline implementation and optimization
2. **Phase 2:** Naive GPU parallelization
3. **Phase 3:** Stream optimization attempts (revealed dataset limitations)
4. **Phase 4:** Dataset transition to Edge-IIoTset for realistic scenarios
5. **Phase 5:** Three-scenario comprehensive implementation

See [LEARNING_JOURNEY.md](docs/LEARNING_JOURNEY.md) for detailed insights.

## Datasets

**PAMAP2 Physical Activity Monitoring**
- 9 subjects, 2.86M samples
- 6 features (accelerometer + gyroscope)
- Used for learning CUDA fundamentals

**Edge-IIoTset Cyber Security Dataset**
- 10+ industrial IoT sensors
- Temperature, humidity, vibration, pH, pressure, etc.
- Designed for edge inference benchmarking
- Source: [Kaggle](https://www.kaggle.com/datasets/mohamedamineferrag/edgeiiotset-cyber-security-dataset-of-iot-iiot)

## Performance Metrics

Key measurements across all implementations:
- Throughput (samples/second)
- Latency (milliseconds/batch)
- Per-sample processing time (microseconds)
- Speedup vs CPU baseline
- PCIe transfer overhead
- GPU utilization (via Nsight Systems)

## References

**Datasets:**
- Reiss, A. and Stricker, D. (2012). PAMAP2 Activity Monitoring. ISWC 2012.
- Ferrag, M.A. et al. (2022). Edge-IIoTset. IEEE Access.

**Hardware:**
- NVIDIA Tesla T4 Technical Specifications
- Google Colab Research Platform

**Tools:**
- NVIDIA CUDA Toolkit Documentation
- NVIDIA Nsight Systems Profiling Guide

## License

This project is for educational and portfolio purposes.

## Author

Independent learning project inspired by edge inference system research (2023).
