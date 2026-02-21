# CUDA-Accelerated Sensor Stream Processing

Industrial bearing vibration analysis system optimized with CUDA for edge inference applications.

## Project Overview

High-performance GPU-accelerated preprocessing pipeline for industrial sensor data, achieving **3.27× speedup** over CPU baseline through systematic CUDA optimization. Demonstrates production-ready techniques for edge inference on professional GPUs.

**Hardware:** NVIDIA RTX A4000 (16GB, Ampere Architecture)  
**Dataset:** NASA IMS Bearing Dataset (44M samples, 2.4GB)  
**Best Performance:** 3.27× speedup via batching optimization

## Key Results

| Implementation | Time/File | Throughput | Speedup vs CPU | Speedup vs Naive |
|----------------|-----------|------------|----------------|------------------|
| CPU Baseline | 0.341 ms | 60.1 M/s | 1.00× | - |
| Naive CUDA | 0.246 ms | 83.3 M/s | 1.38× | 1.00× |
| **Scenario 1: Concurrent** | 0.124 ms | 165.6 M/s | **2.75×** | **1.98×** |
| **Scenario 2: Large Batch** | 0.104 ms | 196.9 M/s | **3.27×** | **2.36×** ⭐ |
| **Scenario 3: Pipeline** | 0.121 ms | 169.3 M/s | **2.81×** | **2.03×** |
| Combined | 0.110 ms | 185.7 M/s | 3.08× | 2.22× |

## Architecture

### Three Optimization Scenarios

**Scenario 1: Multi-Bearing Concurrent Processing**
- 4 concurrent CUDA streams (one per bearing)
- Parallel kernel execution across independent sensors
- Async memory transfers with pinned memory
- **Result:** 2.75× speedup

**Scenario 2: Large Batch Optimization** ⭐ **BEST**
- Batch size 5 files (102K samples per transfer)
- Overlapped H2D, kernel execution, D2H transfers
- Amortized PCIe overhead
- **Result:** 3.27× speedup (winner!)

**Scenario 3: Multi-Stage Pipeline**
- 3-stage pipeline: Normalize → RMS → Peak Detection
- Stream-based execution overlap
- Demonstrates software pipelining
- **Result:** 2.81× speedup

## Project Structure
```
cuda-accelerated-sensor-streaming/
├── src_nasa_bearing/
│   ├── baseline/
│   │   ├── cpu_baseline.cpp           # CPU reference implementation
│   │   └── cuda_naive.cu              # Simple GPU parallelization
│   ├── optimized/
│   │   ├── scenario1_concurrent.cu    # Multi-stream concurrent
│   │   ├── scenario2_batch.cu         # Batch optimization
│   │   ├── scenario3_pipeline.cu      # Pipeline parallelism
│   │   └── scenario_combined.cu       # All techniques combined
│   ├── preprocess_data.cpp            # ASCII to binary converter
│   ├── detailed_profiling.cu          # Timing breakdown utility
│   └── profiling_optimized.cu         # Profiling helper
│
├── data_nasa_bearing/
│   ├── raw/                           # Original NASA dataset
│   └── processed/
│       └── bearing_data.bin           # Preprocessed binary (1.35GB)
│
├── results/nasa_bearing/
│   ├── benchmark_results.txt          # Performance comparison
│   ├── PROFILING_ANALYSIS.md          # Comprehensive profiling insights
│   └── NSIGHT_PROFILING_SUMMARY.md    # Nsight Systems analysis
│
├── profiling/                         # Nsight profiling reports (.nsys-rep)
├── run_all_benchmarks.sh              # Master benchmark script
└── docs/
    └── NASA_IMPLEMENTATION_PLAN.md    # Project roadmap
```

## Quick Start

### Prerequisites
```bash
# CUDA Toolkit 12.x
nvcc --version

# RTX A4000 or equivalent professional GPU
nvidia-smi
```

### Running Benchmarks

**Option 1: Run all scenarios (recommended)**
```bash
chmod +x run_all_benchmarks.sh
./run_all_benchmarks.sh
```

**Option 2: Run individual implementations**
```bash
# CPU Baseline
g++ -O3 -std=c++17 -o cpu_baseline src_nasa_bearing/baseline/cpu_baseline.cpp
./cpu_baseline

# Naive CUDA
nvcc -O3 -o cuda_naive src_nasa_bearing/baseline/cuda_naive.cu
./cuda_naive

# Scenario 1: Concurrent Streams
nvcc -O3 -o scenario1_concurrent src_nasa_bearing/optimized/scenario1_concurrent.cu
./scenario1_concurrent

# Scenario 2: Large Batch (Best Performance)
nvcc -O3 -o scenario2_batch src_nasa_bearing/optimized/scenario2_batch.cu
./scenario2_batch

# Scenario 3: Pipeline
nvcc -O3 -o scenario3_pipeline src_nasa_bearing/optimized/scenario3_pipeline.cu
./scenario3_pipeline
```

## Dataset

**NASA IMS Bearing Dataset (1st Test)**
- **Source:** [NASA Ames Prognostics Center](https://www.kaggle.com/datasets/vinayak123tyagi/bearing-dataset)
- **Size:** 2,156 files, 44.1M samples, 2.4GB
- **Sensors:** 8 channels (4 bearings × 2 accelerometers)
- **Sampling:** 20,480 samples/file at 20 kHz
- **Purpose:** Run-to-failure bearing vibration monitoring

### Preprocessing

Convert ASCII data to binary format:
```bash
g++ -O3 -std=c++17 -o preprocess_data src_nasa_bearing/preprocess_data.cpp
./preprocess_data
```

Output: `data_nasa_bearing/processed/bearing_data.bin` (1.35 GB)

## Key Optimizations

### Memory Management
- **Pinned memory** (`cudaMallocHost`) for DMA transfers
- **Async transfers** (`cudaMemcpyAsync`) for non-blocking operations
- **Memory coalescing** for efficient global memory access

### Execution Strategies
- **CUDA streams** for concurrent kernel execution
- **Batching** to amortize PCIe overhead
- **Pipeline parallelism** for multi-stage workflows

### Profiling Insights

**Naive CUDA bottleneck:**
- PCIe transfers: **97.9%** of GPU time
- Kernel execution: 2.0%
- D2H transfer: 0.1%

**Optimization impact:**
- Scenario 2 reduces transfer overhead via batching
- Fewer kernel launches (25 vs 110)
- Better PCIe utilization (99.5% vs 97.9%)

See [PROFILING_ANALYSIS.md](results/nasa_bearing/PROFILING_ANALYSIS.md) for detailed breakdown.

## Performance Metrics

**Measured using:**
- CUDA Events for GPU timing
- `std::chrono` for CPU timing
- Nsight Systems for profiling

**Key findings:**
1. **Naive CUDA can be slower than CPU** (on some GPUs like T4)
2. **PCIe is the bottleneck** (70% of execution time in naive)
3. **Batching wins** (simplicity + performance)
4. **More complexity ≠ better** (Combined scenario shows overhead)

## Technical Deep Dive

### Why Scenario 2 Wins

**Batching advantages:**
```
Naive:  110 transfers × 655 KB  = 72.1 MB total, high overhead
Batch:   25 transfers × 3.28 MB = 82.0 MB total, low overhead
```

**Fewer kernel launches:**
```
Naive: 110 launches × 19.2 μs = 2.11 ms kernel time
Batch:  25 launches × 24.1 μs = 0.60 ms kernel time
```

**Result:** 3.27× faster than CPU, 2.36× faster than naive CUDA

### Why Combined Isn't Best

Combined scenario uses 12 streams with pipeline complexity:
- Stream management overhead
- Context switching cost
- Complexity without proportional benefit

**Lesson:** Match optimization technique to problem characteristics.

## Profiling

Profile with Nsight Systems (generates .nsys-rep files):
```bash
nsys profile -o profiling/scenario2_batch --stats=true ./scenario2_batch
```

View in Nsight Systems GUI:
```bash
nsys-ui profiling/scenario2_batch.nsys-rep
```

## Hardware Details

**NVIDIA RTX A4000**
- Architecture: Ampere (GA104)
- CUDA Cores: 6,144
- VRAM: 16 GB GDDR6
- Memory Bandwidth: 448 GB/s
- Use Case: Professional workstation, edge inference

## References

**Dataset:**
- NASA Ames Prognostics Center: [IMS Bearing Dataset](https://www.nasa.gov/content/prognostics-center-of-excellence-data-set-repository)

**Tools:**
- NVIDIA CUDA Toolkit 12.x
- NVIDIA Nsight Systems
- Google Colab (development)
- RunPod (RTX A4000 benchmarking)

## License

Educational and portfolio project. Dataset: U.S. Government Works (public domain).

## Author

Professional CUDA optimization project demonstrating industrial sensor processing on edge inference hardware (2024).

---

**Resume Summary:**
> "Built CUDA-accelerated vibration analysis system for industrial bearing fault detection using NASA IMS dataset (44M samples). Achieved 3.27× speedup through batching optimization, 2.75× via concurrent multi-sensor streams, and 2.81× using pipelined execution on RTX A4000. Profiled with Nsight Systems to identify PCIe bottleneck (97.9% of naive execution time) and validated optimizations through detailed timing analysis."
