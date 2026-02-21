# PAMAP2 Dataset Benchmark Results

## Dataset Information

**Source:** PAMAP2 Physical Activity Monitoring  
**Subjects:** 9 (subject101-109)  
**Total Samples:** 2,859,000 (after NaN removal)  
**Features per Sample:** 6 (3-axis accelerometer + 3-axis gyroscope)  
**Batch Size:** 1,000 samples  
**Test Batches:** 500

## Hardware Configurations

### Development Machine (M1 MacBook Pro)
- **CPU:** Apple M1 (ARM architecture)
- **RAM:** Unified memory architecture
- **Compiler:** Apple Clang 17.0.0
- **Optimization:** -O3 -std=c++17

### Benchmark Machine (Google Colab)
- **CPU:** Intel Xeon (2-4 cores, server-grade)
- **GPU:** Tesla T4 (Turing architecture, 16GB VRAM, 2560 CUDA cores)
- **CUDA Version:** 12.8
- **Compiler:** nvcc with -O3

## Benchmark Results

### Test 1: CPU Baseline (Optimized)

**Implementation:**
- Proper z-score normalization (computed mean and std)
- Sliding window moving average (O(n) vs O(n*w))
- Cache-friendly memory access patterns

**Results on M1 Mac:**
```
Total Time:        36.485 ms
Avg Batch Time:    0.073 ms
Throughput:        13,704,261 samples/sec
Per-Sample Time:   0.073 μs
```

**Results on Colab Xeon:**
```
Total Time:        26.952 ms
Avg Batch Time:    0.054 ms
Throughput:        18,551,498 samples/sec
Per-Sample Time:   0.054 μs
```

**Key Insight:** Server-grade Xeon outperformed M1 by ~35% due to higher clock speeds and optimization for throughput workloads.

---

### Test 2: Naive CUDA

**Implementation:**
- Basic GPU parallelization (one thread per sample)
- Synchronous memory transfers (cudaMemcpy)
- No stream optimization
- Simple kernel with minimal optimizations

**Kernel Configuration:**
- Threads per block: 256
- Blocks per grid: 4
- Total threads: 1,024

**Results on Tesla T4:**
```
Total Time:        20.247 ms
Avg Batch Time:    0.040 ms
Throughput:        24,695,670 samples/sec
Per-Sample Time:   0.040 μs
```

**Performance vs CPU:**
- Speedup: 1.35x (0.054 / 0.040)
- Throughput improvement: 33%

**Analysis:** Modest speedup due to:
- Small batch size (24KB fits in CPU cache)
- Memory transfer overhead
- Synchronous execution pattern

---

### Test 3: CUDA with Streams (Attempted Optimization)

**Implementation:**
- 4 concurrent CUDA streams
- Asynchronous memory transfers (cudaMemcpyAsync)
- Pinned host memory (cudaMallocHost)
- Attempted overlapping of H2D, kernel, and D2H

**Results on Tesla T4:**
```
Total Time:        19.026 ms
Avg Batch Time:    0.038 ms
Throughput:        26,279,850 samples/sec
Per-Sample Time:   0.038 μs
```

**Performance vs Naive CUDA:**
- Speedup: 1.05x (0.040 / 0.038)
- Throughput improvement: 6%

**Performance vs CPU:**
- Speedup: 1.42x (0.054 / 0.038)
- Throughput improvement: 42%

**Analysis - Why Streams Didn't Help:**

1. **Batch Size Too Small**
   - 24KB per batch fits entirely in L1 cache
   - Memory transfer time: ~0.01ms (negligible)
   - Nothing meaningful to overlap

2. **File I/O Bottleneck**
   - Sequential file reads block the pipeline
   - GPU idles waiting for data from disk
   - Streams can't overlap when CPU is bottlenecked

3. **Transfer-to-Compute Ratio**
   - Kernel execution: ~0.02ms
   - Memory transfer: ~0.01ms
   - Ratio too low for effective overlap

---

## Comparative Summary

| Implementation | Platform | Batch Time (ms) | Throughput (samples/sec) | Speedup vs CPU |
|----------------|----------|----------------|--------------------------|----------------|
| CPU Baseline | M1 Mac | 0.073 | 13,704,261 | 1.0x (baseline) |
| CPU Baseline | Xeon | 0.054 | 18,551,498 | 1.35x |
| Naive CUDA | Tesla T4 | 0.040 | 24,695,670 | 1.82x |
| CUDA Streams | Tesla T4 | 0.038 | 26,279,850 | 1.92x |

## Lessons Learned

### What Worked
- Optimized CPU implementation (sliding window, proper normalization)
- GPU parallelization for compute-bound operations
- Tesla T4 significantly faster than consumer CPUs

### What Didn't Work
- CUDA streams on small batches (insufficient overlap opportunity)
- Async transfers when file I/O dominates
- Stream optimization without addressing root bottleneck

### Key Insights for Edge Inference

1. **Batch Size Matters**
   - Small batches (< 100KB) don't benefit from stream optimization
   - Need 1MB+ batches or multiple concurrent data sources

2. **Real Edge Scenarios Differ**
   - Single human activity stream != industrial IoT
   - Real systems have multiple sensors or larger processing windows
   - File-based benchmarks don't represent streaming workloads

3. **Next Steps**
   - Multi-sensor concurrent processing (4+ independent streams)
   - Larger batch sizes (10K-100K samples)
   - Pipeline parallelism (multi-stage processing)

## Conclusion

PAMAP2 dataset served as excellent learning foundation for CUDA concepts but revealed limitations for demonstrating advanced GPU optimizations. Batch sizes too small and single-stream nature don't represent typical industrial edge inference workloads.

Project evolution led to Edge-IIoTset dataset selection for realistic multi-sensor edge inference scenarios.

## References

- PAMAP2 Dataset: Reiss, A. and Stricker, D. (2012). ISWC 2012.
- Hardware: Google Colab (Tesla T4), Apple M1 MacBook Pro
- Profiling: CUDA Events, manual timing analysis
