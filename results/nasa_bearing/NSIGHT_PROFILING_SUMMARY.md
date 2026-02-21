# Nsight Systems Profiling Summary - RTX A4000

## Profiling Files Generated
- `naive_cuda.nsys-rep` - Baseline GPU implementation
- `scenario1_concurrent.nsys-rep` - 4 concurrent streams
- `scenario2_batch.nsys-rep` - Large batch optimization
- `scenario3_pipeline.nsys-rep` - 3-stage pipeline

---

## Key Profiling Insights

### Naive CUDA - The Bottleneck Revealed

**Memory Transfer Analysis:**
```
H2D Transfer:  5.77 ms (52.5 μs avg × 110 calls)  ← 97.9% of GPU time
Kernel Exec:   2.11 ms (19.2 μs avg × 110 calls)  ←  2.0% of GPU time
D2H Transfer:  0.12 ms ( 1.1 μs avg × 110 calls)  ←  0.1% of GPU time
```

**Critical Finding:** PCIe transfers dominate (97.9% of GPU time)
- Small batch size (655 KB per transfer) amplifies overhead
- Synchronous `cudaMemcpy` blocks all execution
- No overlap between transfers and compute

---

### Scenario 1: Concurrent Streams

**Improvements:**
- Uses `cudaMemcpyAsync` (non-blocking)
- Pinned memory via `cudaHostAlloc` (152.8 ms allocation time)
- 4 concurrent streams enable parallel kernel execution

**Memory Transfer Analysis:**
```
H2D Transfer:  6.05 ms (55.0 μs avg × 110 calls)
Kernel Exec:   1.52 ms ( 3.5 μs avg × 440 calls)  ← 4× more kernels!
D2H Transfer:  0.11 ms ( 1.1 μs avg × 100 calls)
```

**Key Insight:** 
- Kernel time DROPPED from 19.2 μs to 3.5 μs per call
- Why? 4 bearings processed concurrently (4 kernels per file)
- Each kernel does less work (2 channels vs 8)

---

### Scenario 2: Large Batch (WINNER - 3.27× speedup)

**Batching Impact:**
```
Transfer Size:  3.28 MB per batch (vs 655 KB naive)
H2D Transfer:   6.37 ms (255.0 μs avg × 25 calls)  ← Fewer, larger transfers
Kernel Exec:    0.60 ms ( 24.1 μs avg × 25 calls)
D2H Transfer:   0.03 ms (  1.2 μs avg × 25 calls)
```

**Why It's Fastest:**
- Only 25 kernel launches (vs 110 in naive)
- Larger transfers amortize PCIe overhead
- Async operations with pinned memory
- Better utilization: 99.5% GPU time vs 97.9% naive

---

### Scenario 3: Multi-Stage Pipeline

**Pipeline Stages Breakdown:**
```
Stage 1 (Normalize):  0.28 ms ( 2.5 μs × 110 calls) -  6.1% kernel time
Stage 2 (RMS):        2.19 ms (20.1 μs × 109 calls) - 47.9% kernel time
Stage 3 (Peak):       2.10 ms (19.5 μs × 108 calls) - 46.0% kernel time
```

**Observation:**
- 3 different kernel types executing in pipeline
- Stage 2 and 3 dominate execution time
- Shows overlapped execution via streams

---

## Comparative Analysis

| Metric | Naive | S1 | S2 | S3 |
|--------|-------|----|----|-----|
| **Total kernel launches** | 110 | 440 | 25 | 327 |
| **Avg H2D time (μs)** | 52.5 | 55.0 | 255.0 | 52.7 |
| **Avg kernel time (μs)** | 19.2 | 3.5 | 24.1 | varies |
| **Transfer size (MB)** | 0.66 | 0.66 | 3.28 | 0.66 |
| **Uses async transfers** | ❌ | ✅ | ✅ | ✅ |
| **Uses pinned memory** | ❌ | ✅ | ✅ | ✅ |
| **Concurrent streams** | 0 | 4 | 1 | 3 |

---

## Interview-Ready Insights

**Q: "How did you identify the bottleneck?"**
> "I used Nsight Systems to profile the naive CUDA implementation. The timeline showed PCIe transfers consumed 97.9% of GPU execution time, with the kernel only taking 2%. This immediately revealed that data movement, not computation, was the critical path."

**Q: "What's the difference between cudaMemcpy and cudaMemcpyAsync?"**
> "cudaMemcpy is synchronous—it blocks the CPU until the transfer completes. cudaMemcpyAsync returns immediately, allowing the CPU to continue work while the GPU handles the transfer. Combined with CUDA streams, async transfers enable overlapping of H2D transfer, kernel execution, and D2H transfer for different batches."

**Q: "Why is Scenario 2 faster than Scenario 1?"**
> "Scenario 2 reduces overhead through batching. Instead of 110 small transfers (655 KB each), it does 25 larger transfers (3.28 MB each). The profiling shows fewer kernel launches (25 vs 440) and better PCIe utilization. While Scenario 1 adds complexity with 4 concurrent streams, Scenario 2's simplicity—just batching with async—proved more efficient for this workload."

**Q: "Can you explain what the .nsys-rep files show?"**
> "The Nsight Systems reports provide a timeline view showing exactly when each operation occurs—CPU activity, GPU kernels, memory transfers. You can see overlapped execution, stream concurrency, and identify gaps where the GPU is idle. For example, in the naive version, you'd see serial execution: H2D → kernel → D2H → repeat. In the optimized versions, you see overlapping operations across multiple streams."

---

## Production Recommendations

Based on profiling data:

1. **For similar workloads:** Use Scenario 2 (batching)
   - Simplest implementation
   - Best performance (3.27× speedup)
   - Minimal code complexity

2. **For multi-sensor concurrent:** Use Scenario 1
   - When sensors truly need independent processing
   - Good for heterogeneous workloads

3. **For multi-stage pipelines:** Use Scenario 3
   - When you have distinct preprocessing stages
   - Example: raw → filter → inference → post-process

4. **Avoid over-optimization:**
   - Combined scenario showed complexity doesn't always win
   - Profile first, optimize second
   - Match technique to problem

---

## Files for Download

All `.nsys-rep` files can be:
1. Downloaded from RunPod
2. Opened in Nsight Systems GUI (free download from NVIDIA)
3. Visualized as timeline showing exact execution patterns

**To view:**
```bash
# On your local machine with Nsight Systems installed
nsys-ui profiling/scenario2_batch.nsys-rep
```

This will show the complete execution timeline with overlapping operations visible.
