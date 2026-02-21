# RTX A4000 Performance Profiling Analysis

## Hardware
- **GPU:** NVIDIA RTX A4000 (Ampere Architecture)
- **VRAM:** 16 GB GDDR6
- **CUDA Cores:** 6,144
- **Memory Bandwidth:** 448 GB/s

## Dataset
- **Source:** NASA IMS Bearing Dataset (1st Test)
- **Files:** 2,156 vibration snapshots
- **Samples per file:** 20,480 @ 20 kHz
- **Channels:** 8 (4 bearings × 2 accelerometers)
- **Total data:** 44M samples, 2.4 GB

---

## Performance Results (100 files benchmark)

| Implementation | Time (ms) | Throughput (M/s) | vs CPU | vs Naive |
|----------------|-----------|------------------|--------|----------|
| CPU Baseline | 0.341 | 60.1 | 1.00× | - |
| Naive CUDA | 0.246 | 83.3 | 1.38× | 1.00× |
| Scenario 1: Concurrent | 0.124 | 165.6 | 2.75× | 1.98× |
| Scenario 2: Large Batch | 0.104 | 196.9 | **3.27×** | **2.36×** |
| Scenario 3: Pipeline | 0.121 | 169.3 | 2.81× | 2.03× |
| Combined | 0.110 | 185.7 | 3.08× | 2.22× |

**Winner: Scenario 2 (Large Batch) - 3.27× speedup over CPU**

---

## Detailed Timing Breakdown: Naive CUDA

Micro-benchmark analysis (20 files):

