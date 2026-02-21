#!/bin/bash

echo "=========================================="
echo "NASA Bearing CUDA Performance Comparison"
echo "=========================================="
echo ""

# Create results directory
mkdir -p results/nasa_bearing

# Compile all programs
echo "[1/7] Compiling CPU Baseline..."
g++ -O3 -std=c++17 -o cpu_baseline src_nasa_bearing/baseline/cpu_baseline.cpp

echo "[2/7] Compiling Naive CUDA..."
nvcc -O3 -o cuda_naive src_nasa_bearing/baseline/cuda_naive.cu

echo "[3/7] Compiling Scenario 1 (Concurrent Streams)..."
nvcc -O3 -o scenario1_concurrent src_nasa_bearing/optimized/scenario1_concurrent.cu

echo "[4/7] Compiling Scenario 2 (Large Batches)..."
nvcc -O3 -o scenario2_batch src_nasa_bearing/optimized/scenario2_batch.cu

echo "[5/7] Compiling Scenario 3 (Pipeline)..."
nvcc -O3 -o scenario3_pipeline src_nasa_bearing/optimized/scenario3_pipeline.cu

echo "[6/7] Compiling Combined (All Optimizations)..."
nvcc -O3 -o scenario_combined src_nasa_bearing/optimized/scenario_combined.cu

echo "[7/7] Compilation complete!"
echo ""

# Run benchmarks
echo "Running benchmarks (this will take ~1 minute)..."
echo ""

CPU_RESULT=$(./cpu_baseline)
NAIVE_RESULT=$(./cuda_naive)
S1_RESULT=$(./scenario1_concurrent)
S2_RESULT=$(./scenario2_batch)
S3_RESULT=$(./scenario3_pipeline)
COMBINED_RESULT=$(./scenario_combined)

# Parse results
CPU_TIME=$(echo $CPU_RESULT | cut -d',' -f2)
CPU_THROUGHPUT=$(echo $CPU_RESULT | cut -d',' -f3)

NAIVE_TIME=$(echo $NAIVE_RESULT | cut -d',' -f2)
NAIVE_THROUGHPUT=$(echo $NAIVE_RESULT | cut -d',' -f3)

S1_TIME=$(echo $S1_RESULT | cut -d',' -f2)
S1_THROUGHPUT=$(echo $S1_RESULT | cut -d',' -f3)

S2_TIME=$(echo $S2_RESULT | cut -d',' -f2)
S2_THROUGHPUT=$(echo $S2_RESULT | cut -d',' -f3)

S3_TIME=$(echo $S3_RESULT | cut -d',' -f2)
S3_THROUGHPUT=$(echo $S3_RESULT | cut -d',' -f3)

COMBINED_TIME=$(echo $COMBINED_RESULT | cut -d',' -f2)
COMBINED_THROUGHPUT=$(echo $COMBINED_RESULT | cut -d',' -f3)

# Calculate speedups
NAIVE_VS_CPU=$(echo "scale=2; $CPU_TIME / $NAIVE_TIME" | bc)
S1_VS_CPU=$(echo "scale=2; $CPU_TIME / $S1_TIME" | bc)
S1_VS_NAIVE=$(echo "scale=2; $NAIVE_TIME / $S1_TIME" | bc)
S2_VS_CPU=$(echo "scale=2; $CPU_TIME / $S2_TIME" | bc)
S2_VS_NAIVE=$(echo "scale=2; $NAIVE_TIME / $S2_TIME" | bc)
S3_VS_CPU=$(echo "scale=2; $CPU_TIME / $S3_TIME" | bc)
S3_VS_NAIVE=$(echo "scale=2; $NAIVE_TIME / $S3_TIME" | bc)
COMBINED_VS_CPU=$(echo "scale=2; $CPU_TIME / $COMBINED_TIME" | bc)
COMBINED_VS_NAIVE=$(echo "scale=2; $NAIVE_TIME / $COMBINED_TIME" | bc)

# Display results
echo ""
echo "============================================================================="
echo "                      COMPREHENSIVE PERFORMANCE RESULTS"
echo "============================================================================="
printf "%-30s %12s %15s %10s %10s\n" "Implementation" "Time (ms)" "Throughput" "vs CPU" "vs Naive"
echo "-----------------------------------------------------------------------------"
printf "%-30s %12.3f %15s %10s %10s\n" "CPU Baseline" "$CPU_TIME" "$CPU_THROUGHPUT" "1.00x" "-"
printf "%-30s %12.3f %15s %10s %10s\n" "Naive CUDA" "$NAIVE_TIME" "$NAIVE_THROUGHPUT" "${NAIVE_VS_CPU}x" "1.00x"
printf "%-30s %12.3f %15s %10s %10s\n" "Scenario 1: Concurrent" "$S1_TIME" "$S1_THROUGHPUT" "${S1_VS_CPU}x" "${S1_VS_NAIVE}x"
printf "%-30s %12.3f %15s %10s %10s\n" "Scenario 2: Large Batch" "$S2_TIME" "$S2_THROUGHPUT" "${S2_VS_CPU}x" "${S2_VS_NAIVE}x"
printf "%-30s %12.3f %15s %10s %10s\n" "Scenario 3: Pipeline" "$S3_TIME" "$S3_THROUGHPUT" "${S3_VS_CPU}x" "${S3_VS_NAIVE}x"
printf "%-30s %12.3f %15s %10s %10s\n" "Combined: All Optimizations" "$COMBINED_TIME" "$COMBINED_THROUGHPUT" "${COMBINED_VS_CPU}x" "${COMBINED_VS_NAIVE}x"
echo "============================================================================="

echo ""
echo "Key Optimizations:"
echo "  Scenario 1: 4 concurrent CUDA streams (multi-bearing)"
echo "  Scenario 2: Batch size 5 with async transfers"
echo "  Scenario 3: 3-stage pipeline (normalize → RMS → peak)"
echo "  Combined:   All of the above (12 streams, batch 5, pipeline)"
echo ""

# Save to file
echo "Saving results to results/nasa_bearing/benchmark_results.txt..."
{
    echo "NASA Bearing CUDA Performance Comparison"
    echo "Date: $(date)"
    echo ""
    printf "%-30s %12s %15s %10s %10s\n" "Implementation" "Time (ms)" "Throughput" "vs CPU" "vs Naive"
    echo "-----------------------------------------------------------------------------"
    printf "%-30s %12.3f %15s %10s %10s\n" "CPU Baseline" "$CPU_TIME" "$CPU_THROUGHPUT" "1.00x" "-"
    printf "%-30s %12.3f %15s %10s %10s\n" "Naive CUDA" "$NAIVE_TIME" "$NAIVE_THROUGHPUT" "${NAIVE_VS_CPU}x" "1.00x"
    printf "%-30s %12.3f %15s %10s %10s\n" "Scenario 1: Concurrent" "$S1_TIME" "$S1_THROUGHPUT" "${S1_VS_CPU}x" "${S1_VS_NAIVE}x"
    printf "%-30s %12.3f %15s %10s %10s\n" "Scenario 2: Large Batch" "$S2_TIME" "$S2_THROUGHPUT" "${S2_VS_CPU}x" "${S2_VS_NAIVE}x"
    printf "%-30s %12.3f %15s %10s %10s\n" "Scenario 3: Pipeline" "$S3_TIME" "$S3_THROUGHPUT" "${S3_VS_CPU}x" "${S3_VS_NAIVE}x"
    printf "%-30s %12.3f %15s %10s %10s\n" "Combined: All Optimizations" "$COMBINED_TIME" "$COMBINED_THROUGHPUT" "${COMBINED_VS_CPU}x" "${COMBINED_VS_NAIVE}x"
} > results/nasa_bearing/benchmark_results.txt

echo "✓ Results saved!"
echo ""
echo "Done! 🎉"
