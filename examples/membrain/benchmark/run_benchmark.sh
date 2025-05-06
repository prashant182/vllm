#!/bin/bash
# Script to run benchmark tests against vLLM deployments

# Default values
NAMESPACE="vllm-benchmark"
NUM_PROMPTS=100
CONCURRENCY=10
MAX_TOKENS=100
PREFIX_REUSE_RATE=0.7
OUTPUT_DIR="benchmark_results"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
    --namespace)
      NAMESPACE="$2"
      shift 2
      ;;
    --num-prompts)
      NUM_PROMPTS="$2"
      shift 2
      ;;
    --concurrency)
      CONCURRENCY="$2"
      shift 2
      ;;
    --max-tokens)
      MAX_TOKENS="$2"
      shift 2
      ;;
    --prefix-reuse-rate)
      PREFIX_REUSE_RATE="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# Create output directory if it doesn't exist
mkdir -p "$OUTPUT_DIR"

# Function to run benchmark
run_benchmark() {
  local deployment=$1
  local url="http://${deployment}-service.${NAMESPACE}:8000/v1/chat/completions"
  local output_file="${OUTPUT_DIR}/${deployment}_results.json"
  
  echo "Running benchmark against $deployment..."
  echo "URL: $url"
  echo "Output file: $output_file"
  
  python3 load_test.py \
    --url "$url" \
    --num-prompts "$NUM_PROMPTS" \
    --concurrency "$CONCURRENCY" \
    --max-tokens "$MAX_TOKENS" \
    --prefix-reuse-rate "$PREFIX_REUSE_RATE" \
    --output "$output_file"
}

# Check if deployments are ready
echo "Checking if deployments are ready..."

echo "Benchmarking vllm-membrain deployment..."
run_benchmark "vllm-membrain"

echo "Benchmarking vllm-baseline deployment..."
run_benchmark "vllm-baseline"

echo "Benchmark complete. Results are saved in the $OUTPUT_DIR directory."

# Compare results
echo "Comparing results..."
python3 - <<EOF
import json
import sys

try:
    # Load results
    with open("${OUTPUT_DIR}/vllm-membrain_results.json", "r") as f:
        membrain_results = json.load(f)
        
    with open("${OUTPUT_DIR}/vllm-baseline_results.json", "r") as f:
        baseline_results = json.load(f)
    
    # Extract key metrics
    membrain_latency = membrain_results['latency']['mean']
    baseline_latency = baseline_results['latency']['mean']
    
    membrain_p90 = membrain_results['latency']['p90']
    baseline_p90 = baseline_results['latency']['p90']
    
    membrain_throughput = membrain_results['tokens']['throughput_tokens_per_second']
    baseline_throughput = baseline_results['tokens']['throughput_tokens_per_second']
    
    # Calculate improvements
    latency_improvement = ((baseline_latency - membrain_latency) / baseline_latency) * 100
    p90_improvement = ((baseline_p90 - membrain_p90) / baseline_p90) * 100
    throughput_improvement = ((membrain_throughput - baseline_throughput) / baseline_throughput) * 100
    
    print("\n===== RESULTS COMPARISON =====")
    print(f"Mean Latency (seconds):")
    print(f"  Membrain: {membrain_latency:.3f}s")
    print(f"  Baseline: {baseline_latency:.3f}s")
    print(f"  Improvement: {latency_improvement:.2f}%")
    
    print(f"\np90 Latency (seconds):")
    print(f"  Membrain: {membrain_p90:.3f}s")
    print(f"  Baseline: {baseline_p90:.3f}s")
    print(f"  Improvement: {p90_improvement:.2f}%")
    
    print(f"\nThroughput (tokens/second):")
    print(f"  Membrain: {membrain_throughput:.2f}")
    print(f"  Baseline: {baseline_throughput:.2f}")
    print(f"  Improvement: {throughput_improvement:.2f}%")
    
    print("\nConclusion:")
    if latency_improvement > 0 and throughput_improvement > 0:
        print("Membrain shows significant performance improvements for distributed prefix caching!")
    elif latency_improvement > 0:
        print("Membrain shows faster response times but similar throughput.")
    elif throughput_improvement > 0:
        print("Membrain shows higher throughput but similar latency.")
    else:
        print("No performance improvement observed. Check configuration or test parameters.")
        
except Exception as e:
    print(f"Error comparing results: {e}")
    sys.exit(1)
EOF

echo "Done!"