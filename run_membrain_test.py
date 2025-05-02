#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Test script for vLLM V1 with Membrain distributed caching.
This script loads a small model, runs some test prompts with prefix caching,
and compares performance with and without Membrain.
"""

import os
import time
import argparse

# Enable vLLM V1
os.environ["VLLM_USE_V1"] = "1"

print("Testing vLLM V1 with Membrain...")

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="facebook/opt-125m", 
                   help="Model to test with")
parser.add_argument("--enable-membrain", action="store_true",
                   help="Enable Membrain distributed cache")
args = parser.parse_args()

if args.enable_membrain:
    print("Enabling Membrain distributed cache")
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "1"
    os.environ["VLLM_MEMBRAIN_ENDPOINT"] = "http://localhost:9201"
    os.environ["VLLM_MEMBRAIN_NAMESPACE"] = "default"
else:
    print("Using standard prefix caching (Membrain disabled)")
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "0"

# Import vLLM after setting environment variables
from vllm import LLM, SamplingParams

# Create common prefix for testing
prefix = (
    "You are a helpful programming assistant. "
    "Write a function that calculates the Fibonacci sequence "
    "using dynamic programming. Explain the time and space complexity."
)

# Create a few test prompts with common prefix
prompts = [
    prefix + " Implement it in Python.",
    prefix + " Implement it in JavaScript.",
    prefix + " Implement it in Java.",
    prefix + " Implement it in C++.",
]

# Setup sampling parameters
sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic for consistent comparison
    max_tokens=200,
)

# Initialize LLM
print(f"\nInitializing LLM with model: {args.model}")
start_time = time.time()
llm = LLM(
    model=args.model,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
)
init_time = time.time() - start_time
print(f"Initialization time: {init_time:.2f} seconds")

# First run with a single prompt to ensure cache is populated
print("\nWarmup to populate prefix cache...")
start_time = time.time()
outputs = llm.generate(prompts[0], sampling_params)
warmup_time = time.time() - start_time
print(f"Warmup took {warmup_time:.2f} seconds")

# Run the remaining prompts to test cache hits
print("\nRunning prompts with prefix caching...")
all_times = []

for i, prompt in enumerate(prompts[1:], 1):
    start_time = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start_time
    all_times.append(elapsed)
    
    result = outputs[0].outputs[0].text
    print(f"Prompt {i}:")
    print(f"- Time: {elapsed:.2f} seconds")
    print(f"- Generated {len(result)} characters")
    print(f"- Preview: {result[:100]}...")

# Print summary
print("\nSummary:")
print(f"Average generation time: {sum(all_times)/len(all_times):.2f} seconds")
print(f"Membrain enabled: {args.enable_membrain}")

# Get cache stats if possible
try:
    # For V1, try to access the prefix cache stats
    cache_stats = llm.llm_engine.engine_core.engine_core.scheduler.kv_cache_manager.make_prefix_cache_stats()
    print("\nPrefix cache stats:", cache_stats)
except (AttributeError, Exception) as e:
    print(f"\nCouldn't access cache stats: {e}")

print("\nTest completed!")