#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import os
import time
import argparse
from vllm import LLM, SamplingParams

# Parse arguments
parser = argparse.ArgumentParser(description="Test Membrain distributed caching in vLLM")
parser.add_argument("--model", type=str, default="facebook/opt-125m",
                    help="Model to use for testing")
parser.add_argument("--enable-membrain", action="store_true",
                    help="Enable Membrain distributed caching")
parser.add_argument("--membrain-endpoint", type=str, default="http://localhost:9201",
                    help="Membrain server endpoint")
parser.add_argument("--membrain-namespace", type=str, default="default",
                    help="Membrain namespace")
args = parser.parse_args()

# Common prefix for testing cache hits
prefix = (
    "You are an expert software engineer. "
    "Write a well-documented function that calculates the Fibonacci sequence "
    "efficiently using dynamic programming, with clear comments explaining "
    "the algorithm and time complexity."
)

# Test prompts with common prefix
prompts = [
    prefix + " Implement it in Python.",
    prefix + " Implement it in JavaScript.",
    prefix + " Implement it in Java.",
]

# Set environment variables for vLLM V1 and Membrain
os.environ["VLLM_USE_V1"] = "1"

if args.enable_membrain:
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "1"
    os.environ["VLLM_MEMBRAIN_ENDPOINT"] = args.membrain_endpoint
    os.environ["VLLM_MEMBRAIN_NAMESPACE"] = args.membrain_namespace
    print(f"Testing with Membrain enabled at {args.membrain_endpoint}")
else:
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "0"
    print("Testing with standard vLLM (Membrain disabled)")

# Create sampling params
sampling_params = SamplingParams(
    temperature=0.0,
    max_tokens=200,
)

# Initialize LLM
print(f"Initializing LLM with model: {args.model}")
llm = LLM(
    model=args.model,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,
)

# First run with prompt[0] to cache the prefix
print("\nWarmup to cache prefix...")
start_time = time.time()
outputs = llm.generate(prompts[0], sampling_params)
warmup_time = time.time() - start_time
print(f"Warmup completed in {warmup_time:.2f} seconds")
print(f"Generated {len(outputs[0].outputs[0].text)} tokens")

# Run the next prompts which should benefit from caching
print("\nRunning prompts with cached prefix...")
for i, prompt in enumerate(prompts[1:], 1):
    start_time = time.time()
    outputs = llm.generate(prompt, sampling_params)
    elapsed = time.time() - start_time
    
    generated_text = outputs[0].outputs[0].text
    print(f"\nPrompt {i}:")
    print(f"- Time: {elapsed:.2f} seconds")
    print(f"- Generated tokens: {len(generated_text)}")
    print(f"- Output preview: {generated_text[:100]}...")

# Print cache stats if available
try:
    if hasattr(llm, "get_cache_stats"):
        print("\nCache Stats:")
        cache_stats = llm.get_cache_stats()
        print(cache_stats)
    else:
        # Alternative way to get cache stats (implementation-dependent)
        print("\nNote: Cache stats not directly accessible through LLM interface")
except Exception as e:
    print(f"\nError getting cache stats: {e}")

print("\nTest completed!")