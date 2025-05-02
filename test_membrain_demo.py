#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Focused test script to demonstrate Membrain distributed caching.
This script:
1. Sets up logging to clearly show cache operations
2. Runs a prompt to populate the cache
3. Runs a similar prompt that should hit the cache
4. Shows the Membrain cache hits/misses in the logs
"""

import os
import time
import torch
import logging
import argparse

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Enable vLLM V1
os.environ["VLLM_USE_V1"] = "1"

# Parse arguments
parser = argparse.ArgumentParser(description="Demonstrate Membrain caching")
parser.add_argument("--model", type=str, default="facebook/opt-125m",
                   help="Model to use for testing")
parser.add_argument("--disable-membrain", action="store_true",
                   help="Disable Membrain (use standard prefix caching)")
parser.add_argument("--namespace", type=str, default="test-demo",
                   help="Membrain namespace to use")
args = parser.parse_args()

if not args.disable_membrain:
    print("\n[SETUP] Enabling Membrain distributed caching")
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "1"
    os.environ["VLLM_MEMBRAIN_ENDPOINT"] = "http://localhost:9201"
    os.environ["VLLM_MEMBRAIN_NAMESPACE"] = args.namespace
    print(f"[SETUP] Using namespace: {args.namespace}")
else:
    print("\n[SETUP] Using standard prefix caching (Membrain disabled)")
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "0"

# Import vLLM after setting environment variables
from vllm import LLM, SamplingParams

# Create a very long common prefix to clearly demonstrate prefix caching
# Each token must be exactly the same to get a cache hit
prefix = (
    "You are a helpful programming assistant specializing in Python and JavaScript. "
    "Your task is to help users with coding problems, explain concepts, and provide "
    "well-documented code examples. When writing code, you should always include "
    "clear comments explaining what each section does, use consistent indentation, "
    "and follow best practices for the language. You should always consider edge cases "
    "and potential optimizations in your solutions. When explaining concepts, you "
    "should be clear, concise, and use examples to illustrate your points. "
)

# Using similar prompts that share the same prefix
# First prompt populates the cache, second prompt should use cached prefix
# Reverse the order from the previous run to see if distributed cache hits work
prompts = [
    prefix + "Write a function that finds prime numbers.",
    prefix + "Write a function that calculates the Fibonacci sequence.",
]

# Setup sampling parameters with deterministic output for consistent comparison
sampling_params = SamplingParams(
    temperature=0.0,  # Deterministic
    max_tokens=200,
)

print(f"\n[SETUP] Initializing LLM with model: {args.model}")
print("[SETUP] This may take a minute...")
start_time = time.time()
llm = LLM(
    model=args.model,
    gpu_memory_utilization=0.7,
    enable_prefix_caching=True,  # Enable prefix caching (required for Membrain too)
)
init_time = time.time() - start_time
print(f"[SETUP] Initialization completed in {init_time:.2f} seconds")

# PHASE 1: POPULATE CACHE
print("\n" + "=" * 80)
print("PHASE 1: POPULATE CACHE")
print("=" * 80)
print("\n[TEST] Running first prompt to populate cache...")
print(f"[TEST] Prompt length: {len(prompts[0])} characters")
start_time = time.time()
outputs = llm.generate(prompts[0], sampling_params)
phase1_time = time.time() - start_time
result = outputs[0].outputs[0].text

print(f"[RESULT] Generation completed in {phase1_time:.2f} seconds")
print(f"[RESULT] Generated {len(result)} characters")
print("[RESULT] First few lines of output:")
print("---")
print("\n".join(result.strip().split("\n")[:5]))
print("---")

# Wait a moment to make log messages easier to read
time.sleep(1)

# PHASE 2: HIT CACHE
print("\n" + "=" * 80)
print("PHASE 2: TEST CACHE HIT")
print("=" * 80)
print("\n[TEST] Running second prompt which should hit the cache...")
print(f"[TEST] Common prefix length: {len(prefix)} characters")
print(f"[TEST] Total prompt length: {len(prompts[1])} characters")
start_time = time.time()
outputs = llm.generate(prompts[1], sampling_params)
phase2_time = time.time() - start_time
result = outputs[0].outputs[0].text

print(f"[RESULT] Generation completed in {phase2_time:.2f} seconds")
print(f"[RESULT] Generated {len(result)} characters")
print("[RESULT] First few lines of output:")
print("---")
print("\n".join(result.strip().split("\n")[:5]))
print("---")

# PHASE 3: SUMMARY
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"[SUMMARY] First generation time: {phase1_time:.2f} seconds")
print(f"[SUMMARY] Second generation time: {phase2_time:.2f} seconds")
print(f"[SUMMARY] {'Membrain' if not args.disable_membrain else 'Standard prefix caching'} enabled")

# If you observe "MEMBRAIN HIT" log messages during Phase 2, 
# it confirms that the distributed cache is working properly
print("\n[HELP] Look for 'MEMBRAIN HIT' log messages above to confirm cache hits")
print("[HELP] If you don't see any, check that the server is running")