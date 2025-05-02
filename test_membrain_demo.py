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

# Configure logging to highlight Membrain operations
logging.basicConfig(
    level=logging.INFO,  # Use INFO level for less noise
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

# Set specific loggers to DEBUG for more detailed output
membrain_loggers = [
    'vllm.v1.core.membrain',
    'vllm.v1.core.membrain_kvmanager',
]
for logger_name in membrain_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.DEBUG)
    
# Set other loggers to WARNING to reduce noise
other_loggers = [
    'vllm.distributed',
    'vllm.worker',
    'vllm.logger',
    'vllm.utils',
]
for logger_name in other_loggers:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.WARNING)

# Enable vLLM V1 
os.environ["VLLM_USE_V1"] = "1" 
# Make sure this is set before importing vllm
print("[SETUP] Setting VLLM_USE_V1=1")

# Parse arguments
parser = argparse.ArgumentParser(description="Demonstrate Membrain caching")
parser.add_argument("--model", type=str, default="facebook/opt-125m",
                   help="Model to use for testing")
parser.add_argument("--disable-membrain", action="store_true",
                   help="Disable Membrain (use standard prefix caching)")
parser.add_argument("--namespace", type=str, default="default",
                   help="Membrain namespace to use")
args = parser.parse_args()

if not args.disable_membrain:
    print("\n[SETUP] Enabling Membrain distributed caching")
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "1"
    os.environ["VLLM_MEMBRAIN_ENDPOINT"] = "http://localhost:9201"
    os.environ["VLLM_MEMBRAIN_NAMESPACE"] = args.namespace
    print(f"[SETUP] Using endpoint: http://localhost:9201")
    print(f"[SETUP] Using namespace: {args.namespace}")
    
    # Verify environment variables are correctly set
    print(f"[SETUP] VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")
    print(f"[SETUP] VLLM_MEMBRAIN_ENABLED={os.environ.get('VLLM_MEMBRAIN_ENABLED')}")
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

# Display cache metrics if available
if not args.disable_membrain:
    try:
        # Access internal llm engine and then kv cache manager to get metrics
        if hasattr(llm, '_llm_engine') and hasattr(llm._llm_engine, 'scheduler'):
            kv_cache_manager = llm._llm_engine.scheduler.kv_cache_manager
            if hasattr(kv_cache_manager, 'get_metrics'):
                metrics = kv_cache_manager.get_metrics()
                print("\n[METRICS] Membrain Cache Statistics:")
                if 'membrain' in metrics:
                    memb_metrics = metrics['membrain']
                    
                    # Client-side metrics
                    print(f"  Client-side Metrics:")
                    print(f"  - Store attempts: {memb_metrics.get('store_attempts', 'N/A')}")
                    print(f"  - Store successes: {memb_metrics.get('store_successes', 'N/A')}")
                    store_rate = 0
                    if memb_metrics.get('store_attempts', 0) > 0:
                        store_rate = memb_metrics.get('store_successes', 0) / memb_metrics.get('store_attempts', 0) * 100
                    print(f"  - Store success rate: {store_rate:.1f}%")
                    print(f"  - Load attempts: {memb_metrics.get('load_attempts', 'N/A')}")
                    print(f"  - Load successes: {memb_metrics.get('load_successes', 'N/A')}")
                    load_rate = 0
                    if memb_metrics.get('load_attempts', 0) > 0:
                        load_rate = memb_metrics.get('load_successes', 0) / memb_metrics.get('load_attempts', 0) * 100
                    print(f"  - Load success rate: {load_rate:.1f}%")
                    print(f"  - Remote blocks tracked: {memb_metrics.get('tracked_remote_blocks', 'N/A')}")
                    
                    # Remote server metrics  
                    print(f"\n  Server-side Metrics:")
                    print(f"  - Server hits: {memb_metrics.get('hits', 'N/A')}")
                    print(f"  - Server misses: {memb_metrics.get('misses', 'N/A')}")
                    hit_rate = memb_metrics.get('hit_rate', 0) * 100
                    print(f"  - Server hit rate: {hit_rate:.1f}%")
                    
                    # Test result determination
                    if memb_metrics.get('store_successes', 0) > 0:
                        print(f"\n[SUCCESS] Successfully stored {memb_metrics.get('store_successes')} blocks in Membrain!")
                    else:
                        print(f"\n[FAILURE] Failed to store any blocks in Membrain")
                        print(f"  Check that Membrain server is running at {os.environ.get('VLLM_MEMBRAIN_ENDPOINT')}")
                else:
                    print("  No Membrain metrics available. Metrics collection may be disabled.")
    except Exception as e:
        print(f"[ERROR] Could not retrieve metrics: {e}")
        
    print("\n[DEBUG] Let's force manual caching of a block for testing:")
    try:
        # Try to manually cache a block to prove the Membrain service works
        if hasattr(llm, '_llm_engine') and hasattr(llm._llm_engine, 'scheduler'):
            kv_cache_manager = llm._llm_engine.scheduler.kv_cache_manager
            if hasattr(kv_cache_manager, 'membrain'):
                import torch
                test_key = f"test-manual-{int(time.time())}"
                test_tensor = torch.ones((16, 16), dtype=torch.float16)
                
                print(f"  Creating test tensor with key: {test_key}")
                
                success = kv_cache_manager._event_loop.run_until_complete(
                    kv_cache_manager.membrain.store_block(
                        test_key, 
                        test_tensor,
                        metadata={"source": "manual_test"}
                    )
                )
                
                print(f"  Manual block storage success: {success}")
                
                # Try to retrieve it
                retrieved = kv_cache_manager._event_loop.run_until_complete(
                    kv_cache_manager.membrain.load_block(test_key)
                )
                
                if retrieved is not None:
                    print(f"  Successfully retrieved manual test block with shape: {retrieved.shape}")
                else:
                    print(f"  Failed to retrieve manual test block!")
    except Exception as e:
        print(f"  [ERROR] Manual caching test failed: {e}")
        import traceback
        print(f"  {traceback.format_exc()}")

# If you observe "MEMBRAIN HIT" log messages during Phase 2, 
# it confirms that the distributed cache is working properly
print("\n[HELP] Look for 'MEMBRAIN HIT' or 'MEMBRAIN STORE' log messages above to confirm cache operation")
print("[HELP] If you don't see any, check that the server is running")