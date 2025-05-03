#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""
Focused test script to demonstrate Membrain tiered caching.
This script:
1. Sets up logging to clearly show cache operations
2. Runs prompts to populate the cache hierarchy (GPU, CPU, and remote)
3. Shows the effectiveness of each tier in the cache hierarchy
4. Reports metrics on cache hits/misses across all tiers
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
    'vllm.v1.core.cpu_cache',
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
parser = argparse.ArgumentParser(description="Demonstrate Membrain tiered caching")
parser.add_argument("--model", type=str, default="facebook/opt-125m",
                   help="Model to use for testing")
parser.add_argument("--disable-membrain", action="store_true",
                   help="Disable Membrain (use standard prefix caching)")
parser.add_argument("--namespace", type=str, default="default",
                   help="Membrain namespace to use")
parser.add_argument("--cpu-cache-size", type=float, default=2,
                   help="CPU cache size in GB")
parser.add_argument("--skip-cpu-test", action="store_true",
                   help="Skip CPU tier forcing test")
parser.add_argument("--force-cpu-cache", action="store_true",
                   help="Force storing blocks in CPU cache")
parser.add_argument("--force-remote-cache", action="store_true",
                   help="Force storing blocks in remote cache")
args = parser.parse_args()

if not args.disable_membrain:
    print("\n[SETUP] Enabling Membrain distributed caching")
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "1"
    os.environ["VLLM_MEMBRAIN_ENDPOINT"] = "http://localhost:9201"
    os.environ["VLLM_MEMBRAIN_NAMESPACE"] = args.namespace
    os.environ["VLLM_MEMBRAIN_CPU_CACHE_SIZE_GB"] = str(args.cpu_cache_size)
    
    # Set force caching environment variables if requested
    if args.force_cpu_cache:
        os.environ["VLLM_FORCE_CPU_CACHE"] = "1" 
        print("[SETUP] Forcing CPU cache storage (VLLM_FORCE_CPU_CACHE=1)")
        
    if args.force_remote_cache:
        os.environ["VLLM_FORCE_REMOTE_CACHE"] = "1"
        print("[SETUP] Forcing remote cache storage (VLLM_FORCE_REMOTE_CACHE=1)")
        
    print(f"[SETUP] Using endpoint: http://localhost:9201")
    print(f"[SETUP] Using namespace: {args.namespace}")
    print(f"[SETUP] CPU cache size: {args.cpu_cache_size}GB")
    
    # Verify environment variables are correctly set
    print(f"[SETUP] VLLM_USE_V1={os.environ.get('VLLM_USE_V1')}")
    print(f"[SETUP] VLLM_MEMBRAIN_ENABLED={os.environ.get('VLLM_MEMBRAIN_ENABLED')}")
    print(f"[SETUP] VLLM_MEMBRAIN_CPU_CACHE_SIZE_GB={os.environ.get('VLLM_MEMBRAIN_CPU_CACHE_SIZE_GB')}")
    print(f"[SETUP] VLLM_FORCE_CPU_CACHE={os.environ.get('VLLM_FORCE_CPU_CACHE', '0')}")
    print(f"[SETUP] VLLM_FORCE_REMOTE_CACHE={os.environ.get('VLLM_FORCE_REMOTE_CACHE', '0')}")
    
    # Patch the membrain_kvmanager module to print more debug info
    try:
        import vllm.v1.core.membrain_kvmanager
        
        # Get original methods
        original_cache_full_blocks = vllm.v1.core.membrain_kvmanager.MembrainKVCacheManager.cache_full_blocks
        
        # Create more descriptive cache_full_blocks
        def debug_cache_full_blocks(self, request, blocks, block_hashes, 
                             num_cached_blocks, num_full_blocks, block_size, hash_fn):
            print(f"\n🛠️ DEBUG: cache_full_blocks called for request {request.request_id}")
            print(f"🛠️ DEBUG: Total blocks: {len(blocks)}, cached: {num_cached_blocks}, full: {num_full_blocks}")
            
            if num_cached_blocks < num_full_blocks:
                print(f"🛠️ DEBUG: Will cache {num_full_blocks - num_cached_blocks} new blocks")
            else:
                print(f"🛠️ DEBUG: No new blocks to cache")
            
            # Call original with detailed logging
            result = original_cache_full_blocks(self, request, blocks, block_hashes, 
                                      num_cached_blocks, num_full_blocks, block_size, hash_fn)
            
            # Report after caching
            if hasattr(self, 'remote_blocks'):
                print(f"🛠️ DEBUG: After caching, remote_blocks has {len(self.remote_blocks)} blocks")
            if hasattr(self, 'cpu_blocks'):
                print(f"🛠️ DEBUG: After caching, cpu_blocks has {len(self.cpu_blocks)} blocks")
            
            return result
            
        # Replace the methods
        vllm.v1.core.membrain_kvmanager.MembrainKVCacheManager.cache_full_blocks = debug_cache_full_blocks
            
        print("[SETUP] Enhanced debug logging installed for membrain_kvmanager")
    except Exception as e:
        print(f"[SETUP] Could not patch membrain_kvmanager for debug: {e}")
else:
    print("\n[SETUP] Using standard prefix caching (Membrain disabled)")
    os.environ["VLLM_MEMBRAIN_ENABLED"] = "0"

# Import vLLM after setting environment variables
from vllm import LLM, SamplingParams

# Create a long common prefix to clearly demonstrate prefix caching
prefix = (
    "You are a helpful programming assistant specializing in Python and JavaScript. "
    "Your task is to help users with coding problems, explain concepts, and provide "
    "well-documented code examples. When writing code, you should always include "
    "clear comments explaining what each section does, use consistent indentation, "
    "and follow best practices for the language. You should always consider edge cases "
    "and potential optimizations in your solutions. When explaining concepts, you "
    "should be clear, concise, and use examples to illustrate your points. "
)

# Using a set of prompts with shared prefix to test the cache hierarchy
prompts = [
    # First batch - populate the cache with different prompts sharing prefix
    prefix + "Write a function that finds prime numbers.",
    prefix + "Write a function that calculates the Fibonacci sequence.",
    prefix + "Explain the concept of recursion with examples.",
    
    # Second batch - Use the same prompts to test cache hits from different tiers
    prefix + "Write a function that finds prime numbers.",
    prefix + "Write a function that calculates the Fibonacci sequence.",
    prefix + "Explain the concept of recursion with examples.",
]

# Additional prompts for CPU cache testing
cpu_cache_prompts = [
    prefix + "Explain the concept of dynamic programming.",
    prefix + "What are decorators in Python?",
    prefix + "Explain how JavaScript closures work.",
]

# Setup sampling parameters with deterministic output
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
print("PHASE 1: POPULATE CACHE (First Generation)")
print("=" * 80)

# First prompt - initially populating the cache
print("\n[TEST] Running first prompt...")
print(f"[TEST] Prompt length: {len(prompts[0])} characters")
start_time = time.time()
outputs = llm.generate([prompts[0]], sampling_params)
p1_time = time.time() - start_time
result = outputs[0].outputs[0].text
print(f"[RESULT] First generation completed in {p1_time:.2f} seconds")

# Wait a moment to make log messages easier to read
time.sleep(1)

# Second prompt - uses prefix from cache but generates new content
print("\n[TEST] Running second prompt (should hit prefix cache)...")
print(f"[TEST] Common prefix length: {len(prefix)} characters")
start_time = time.time()
outputs = llm.generate([prompts[1]], sampling_params)
p2_time = time.time() - start_time
result = outputs[0].outputs[0].text
print(f"[RESULT] Second generation completed in {p2_time:.2f} seconds")

# Wait a moment to make log messages easier to read
time.sleep(1)

# Third prompt - uses prefix from cache but generates new content
print("\n[TEST] Running third prompt (should hit prefix cache)...")
print(f"[TEST] Common prefix length: {len(prefix)} characters")
start_time = time.time()
outputs = llm.generate([prompts[2]], sampling_params)
p3_time = time.time() - start_time
result = outputs[0].outputs[0].text
print(f"[RESULT] Third generation completed in {p3_time:.2f} seconds")

# PHASE 2: TESTING CACHE TIERS (GPU -> CPU -> REMOTE)
print("\n" + "=" * 80)
print("PHASE 2: TEST CACHE TIER HIERARCHY")
print("=" * 80)

# Wait longer to make sure cache contents might be evicted from GPU to CPU/remote tiers
print("\n[TEST] Waiting 3 seconds to allow cache migration between tiers...")
time.sleep(3)

# First repeat test (should hit GPU or CPU cache)
print("\n[TEST] Repeating first prompt (should hit cache tier)...")
start_time = time.time()
outputs = llm.generate([prompts[3]], sampling_params)
p4_time = time.time() - start_time
result = outputs[0].outputs[0].text
print(f"[RESULT] Repeated first prompt completed in {p4_time:.2f} seconds")
p1_speedup = (p1_time - p4_time) / p1_time * 100 if p1_time > 0 else 0
print(f"[RESULT] {p1_speedup:.1f}% faster than first generation")

# Wait a moment to make log messages easier to read
time.sleep(1)

# Second repeat test (should hit GPU or CPU cache)
print("\n[TEST] Repeating second prompt (should hit cache tier)...")
start_time = time.time()
outputs = llm.generate([prompts[4]], sampling_params)
p5_time = time.time() - start_time
result = outputs[0].outputs[0].text
print(f"[RESULT] Repeated second prompt completed in {p5_time:.2f} seconds")
p2_speedup = (p2_time - p5_time) / p2_time * 100 if p2_time > 0 else 0
print(f"[RESULT] {p2_speedup:.1f}% faster than second generation")

# Wait a moment to make log messages easier to read
time.sleep(1)

# Third repeat test (should hit GPU or CPU cache)
print("\n[TEST] Repeating third prompt (should hit cache tier)...")
start_time = time.time()
outputs = llm.generate([prompts[5]], sampling_params)
p6_time = time.time() - start_time
result = outputs[0].outputs[0].text
print(f"[RESULT] Repeated third prompt completed in {p6_time:.2f} seconds")
p3_speedup = (p3_time - p6_time) / p3_time * 100 if p3_time > 0 else 0
print(f"[RESULT] {p3_speedup:.1f}% faster than third generation")

# PHASE 3: EXPLICIT CPU CACHE TEST
if not args.disable_membrain and not args.skip_cpu_test:
    print("\n" + "=" * 80)
    print("PHASE 3: EXPLICIT CPU CACHE TEST")
    print("=" * 80)
    
    # Direct access to KV cache manager to set debug hooks
    kv_cache_manager = None
    original_method = None
    try:
        if hasattr(llm, '_llm_engine'):
            engine = llm._llm_engine
            if hasattr(engine, 'scheduler') and engine.scheduler is not None:
                scheduler = engine.scheduler
                if hasattr(scheduler, 'kv_cache_manager') and scheduler.kv_cache_manager is not None:
                    kv_cache_manager = scheduler.kv_cache_manager
                    print(f"\n[DEBUG] Found KV cache manager: {type(kv_cache_manager).__name__}")
                    
                    # Install hook to log block storage
                    if hasattr(kv_cache_manager, 'cache_full_blocks'):
                        original_method = kv_cache_manager.cache_full_blocks
                        
                        # Create wrapper function that logs calls then calls original
                        def instrumented_cache_full_blocks(self, request, blocks, block_hashes, 
                                                          num_cached_blocks, num_full_blocks,
                                                          block_size, hash_fn):
                            print(f"\n[BLOCK STORAGE] cache_full_blocks called with {len(blocks)} blocks")
                            print(f"[BLOCK STORAGE] num_cached_blocks={num_cached_blocks}, num_full_blocks={num_full_blocks}")
                            
                            # Call original method
                            result = original_method(self, request, blocks, block_hashes, 
                                                  num_cached_blocks, num_full_blocks,
                                                  block_size, hash_fn)
                            
                            print(f"[BLOCK STORAGE] Finished cache_full_blocks call")
                            return result
                        
                        # Replace with instrumented version
                        kv_cache_manager.__class__.cache_full_blocks = instrumented_cache_full_blocks
                        print("[DEBUG] Successfully installed block storage monitoring")
    except Exception as e:
        print(f"[ERROR] Failed to install monitoring hooks: {e}")
    
    # First, run new prompts to generate more cache blocks for testing
    print("\n[TEST] Generating additional content to populate cache...")
    print(f"[TEST] Running {len(cpu_cache_prompts)} new prompts...")
    
    for i, prompt in enumerate(cpu_cache_prompts):
        print(f"\n[TEST] Running additional prompt {i+1}...")
        outputs = llm.generate([prompt], sampling_params)
        result = outputs[0].outputs[0].text
        print(f"[RESULT] Generated additional content {i+1}")
        time.sleep(0.5)
        
        # After each prompt, check if blocks were stored
        if kv_cache_manager is not None:
            if hasattr(kv_cache_manager, 'remote_blocks'):
                print(f"[DEBUG] Remote blocks tracked: {len(kv_cache_manager.remote_blocks)}")
            if hasattr(kv_cache_manager, 'cpu_blocks'):
                print(f"[DEBUG] CPU blocks tracked: {len(kv_cache_manager.cpu_blocks)}")
    
    # Now force some blocks into CPU cache if we found the cache manager
    if kv_cache_manager is not None:
        print("\n[TEST] Forcing blocks from GPU to CPU cache...")
        if hasattr(kv_cache_manager, 'force_cache_to_cpu'):
            num_blocks_to_move = 10
            moved_blocks = kv_cache_manager.force_cache_to_cpu(num_blocks=num_blocks_to_move)
            print(f"[RESULT] Successfully moved {moved_blocks} blocks to CPU cache")
            
            # Print CPU cache metrics
            if hasattr(kv_cache_manager, 'get_metrics'):
                metrics = kv_cache_manager.get_metrics()
                if 'cpu_tier' in metrics:
                    cpu_metrics = metrics['cpu_tier']
                    print("\n[METRICS] CPU Cache After Forcing:")
                    print(f"  - Size: {cpu_metrics.get('size_mb', 0):.1f}MB / {cpu_metrics.get('max_size_mb', 0):.1f}MB")
                    print(f"  - Entries: {cpu_metrics.get('entries', 'N/A')}")
            
            # Now run the same prompts again to test CPU cache hits
            print("\n[TEST] Testing CPU cache hits by re-running prompts...")
            for i, prompt in enumerate(cpu_cache_prompts):
                print(f"\n[TEST] Re-running additional prompt {i+1} (should hit CPU cache)...")
                start_time = time.time()
                outputs = llm.generate([prompt], sampling_params)
                elapsed_time = time.time() - start_time
                result = outputs[0].outputs[0].text
                print(f"[RESULT] Completed in {elapsed_time:.2f} seconds")
        else:
            print("[ERROR] force_cache_to_cpu method not available in KVCacheManager")
    else:
        print("[ERROR] Could not access KV cache manager")

# PHASE 4: SUMMARY AND METRICS
print("\n" + "=" * 80)
print("PHASE 4: SUMMARY AND METRICS")
print("=" * 80)
print(f"[SUMMARY] First generation time: {p1_time:.2f} seconds")
print(f"[SUMMARY] Second generation time: {p2_time:.2f} seconds")
print(f"[SUMMARY] Third generation time: {p3_time:.2f} seconds")
print(f"[SUMMARY] First repeat time: {p4_time:.2f} seconds (speedup: {p1_speedup:.1f}%)")
print(f"[SUMMARY] Second repeat time: {p5_time:.2f} seconds (speedup: {p2_speedup:.1f}%)")
print(f"[SUMMARY] Third repeat time: {p6_time:.2f} seconds (speedup: {p3_speedup:.1f}%)")
print(f"[SUMMARY] {'Membrain tiered caching' if not args.disable_membrain else 'Standard prefix caching'} enabled")

# Display cache metrics if available
if not args.disable_membrain:
    try:
        # Get KV cache manager using consistent approach
        kv_cache_manager = None
        if hasattr(llm, '_llm_engine'):
            engine = llm._llm_engine
            if hasattr(engine, 'scheduler') and engine.scheduler is not None:
                scheduler = engine.scheduler
                if hasattr(scheduler, 'kv_cache_manager') and scheduler.kv_cache_manager is not None:
                    kv_cache_manager = scheduler.kv_cache_manager
        
        if kv_cache_manager is not None and hasattr(kv_cache_manager, 'get_metrics'):
            metrics = kv_cache_manager.get_metrics()
            print("\n[METRICS] Membrain Tiered Cache Statistics:")
            
            # GPU Tier metrics
            if 'gpu_tier' in metrics:
                gpu_metrics = metrics['gpu_tier']
                print("\n  GPU Tier Metrics:")
                print(f"  - Queries: {gpu_metrics.get('queries', 'N/A')}")
                print(f"  - Hits: {gpu_metrics.get('hits', 'N/A')}")
                print(f"  - Hit rate: {(gpu_metrics.get('hits', 0) / max(1, gpu_metrics.get('queries', 1)) * 100):.1f}%")
                print(f"  - Requests: {gpu_metrics.get('requests', 'N/A')}")
            
            # CPU Tier metrics
            if 'cpu_tier' in metrics:
                cpu_metrics = metrics['cpu_tier']
                print("\n  CPU Tier Metrics:")
                print(f"  - Size: {cpu_metrics.get('size_mb', 0):.1f}MB / {cpu_metrics.get('max_size_mb', 0):.1f}MB ({cpu_metrics.get('utilization', 0) * 100:.1f}%)")
                print(f"  - Entries: {cpu_metrics.get('entries', 'N/A')}")
                print(f"  - Hits: {cpu_metrics.get('hits', 'N/A')}")
                print(f"  - Misses: {cpu_metrics.get('misses', 'N/A')}")
                hit_rate = cpu_metrics.get('hit_rate', 0) * 100
                print(f"  - Hit rate: {hit_rate:.1f}%")
                print(f"  - Evictions: {cpu_metrics.get('evictions', 'N/A')}")
                
                print("\n  CPU Cache Access Stats:")
                cpu_load_attempts = getattr(kv_cache_manager, 'cpu_load_attempts', 0)
                cpu_load_successes = getattr(kv_cache_manager, 'cpu_load_successes', 0)
                cpu_store_attempts = getattr(kv_cache_manager, 'cpu_store_attempts', 0)
                cpu_store_successes = getattr(kv_cache_manager, 'cpu_store_successes', 0)
                print(f"  - Load attempts: {cpu_load_attempts}")
                print(f"  - Load successes: {cpu_load_successes}")
                print(f"  - Store attempts: {cpu_store_attempts}")
                print(f"  - Store successes: {cpu_store_successes}")
                cpu_load_rate = cpu_load_successes / max(1, cpu_load_attempts) * 100
                cpu_store_rate = cpu_store_successes / max(1, cpu_store_attempts) * 100
                print(f"  - Load success rate: {cpu_load_rate:.1f}%")
                print(f"  - Store success rate: {cpu_store_rate:.1f}%")
            
            # Remote Tier metrics  
            if 'remote_tier' in metrics:
                remote_metrics = metrics['remote_tier']
                print("\n  Remote Tier Metrics:")
                print(f"  - Store attempts: {remote_metrics.get('store_attempts', 'N/A')}")
                print(f"  - Store successes: {remote_metrics.get('store_successes', 'N/A')}")
                store_rate = 0
                if remote_metrics.get('store_attempts', 0) > 0:
                    store_rate = remote_metrics.get('store_successes', 0) / remote_metrics.get('store_attempts', 0) * 100
                print(f"  - Store success rate: {store_rate:.1f}%")
                print(f"  - Load attempts: {remote_metrics.get('load_attempts', 'N/A')}")
                print(f"  - Load successes: {remote_metrics.get('load_successes', 'N/A')}")
                load_rate = 0
                if remote_metrics.get('load_attempts', 0) > 0:
                    load_rate = remote_metrics.get('load_successes', 0) / remote_metrics.get('load_attempts', 0) * 100
                print(f"  - Load success rate: {load_rate:.1f}%")
                print(f"  - Remote blocks tracked: {remote_metrics.get('tracked_remote_blocks', 'N/A')}")
                
                # Remote server metrics if available
                if 'hits' in remote_metrics:
                    print(f"\n  Server-side Remote Metrics:")
                    print(f"  - Server hits: {remote_metrics.get('hits', 'N/A')}")
                    print(f"  - Server misses: {remote_metrics.get('misses', 'N/A')}")
                    hit_rate = remote_metrics.get('hit_rate', 0) * 100
                    print(f"  - Server hit rate: {hit_rate:.1f}%")
            
            # Test result determination
            remote_success = metrics.get('remote_tier', {}).get('store_successes', 0) > 0
            cpu_success = metrics.get('cpu_tier', {}).get('entries', 0) > 0
            
            if remote_success and cpu_success:
                print(f"\n[SUCCESS] Successfully used all cache tiers (GPU, CPU, and remote)!")
            elif remote_success:
                print(f"\n[PARTIAL SUCCESS] Successfully used GPU and remote tiers, but CPU tier might not be working")
            elif cpu_success:
                print(f"\n[PARTIAL SUCCESS] Successfully used GPU and CPU tiers, but remote tier might not be working")
                print(f"  Check that Membrain server is running at {os.environ.get('VLLM_MEMBRAIN_ENDPOINT')}")
            else:
                print(f"\n[FAILURE] Failed to use CPU or remote tiers")
                print(f"  Check that Membrain server is running at {os.environ.get('VLLM_MEMBRAIN_ENDPOINT')}")
        else:
            print("[ERROR] KV cache manager not found or get_metrics method not available")
    except Exception as e:
        print(f"[ERROR] Could not retrieve metrics: {e}")
        import traceback
        print(traceback.format_exc())

# PHASE 5: VALIDATION TEST FOR MEMBRAIN CONNECTIVITY
if not args.disable_membrain:
    print("\n" + "=" * 80)
    print("PHASE 5: VALIDATION TEST FOR MEMBRAIN CONNECTIVITY")
    print("=" * 80)
    
    print("\n[TEST] Validating Membrain connectivity with manual test...")
    try:
        # Get KV cache manager using consistent approach
        kv_cache_manager = None
        if hasattr(llm, '_llm_engine'):
            engine = llm._llm_engine
            if hasattr(engine, 'scheduler') and engine.scheduler is not None:
                scheduler = engine.scheduler
                if hasattr(scheduler, 'kv_cache_manager') and scheduler.kv_cache_manager is not None:
                    kv_cache_manager = scheduler.kv_cache_manager
        
        if kv_cache_manager is not None and hasattr(kv_cache_manager, 'membrain') and kv_cache_manager.membrain is not None:
            test_key = f"test-manual-{int(time.time())}"
            test_tensor = torch.ones((16, 16), dtype=torch.float16)
            
            print(f"  Creating test tensor with key: {test_key}")
            
            if hasattr(kv_cache_manager, '_event_loop') and kv_cache_manager._event_loop is not None:
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
                    print("\n[SUCCESS] Membrain connectivity confirmed!")
                else:
                    print(f"  Failed to retrieve manual test block!")
                    print("\n[FAILURE] Membrain service appears to be unreachable")
            else:
                print("[ERROR] Event loop not found in KV cache manager")
        else:
            print("[ERROR] Membrain client not found in KV cache manager")
    except Exception as e:
        print(f"  [ERROR] Manual connectivity test failed: {e}")
        import traceback
        print(traceback.format_exc())
        print("\n[FAILURE] Membrain service test failed, check server status")

print("\n[HELP] Check log messages for:")
print("  - GPU cache: 'get_computed_blocks' returns blocks")
print("  - CPU cache: 'CPU cache: Loaded block' messages")
print("  - Membrain: 'Successfully loaded block' messages")

print("\n" + "=" * 80)
print("TEST COMPLETE")
print("=" * 80)