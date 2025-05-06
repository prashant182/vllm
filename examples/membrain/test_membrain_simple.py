#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simple test for Membrain KV cache sharing.
This script directly uses the LLM class instead of servers.
"""
import os
import time
import torch

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Enable vLLM v1 multiprocessing
os.environ["VLLM_USE_V1"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
os.environ["VLLM_LOGGING_LEVEL"] = "DEBUG"

# Membrain-related configuration 
membrain_endpoint = "http://localhost:9201"  # Membrain server endpoint
membrain_namespace = "vllm_kv_test_simple"  # Namespace for KV caches
chunk_size = 256  # Token chunk size for chunking

prompt = "Hello, how are you? " * 50  # Long prompt to demonstrate caching

def check_membrain_server():
    """Check if Membrain server is running."""
    import requests
    try:
        response = requests.get(f"{membrain_endpoint}/healthz", verify=False)
        if response.status_code == 200:
            print("Membrain server is healthy and ready.")
            return True
        else:
            print(f"Membrain server returned status code {response.status_code}.")
            return False
    except Exception as e:
        print(f"Failed to connect to Membrain server: {e}")
        return False

def main():
    # Check if Membrain server is running
    if not check_membrain_server():
        print(f"Membrain server not available at {membrain_endpoint}.")
        print("Please make sure the server is running before continuing.")
        return
    
    # Use a small model that doesn't require authorization
    model = "facebook/opt-125m"
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)
    
    # First run as producer
    print("\n=== Running as producer ===")
    
    producer_config = KVTransferConfig.from_cli(
        f'{{"kv_connector":"MembrainConnectorV1", "kv_role":"kv_producer", '
        f'"kv_connector_extra_config": {{"membrain_endpoint":"{membrain_endpoint}", '
        f'"membrain_namespace":"{membrain_namespace}", "chunk_size":{chunk_size}, '
        f'"max_chunk_size":10485760}}}}'
    )
    
    print("Initializing producer LLM...")
    producer_llm = LLM(
        model=model,
        kv_transfer_config=producer_config,
        max_model_len=2048,
        enforce_eager=True
    )
    
    print(f"Generating with prompt length: {len(prompt)}")
    start_time = time.time()
    outputs = producer_llm.generate([prompt], sampling_params)
    producer_time = time.time() - start_time
    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Producer generated text: {generated_text!r}")
    print(f"Producer generation took {producer_time:.2f} seconds")
    
    # Wait for KV cache to be stored in Membrain
    print("\nWaiting for KV cache to be stored...")
    time.sleep(3)
    
    # Now run as consumer
    print("\n=== Running as consumer ===")
    
    # Clean up CUDA memory
    del producer_llm
    torch.cuda.empty_cache()
    
    consumer_config = KVTransferConfig.from_cli(
        f'{{"kv_connector":"MembrainConnectorV1", "kv_role":"kv_consumer", '
        f'"kv_connector_extra_config": {{"membrain_endpoint":"{membrain_endpoint}", '
        f'"membrain_namespace":"{membrain_namespace}", "chunk_size":{chunk_size}, '
        f'"max_chunk_size":10485760}}}}'
    )
    
    print("Initializing consumer LLM...")
    consumer_llm = LLM(
        model=model, 
        kv_transfer_config=consumer_config,
        max_model_len=2048,
        enforce_eager=True
    )
    
    print(f"Generating with prompt length: {len(prompt)}")
    start_time = time.time()
    outputs = consumer_llm.generate([prompt], sampling_params)
    consumer_time = time.time() - start_time
    
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Consumer generated text: {generated_text!r}")
    print(f"Consumer generation took {consumer_time:.2f} seconds")
    
    # Show performance comparison
    print("\n=== Performance Comparison ===")
    speedup = producer_time / max(consumer_time, 0.001)
    print(f"Producer time: {producer_time:.2f}s, Consumer time: {consumer_time:.2f}s")
    print(f"Speedup from using cached KV: {speedup:.2f}x")
    
    if speedup > 1.5:
        print("✓ KV cache sharing appears to be working correctly!")
    else:
        print("⚠ KV cache sharing might not be working optimally.")
        print("  The consumer should be faster than the producer due to cached KV.")

if __name__ == "__main__":
    main()