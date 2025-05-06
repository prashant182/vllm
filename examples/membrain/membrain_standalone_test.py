#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the usage of remote KV cache sharing with Membrain
using the standalone OpenAI-compatible API server.

We will launch 2 vLLM API server instances, and KV cache is transferred as follows:
(1) vLLM server 1 (producer) -> Membrain server (KV cache store)
(2) Membrain server -> vLLM server 2 (consumer) (KV cache reuse/retrieve)

Note: Ensure a Membrain server is running at the endpoint specified.
The default endpoint is http://localhost:9201.
"""
import os
import time
import subprocess
import signal
import requests
import json
from multiprocessing import Event, Process

# Membrain-related configuration
membrain_endpoint = "http://localhost:9201"  # Membrain server endpoint
membrain_namespace = "vllm_kv_test"  # Namespace for KV caches
chunk_size = 256  # Token chunk size for chunking

# Define ports for the two servers
producer_port = 8000
consumer_port = 8001

# Define a long prompt to demonstrate caching
prompt = "Hello, how are you? " * 100  # Long prompt to demonstrate caching

def check_membrain_server():
    """Check if Membrain server is running."""
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

def start_producer_server():
    """Start the producer vLLM server."""
    # KV transfer config for producer
    kv_config = {
        "kv_connector": "MembrainConnectorV1",
        "kv_role": "kv_producer",
        "kv_connector_extra_config": {
            "membrain_endpoint": membrain_endpoint,
            "membrain_namespace": membrain_namespace,
            "chunk_size": chunk_size
        }
    }
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["VLLM_USE_V1"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    
    # Use a small open-source model that doesn't require authorization
    model = "facebook/opt-125m"
    
    # Start the server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(producer_port),
        "--kv-transfer-config", json.dumps(kv_config),
        "--enable-prefix-caching",
        "--max-model-len", "2048"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        env=env,
        # Don't pipe stdout/stderr to capture them directly
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def start_consumer_server():
    """Start the consumer vLLM server."""
    # KV transfer config for consumer
    kv_config = {
        "kv_connector": "MembrainConnectorV1",
        "kv_role": "kv_consumer",
        "kv_connector_extra_config": {
            "membrain_endpoint": membrain_endpoint,
            "membrain_namespace": membrain_namespace,
            "chunk_size": chunk_size
        }
    }
    
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"  # Use same GPU as producer since we're testing
    env["VLLM_USE_V1"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    env["VLLM_LOGGING_LEVEL"] = "DEBUG"
    
    # Use a small open-source model that doesn't require authorization
    model = "facebook/opt-125m"
    
    # Start the server
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", model,
        "--host", "127.0.0.1",
        "--port", str(consumer_port),
        "--kv-transfer-config", json.dumps(kv_config),
        "--enable-prefix-caching",
        "--max-model-len", "2048"
    ]
    
    print(f"Running command: {' '.join(cmd)}")
    
    process = subprocess.Popen(
        cmd,
        env=env,
        # Don't pipe stdout/stderr to capture them directly
        # stdout=subprocess.PIPE,
        # stderr=subprocess.PIPE,
        text=True
    )
    
    return process

def wait_for_server(port, timeout=30):
    """Wait for the server to start."""
    start_time = time.time()
    while time.time() - start_time < timeout:
        try:
            response = requests.get(f"http://127.0.0.1:{port}/v1/models")
            if response.status_code == 200:
                print(f"Server on port {port} is ready.")
                return True
        except requests.exceptions.RequestException:
            time.sleep(1)
    
    print(f"Timeout waiting for server on port {port}")
    return False

def generate_with_producer():
    """Generate text with the producer server."""
    url = f"http://127.0.0.1:{producer_port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0
    }
    
    print("Sending request to producer server...")
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["text"]
            print(f"Generated text from producer: {generated_text!r}")
            print(f"Producer generation took {duration:.2f} seconds")
            return result
        else:
            print(f"Error from producer server: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception while generating from producer: {e}")
        return None

def generate_with_consumer():
    """Generate text with the consumer server."""
    url = f"http://127.0.0.1:{consumer_port}/v1/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": 10,
        "temperature": 0
    }
    
    print("Sending request to consumer server...")
    start_time = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, timeout=60)
        duration = time.time() - start_time
        
        if response.status_code == 200:
            result = response.json()
            generated_text = result["choices"][0]["text"]
            print(f"Generated text from consumer: {generated_text!r}")
            print(f"Consumer generation took {duration:.2f} seconds (should be faster if cache is shared)")
            return result
        else:
            print(f"Error from consumer server: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Exception while generating from consumer: {e}")
        return None

def main():
    # Check if Membrain server is running
    if not check_membrain_server():
        print(f"Membrain server not available at {membrain_endpoint}.")
        print("Please make sure the server is running before continuing.")
        return
    
    try:
        # Start the producer server
        print("Starting producer server...")
        producer_process = start_producer_server()
        
        # Wait for producer to start
        if not wait_for_server(producer_port):
            print("Failed to start producer server")
            producer_process.terminate()
            return
        
        # Start the consumer server
        print("Starting consumer server...")
        consumer_process = start_consumer_server()
        
        # Wait for consumer to start
        if not wait_for_server(consumer_port):
            print("Failed to start consumer server")
            producer_process.terminate()
            consumer_process.terminate()
            return
        
        # Give servers a moment to fully initialize
        time.sleep(5)
        
        # Generate text with producer (will store KV cache)
        producer_result = generate_with_producer()
        
        if producer_result:
            # Give Membrain time to process
            print("Waiting for KV cache to be stored...")
            time.sleep(3)
            
            # Generate text with consumer (should reuse KV cache)
            consumer_result = generate_with_consumer()

    except KeyboardInterrupt:
        print("Interrupted by user")
    finally:
        # Clean up
        print("Shutting down servers...")
        if 'producer_process' in locals():
            producer_process.terminate()
            producer_process.wait()
        
        if 'consumer_process' in locals():
            consumer_process.terminate()
            consumer_process.wait()
        
        print("All processes finished.")

if __name__ == "__main__":
    main()