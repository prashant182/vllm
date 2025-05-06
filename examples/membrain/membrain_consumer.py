#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Consumer client for Membrain KV cache sharing system.
This script starts a vLLM server configured as a KV cache consumer,
which will retrieve KV caches from a Membrain server.
"""
import os
import time
import subprocess
import argparse
import json
import requests

def check_membrain_server(membrain_endpoint):
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

def parse_args():
    parser = argparse.ArgumentParser(description="Start a vLLM server configured as a KV cache consumer")
    parser.add_argument("--model", type=str, default="facebook/opt-125m",
                        help="Model to use (default: facebook/opt-125m)")
    parser.add_argument("--port", type=int, default=8001,
                        help="Port for the server (default: 8001)")
    parser.add_argument("--gpu", type=str, default="1",
                        help="GPU device to use (default: 1)")
    parser.add_argument("--membrain-endpoint", type=str, default="http://localhost:9201",
                        help="Membrain server endpoint (default: http://localhost:9201)")
    parser.add_argument("--membrain-namespace", type=str, default="vllm_kv",
                        help="Namespace for KV caches (default: vllm_kv)")
    parser.add_argument("--chunk-size", type=int, default=256,
                        help="Token chunk size for chunking (default: 256)")
    parser.add_argument("--verbose", action="store_true",
                        help="Enable verbose logging")
    parser.add_argument("--max-model-len", type=int, default=2048,
                        help="Maximum model sequence length (default: 2048)")
    parser.add_argument("--tensor-parallel-size", type=int, default=1,
                        help="Tensor parallel size (default: 1)")
    parser.add_argument("--both", action="store_true",
                        help="Configure as both producer and consumer (default: consumer only)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Check if Membrain server is running
    if not check_membrain_server(args.membrain_endpoint):
        print(f"Membrain server not available at {args.membrain_endpoint}.")
        print("Please make sure the server is running before continuing.")
        return
    
    # KV transfer config for consumer
    kv_config = {
        "kv_connector": "MembrainConnectorV1",
        "kv_role": "kv_both" if args.both else "kv_consumer",
        "kv_connector_extra_config": {
            "membrain_endpoint": args.membrain_endpoint,
            "membrain_namespace": args.membrain_namespace,
            "chunk_size": args.chunk_size,
        }
    }
    
    # Set up environment
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = args.gpu
    env["VLLM_USE_V1"] = "1"
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    
    if args.verbose:
        env["VLLM_LOGGING_LEVEL"] = "DEBUG"
        env["VLLM_CONFIGURE_LOGGING"] = "1"
    
    # Build command
    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model,
        "--host", "0.0.0.0",
        "--port", str(args.port),
        "--kv-transfer-config", json.dumps(kv_config),
        "--enable-prefix-caching",
        "--max-model-len", str(args.max_model_len),
    ]
    
    if args.tensor_parallel_size > 1:
        cmd.extend(["--tensor-parallel-size", str(args.tensor_parallel_size)])
    
    # Start the server
    role = "consumer/producer" if args.both else "consumer"
    print(f"Starting {role} server on port {args.port} with model {args.model}...")
    print("Server command:", " ".join(cmd))
    
    try:
        process = subprocess.Popen(
            cmd,
            env=env,
            text=True
        )
        
        # Keep the server running until interrupted
        print("Server is starting... Press Ctrl+C to stop.")
        process.wait()
    
    except KeyboardInterrupt:
        print("\nShutting down server...")
        process.terminate()
        process.wait()
        print("Server stopped.")

if __name__ == "__main__":
    main()