#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""
Simple client for testing Membrain KV cache sharing system.
"""
import argparse
import time
import requests
import json

def generate_text(url, prompt, max_tokens=10):
    """Generate text using OpenAI API."""
    headers = {"Content-Type": "application/json"}
    data = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0
    }
    
    start_time = time.time()
    response = requests.post(f"{url}/v1/completions", headers=headers, json=data)
    duration = time.time() - start_time
    
    result = response.json() if response.status_code == 200 else {"error": response.text}
    return result, duration

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--producer", default="http://localhost:8000", help="Producer server URL")
    parser.add_argument("--consumer", default="http://localhost:8001", help="Consumer server URL")
    args = parser.parse_args()
    
    # Long prompt to demonstrate caching
    prompt = "Hello, how are you? " * 100
    
    # Generate with producer
    print("Sending request to producer...")
    producer_result, producer_time = generate_text(args.producer, prompt)
    if "error" in producer_result:
        print(f"Producer error: {producer_result['error']}")
        return
        
    generated_text = producer_result["choices"][0]["text"]
    print(f"Producer output: {generated_text!r}")
    print(f"Producer time: {producer_time:.2f}s")
    
    # Wait for KV cache to be stored
    print("Waiting for KV cache to be stored...")
    time.sleep(3)
    
    # Generate with consumer
    print("\nSending request to consumer...")
    consumer_result, consumer_time = generate_text(args.consumer, prompt)
    if "error" in consumer_result:
        print(f"Consumer error: {consumer_result['error']}")
        return
        
    generated_text = consumer_result["choices"][0]["text"]
    print(f"Consumer output: {generated_text!r}")
    print(f"Consumer time: {consumer_time:.2f}s")
    
    # Evaluate results
    speedup = producer_time / max(consumer_time, 0.001)
    print(f"\nSpeedup: {speedup:.2f}x")

if __name__ == "__main__":
    main()