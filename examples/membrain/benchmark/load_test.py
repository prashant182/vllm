#!/usr/bin/env python3
"""
Load testing script for vLLM with Membrain benchmarking
"""
import asyncio
import json
import random
import time
import argparse
import statistics
import numpy as np
from typing import List, Dict, Any, Tuple
import aiohttp
import pandas as pd
from tqdm import tqdm


# Common prefixes to use for testing prefix caching
COMMON_PREFIXES = [
    "Analyze the following text and provide a detailed summary: ",
    "I'm working on a research project about artificial intelligence. Here's an excerpt: ",
    "Review this document and extract the key points for discussion: ",
    "I need help understanding the following scientific paper abstract: ",
    "Translate the following content into simple terms that a high school student could understand: ",
]

# Content variations to create diversity
CONTENT_VARIATIONS = [
    "The history of deep learning models begins with neural networks, which were first proposed in the 1940s. "
    "Over decades, researchers made incremental improvements, but it wasn't until the early 2010s when deep "
    "learning saw a significant breakthrough with the success of AlexNet in the ImageNet competition. This marked "
    "the beginning of the deep learning revolution that has transformed fields ranging from computer vision to "
    "natural language processing. Subsequent advancements like transformers, introduced in 2017 with the paper "
    "'Attention Is All You Need,' have further accelerated progress, enabling the development of models like GPT, "
    "BERT, and T5.",
    
    "Climate change represents one of the most significant challenges facing humanity in the 21st century. "
    "Rising global temperatures have been linked to increased frequency and severity of extreme weather events, "
    "including hurricanes, droughts, and floods. The Intergovernmental Panel on Climate Change (IPCC) has "
    "projected that without substantial reductions in greenhouse gas emissions, global temperatures could rise "
    "by more than 1.5°C above pre-industrial levels by 2050, leading to catastrophic consequences for ecosystems "
    "and human societies worldwide. Addressing this challenge requires coordinated international action, "
    "technological innovation, and significant changes to energy systems.",
    
    "Quantum computing represents a paradigm shift in computational capabilities. Unlike classical computers "
    "that use bits to represent either 0 or 1, quantum computers utilize quantum bits, or qubits, which can "
    "exist in multiple states simultaneously due to the principle of superposition. This property, along with "
    "quantum entanglement, allows quantum computers to perform certain calculations exponentially faster than "
    "their classical counterparts. Potential applications include cryptography, drug discovery, materials "
    "science, and optimization problems. Despite recent advances, significant technical challenges remain in "
    "scaling quantum systems and reducing error rates."
]

# Create document templates with repetitive sections to maximize prefix caching potential
def create_document_template(length_multiplier: int = 3) -> str:
    """Create a document with repetitive sections to test prefix caching"""
    paragraphs = []
    
    for _ in range(length_multiplier):
        for content in CONTENT_VARIATIONS:
            paragraphs.append(content)
            
    return "\n\n".join(paragraphs)

def generate_prompts(num_prompts: int, prefix_reuse_rate: float = 0.7) -> List[str]:
    """
    Generate a list of diverse prompts with controlled prefix reuse
    
    Args:
        num_prompts: Number of prompts to generate
        prefix_reuse_rate: Rate at which prefixes should be reused (0.0-1.0)
        
    Returns:
        List of generated prompts
    """
    prompts = []
    document_template = create_document_template()
    
    for i in range(num_prompts):
        # Decide whether to reuse a prefix or create a unique one
        if i > 0 and random.random() < prefix_reuse_rate:
            # Reuse an existing prefix
            prefix = random.choice(COMMON_PREFIXES)
        else:
            # Create a unique prefix by adding unique identifiers
            prefix = random.choice(COMMON_PREFIXES) + f"(Unique-ID-{i}) "
        
        # Create the full prompt
        prompt = prefix + document_template
        
        # Sometimes truncate or extend the document to create variations
        if random.random() < 0.3:
            words = prompt.split()
            cutoff = random.randint(int(len(words) * 0.7), len(words))
            prompt = " ".join(words[:cutoff])
        
        prompts.append(prompt)
    
    return prompts

async def send_request(session: aiohttp.ClientSession, 
                      url: str, 
                      prompt: str, 
                      max_tokens: int = 100) -> Tuple[float, Dict]:
    """
    Send a request to the vLLM API and measure latency
    
    Returns:
        Tuple of (latency_seconds, response_json)
    """
    payload = {
        "model": "/models/Llama-3.3-70B-Instruct",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "max_tokens": max_tokens
    }
    
    start_time = time.time()
    async with session.post(url, json=payload) as response:
        response_json = await response.json()
        end_time = time.time()
        
    latency = end_time - start_time
    return latency, response_json

async def benchmark(url: str, 
                   prompts: List[str], 
                   concurrency: int = 5,
                   max_tokens: int = 100) -> pd.DataFrame:
    """
    Run benchmark against the given URL with the provided prompts
    
    Args:
        url: API endpoint URL
        prompts: List of prompts to send
        concurrency: Number of concurrent requests
        max_tokens: Maximum tokens to generate per request
        
    Returns:
        DataFrame with benchmark results
    """
    # First verify the API endpoint is responsive
    async with aiohttp.ClientSession() as session:
        try:
            # Check if the URL is for chat completions and convert to base URL
            base_url = url
            if base_url.endswith("/chat/completions"):
                base_url = base_url.rsplit("/chat", 1)[0]
            
            models_url = f"{base_url}/models"
            print(f"Checking models at: {models_url}")
            
            async with session.get(models_url) as response:
                if response.status != 200:
                    print(f"ERROR: API endpoint returned status {response.status}")
                    print(await response.text())
                    raise ValueError(f"API endpoint not ready: {url}")
                models_data = await response.json()
                print(f"Available models: {[model['id'] for model in models_data.get('data', [])]}")
        except Exception as e:
            print(f"ERROR: Failed to connect to API endpoint: {e}")
            raise
            
    results = []
    semaphore = asyncio.Semaphore(concurrency)
    
    async with aiohttp.ClientSession() as session:
        async def process_prompt(prompt_idx: int, prompt: str):
            async with semaphore:
                try:
                    latency, response = await send_request(session, url, prompt, max_tokens)
                    prompt_tokens = len(prompt.split())
                    
                    # Check for errors in the response
                    if "error" in response:
                        print(f"Error in response: {response['error']}")
                        return {
                            "prompt_idx": prompt_idx,
                            "prompt_tokens": prompt_tokens,
                            "completion_tokens": 0,
                            "latency": latency,
                            "success": False,
                            "error": response["error"].get("message", "Unknown error")
                        }
                    
                    # Extract completion tokens from response
                    completion = response.get("choices", [{}])[0].get("message", {}).get("content", "")
                    completion_tokens = len(completion.split()) if completion else 0
                    
                    # Print the first few responses to verify
                    if prompt_idx < 3:  # Only print first 3 for clarity
                        print(f"Sample response {prompt_idx}:")
                        print(f"  Status: Success")
                        print(f"  Completion: {completion[:50]}..." if completion else "No completion")
                        print(f"  Latency: {latency:.3f}s")
                    
                    return {
                        "prompt_idx": prompt_idx,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "latency": latency,
                        "success": True
                    }
                except Exception as e:
                    print(f"Error processing prompt {prompt_idx}: {str(e)}")
                    return {
                        "prompt_idx": prompt_idx,
                        "prompt_tokens": len(prompt.split()),
                        "completion_tokens": 0,
                        "latency": 0.0,
                        "success": False,
                        "error": str(e)
                    }
        
        # Process all prompts with progress bar
        tasks = [process_prompt(i, prompt) for i, prompt in enumerate(prompts)]
        for future in tqdm(asyncio.as_completed(tasks), total=len(tasks)):
            result = await future
            if result:
                results.append(result)
    
    return pd.DataFrame(results)

def analyze_results(df: pd.DataFrame) -> Dict[str, Any]:
    """Analyze benchmark results"""
    successful_requests = df[df.success == True]
    
    if len(successful_requests) == 0:
        return {
            "success_rate": 0.0,
            "total_requests": len(df),
            "successful_requests": 0,
            "error_rate": 1.0,
            "latency": {
                "mean": 0.0,
                "median": 0.0,
                "p90": 0.0,
                "p95": 0.0,
                "p99": 0.0,
                "min": 0.0,
                "max": 0.0,
                "std": 0.0
            },
            "tokens": {
                "total_prompt_tokens": 0,
                "total_completion_tokens": 0,
                "avg_prompt_tokens": 0.0,
                "avg_completion_tokens": 0.0,
                "throughput_tokens_per_second": 0.0
            }
        }
    
    # Calculate latency statistics
    latencies = successful_requests.latency.tolist()
    
    # Calculate token statistics
    total_prompt_tokens = successful_requests.prompt_tokens.sum()
    total_completion_tokens = successful_requests.completion_tokens.sum()
    total_tokens = total_prompt_tokens + total_completion_tokens
    
    # Calculate throughput
    total_time = sum(latencies)
    tokens_per_second = total_tokens / total_time if total_time > 0 else 0
    
    return {
        "success_rate": len(successful_requests) / len(df) if len(df) > 0 else 0.0,
        "total_requests": len(df),
        "successful_requests": len(successful_requests),
        "error_rate": 1.0 - (len(successful_requests) / len(df) if len(df) > 0 else 0.0),
        "latency": {
            "mean": statistics.mean(latencies) if latencies else 0.0,
            "median": statistics.median(latencies) if latencies else 0.0,
            "p90": np.percentile(latencies, 90) if latencies else 0.0,
            "p95": np.percentile(latencies, 95) if latencies else 0.0,
            "p99": np.percentile(latencies, 99) if latencies else 0.0,
            "min": min(latencies) if latencies else 0.0,
            "max": max(latencies) if latencies else 0.0,
            "std": statistics.stdev(latencies) if len(latencies) > 1 else 0.0
        },
        "tokens": {
            "total_prompt_tokens": int(total_prompt_tokens),
            "total_completion_tokens": int(total_completion_tokens),
            "avg_prompt_tokens": float(successful_requests.prompt_tokens.mean()),
            "avg_completion_tokens": float(successful_requests.completion_tokens.mean()),
            "throughput_tokens_per_second": float(tokens_per_second)
        }
    }

async def main():
    parser = argparse.ArgumentParser(description="vLLM Benchmark Tool")
    parser.add_argument("--url", type=str, required=True, 
                        help="API endpoint URL (e.g., http://vllm-membrain-service:8000/v1/chat/completions)")
    parser.add_argument("--num-prompts", type=int, default=100,
                        help="Number of prompts to generate")
    parser.add_argument("--concurrency", type=int, default=5,
                        help="Number of concurrent requests")
    parser.add_argument("--max-tokens", type=int, default=100,
                        help="Maximum tokens to generate per request")
    parser.add_argument("--prefix-reuse-rate", type=float, default=0.7,
                        help="Rate at which prefixes should be reused (0.0-1.0)")
    parser.add_argument("--output", type=str, default="benchmark_results.json",
                        help="Output file for benchmark results")
    
    args = parser.parse_args()
    
    print(f"Generating {args.num_prompts} prompts with prefix reuse rate of {args.prefix_reuse_rate}")
    prompts = generate_prompts(args.num_prompts, args.prefix_reuse_rate)
    
    print(f"Starting benchmark against {args.url} with concurrency {args.concurrency}")
    df = await benchmark(args.url, prompts, args.concurrency, args.max_tokens)
    
    # Save raw results
    df.to_csv(f"{args.output.split('.')[0]}_raw.csv", index=False)
    
    # Analyze and save results
    results = analyze_results(df)
    results["config"] = {
        "url": args.url,
        "num_prompts": args.num_prompts,
        "concurrency": args.concurrency,
        "max_tokens": args.max_tokens,
        "prefix_reuse_rate": args.prefix_reuse_rate
    }
    
    print("\n====== BENCHMARK RESULTS ======")
    print(f"Success rate: {results['success_rate']:.2%}")
    print(f"Total requests: {results['total_requests']}")
    print(f"Successful requests: {results['successful_requests']}")
    print("\n--- Latency (seconds) ---")
    print(f"Mean: {results['latency']['mean']:.3f}")
    print(f"Median: {results['latency']['median']:.3f}")
    print(f"p90: {results['latency']['p90']:.3f}")
    print(f"p95: {results['latency']['p95']:.3f}")
    print(f"p99: {results['latency']['p99']:.3f}")
    print(f"Min: {results['latency']['min']:.3f}")
    print(f"Max: {results['latency']['max']:.3f}")
    print("\n--- Token Statistics ---")
    print(f"Total prompt tokens: {results['tokens']['total_prompt_tokens']}")
    print(f"Total completion tokens: {results['tokens']['total_completion_tokens']}")
    print(f"Avg prompt tokens: {results['tokens']['avg_prompt_tokens']:.1f}")
    print(f"Avg completion tokens: {results['tokens']['avg_completion_tokens']:.1f}")
    print(f"Throughput: {results['tokens']['throughput_tokens_per_second']:.1f} tokens/sec")
    
    # Save results to file
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {args.output}")

if __name__ == "__main__":
    asyncio.run(main())
    