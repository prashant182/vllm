#!/usr/bin/env python3
import argparse
import asyncio
import json
import random
import statistics
import time
from typing import List, Dict, Any

import aiohttp
import numpy as np


class PrefixCachingBenchmark:
    def __init__(
        self,
        api_url: str = "http://localhost:8083/v1/completions",
        model: str = "/models",
        max_tokens: int = 128,
        temperature: float = 0.7,
        num_requests: int = 100,
        concurrency: int = 10,
        prompt_file: str = None,
        repetition_ratio: float = 0.5,
    ):
        self.api_url = api_url
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.num_requests = num_requests
        self.concurrency = concurrency
        self.prompt_file = prompt_file
        self.repetition_ratio = repetition_ratio
        self.prompts = []
        self.results = []
        self.semaphore = asyncio.Semaphore(concurrency)

    def load_prompts(self):
        """Load prompts from file or generate sample prompts"""
        if self.prompt_file:
            with open(self.prompt_file, "r") as f:
                self.prompts = [line.strip() for line in f if line.strip()]
        else:
            # Generate some sample prompts
            base_prompts = [
                "Explain the concept of prefix caching in language models.",
                "What are the benefits of using transformers for NLP tasks?",
                "Summarize the key innovations in the Llama model architecture.",
                "How does attention mechanism work in transformer models?",
                "Explain the concept of TTFT (Time To First Token) in LLMs.",
                "What are the main differences between Llama and GPT models?",
                "Describe the key features of vLLM's paged attention mechanism.",
                "How does continuous batching improve LLM inference throughput?",
                "Explain the concept of KV caching in language model inference.",
                "What are the challenges in serving large language models efficiently?",
            ]

            # Generate prompts with prefix variations to test caching
            for prompt in base_prompts:
                self.prompts.append(prompt)
                # Add variations with common prefixes
                self.prompts.append(f"{prompt} Provide examples.")
                self.prompts.append(f"{prompt} Compare with other approaches.")

        # Ensure we have enough prompts
        if len(self.prompts) < self.num_requests:
            self.prompts = (
                self.prompts * (self.num_requests // len(self.prompts) + 1)
            )[:self.num_requests]

    def prepare_request_sequence(self) -> List[str]:
        """Prepare sequence of requests with controlled prompt repetition"""
        request_sequence = []
        
        # First pass: select original prompts
        original_count = int(self.num_requests * (1 - self.repetition_ratio))
        original_prompts = random.sample(self.prompts, min(original_count, len(self.prompts)))
        request_sequence.extend(original_prompts)
        
        # Second pass: add repeated prompts to benefit from caching
        remaining = self.num_requests - len(request_sequence)
        if remaining > 0:
            repeated_prompts = random.choices(original_prompts, k=remaining)
            request_sequence.extend(repeated_prompts)
        
        # Shuffle to create a more realistic pattern
        random.shuffle(request_sequence)
        return request_sequence

    async def send_request(self, prompt: str, session: aiohttp.ClientSession) -> Dict[str, Any]:
        """Send a single completion request and measure timing"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
        }
        
        start_time = time.time()
        first_token_time = None
        completion_time = None
        
        async with self.semaphore:
            try:
                async with session.post(
                    self.api_url, json=payload, timeout=60
                ) as response:
                    # Mark time for first token (when we get the response headers)
                    first_token_time = time.time()
                    
                    response_json = await response.json()
                    completion_time = time.time()
                    
                    return {
                        "prompt": prompt,
                        "prompt_length": len(prompt.split()),
                        "ttft": first_token_time - start_time,
                        "total_time": completion_time - start_time,
                        "throughput": self.max_tokens / (completion_time - first_token_time) if first_token_time else 0,
                        "success": True,
                        "response": response_json,
                    }
                    
            except Exception as e:
                return {
                    "prompt": prompt,
                    "prompt_length": len(prompt.split()),
                    "ttft": None,
                    "total_time": None,
                    "throughput": 0,
                    "success": False,
                    "error": str(e),
                }

    async def run_benchmark(self):
        """Run the benchmark with the specified parameters"""
        self.load_prompts()
        request_prompts = self.prepare_request_sequence()
        
        print(f"Running benchmark with {self.num_requests} requests ({self.repetition_ratio*100:.1f}% repeated prompts)")
        print(f"Concurrency: {self.concurrency}")
        
        async with aiohttp.ClientSession() as session:
            tasks = []
            for prompt in request_prompts:
                tasks.append(self.send_request(prompt, session))
            
            print("Sending requests...")
            self.results = await asyncio.gather(*tasks)

    def analyze_results(self):
        """Analyze the benchmark results"""
        successful_results = [r for r in self.results if r["success"]]
        failed_results = [r for r in self.results if not r["success"]]
        
        if not successful_results:
            print("No successful requests to analyze!")
            return
        
        ttft_values = [r["ttft"] for r in successful_results]
        total_times = [r["total_time"] for r in successful_results]
        throughput_values = [r["throughput"] for r in successful_results]
        
        # Identify unique vs repeated prompts
        unique_prompts = set()
        first_occurrences = []
        repeated_occurrences = []
        
        for result in successful_results:
            prompt = result["prompt"]
            if prompt not in unique_prompts:
                unique_prompts.add(prompt)
                first_occurrences.append(result)
            else:
                repeated_occurrences.append(result)
        
        # Calculate metrics
        metrics = {
            "total_requests": len(self.results),
            "successful_requests": len(successful_results),
            "failed_requests": len(failed_results),
            "success_rate": len(successful_results) / len(self.results) if self.results else 0,
            
            "avg_ttft": statistics.mean(ttft_values),
            "median_ttft": statistics.median(ttft_values),
            "p90_ttft": np.percentile(ttft_values, 90),
            "p95_ttft": np.percentile(ttft_values, 95),
            "p99_ttft": np.percentile(ttft_values, 99),
            
            "avg_total_time": statistics.mean(total_times),
            "median_total_time": statistics.median(total_times),
            "avg_throughput": statistics.mean(throughput_values),
            
            "unique_prompts": len(first_occurrences),
            "repeated_prompts": len(repeated_occurrences),
        }
        
        # Compare first vs repeated occurrences (for caching benefit analysis)
        if first_occurrences and repeated_occurrences:
            first_ttft = [r["ttft"] for r in first_occurrences]
            repeated_ttft = [r["ttft"] for r in repeated_occurrences]
            
            metrics.update({
                "first_occurrence_avg_ttft": statistics.mean(first_ttft),
                "repeated_occurrence_avg_ttft": statistics.mean(repeated_ttft),
                "ttft_improvement_ratio": statistics.mean(first_ttft) / statistics.mean(repeated_ttft)
                    if statistics.mean(repeated_ttft) > 0 else 0,
                "ttft_improvement_ms": (statistics.mean(first_ttft) - statistics.mean(repeated_ttft)) * 1000,
            })
        
        # Print results
        print("\n===== BENCHMARK RESULTS =====")
        print(f"Total Requests: {metrics['total_requests']}")
        print(f"Success Rate: {metrics['success_rate']*100:.2f}%")
        print(f"Failed Requests: {metrics['failed_requests']}")
        
        print("\n----- TTFT (Time to First Token) -----")
        print(f"Average TTFT: {metrics['avg_ttft']*1000:.2f} ms")
        print(f"Median TTFT: {metrics['median_ttft']*1000:.2f} ms")
        print(f"P90 TTFT: {metrics['p90_ttft']*1000:.2f} ms")
        print(f"P95 TTFT: {metrics['p95_ttft']*1000:.2f} ms")
        print(f"P99 TTFT: {metrics['p99_ttft']*1000:.2f} ms")
        
        print("\n----- Overall Performance -----")
        print(f"Average Total Time: {metrics['avg_total_time']*1000:.2f} ms")
        print(f"Average Throughput: {metrics['avg_throughput']:.2f} tokens/sec")
        
        if "ttft_improvement_ratio" in metrics:
            print("\n----- Prefix Caching Impact -----")
            print(f"First Occurrence Avg TTFT: {metrics['first_occurrence_avg_ttft']*1000:.2f} ms")
            print(f"Repeated Occurrence Avg TTFT: {metrics['repeated_occurrence_avg_ttft']*1000:.2f} ms")
            print(f"TTFT Improvement: {metrics['ttft_improvement_ms']:.2f} ms")
            print(f"TTFT Speedup: {metrics['ttft_improvement_ratio']:.2f}x")
        
        # Save detailed results to file
        with open("prefix_cache_benchmark_results.json", "w") as f:
            json.dump({
                "config": {
                    "api_url": self.api_url,
                    "model": self.model,
                    "max_tokens": self.max_tokens,
                    "temperature": self.temperature,
                    "num_requests": self.num_requests,
                    "concurrency": self.concurrency,
                    "repetition_ratio": self.repetition_ratio,
                },
                "metrics": metrics,
                "raw_results": [
                    {k: v for k, v in r.items() if k != "response"}
                    for r in self.results
                ]
            }, f, indent=2)
        
        print(f"\nDetailed results saved to prefix_cache_benchmark_results.json")


async def main():
    parser = argparse.ArgumentParser(description="Benchmark vLLM with prefix caching")
    parser.add_argument("--api-url", default="http://localhost:8083/v1/completions", 
                        help="API endpoint URL")
    parser.add_argument("--model", default="/models", 
                        help="Model name or path")
    parser.add_argument("--max-tokens", type=int, default=128, 
                        help="Maximum number of tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.7, 
                        help="Sampling temperature")
    parser.add_argument("--num-requests", type=int, default=100, 
                        help="Number of requests to send")
    parser.add_argument("--concurrency", type=int, default=10, 
                        help="Number of concurrent requests")
    parser.add_argument("--prompt-file", 
                        help="File containing prompts (one per line)")
    parser.add_argument("--repetition-ratio", type=float, default=0.5, 
                        help="Ratio of repeated prompts (0.0 - 1.0)")
    
    args = parser.parse_args()
    
    benchmark = PrefixCachingBenchmark(
        api_url=args.api_url,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        num_requests=args.num_requests,
        concurrency=args.concurrency,
        prompt_file=args.prompt_file,
        repetition_ratio=args.repetition_ratio,
    )
    
    await benchmark.run_benchmark()
    benchmark.analyze_results()


if __name__ == "__main__":
    asyncio.run(main())