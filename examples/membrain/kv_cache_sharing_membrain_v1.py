# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of remote KV cache sharing
with Membrain.
We will launch 2 vllm instances, and KV cache is transferred as follows: 
(1) vLLM instance 1 -> Membrain server (KV cache store).
(2) Membrain server -> vLLM instance 2 (KV cache reuse/retrieve).

Note: Ensure a Membrain server is running at the endpoint specified.
The default endpoint is http://localhost:9201.
"""
import os
import time
from multiprocessing import Event, Process

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

# Enable vLLM v1 multiprocessing
os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "1"
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# Membrain-related configuration 
membrain_endpoint = "http://localhost:9201"  # Membrain server endpoint
membrain_namespace = "vllm_kv"  # Namespace for KV caches
chunk_size = 256  # Token chunk size for chunking

prompts = [
    "Hello, how are you?" * 100,  # Long prompt to demonstrate caching
]


def run_store(store_done, prompts):
    # We use GPU 0 for KV cache store process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig.from_cli(
        f'{{"kv_connector":"MembrainConnectorV1", "kv_role":"kv_both", '
        f'"kv_connector_extra_config": {{"membrain_endpoint":"{membrain_endpoint}", '
        f'"membrain_namespace":"{membrain_namespace}", "chunk_size":{chunk_size}}}}}')
    
    # Use a small open-source model that doesn't require authorization
    llm = LLM(model="facebook/opt-125m",
              kv_transfer_config=ktc,
              max_model_len=2048,
              gpu_memory_utilization=0.8,
              enforce_eager=True)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text from store process: {generated_text!r}")
    print("KV cache store is finished.")
    store_done.set()


def run_retrieve(store_done, prompts, timeout=1):
    # We use GPU 1 for KV cache retrieve process.
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=10)

    ktc = KVTransferConfig.from_cli(
        f'{{"kv_connector":"MembrainConnectorV1", "kv_role":"kv_both", '
        f'"kv_connector_extra_config": {{"membrain_endpoint":"{membrain_endpoint}", '
        f'"membrain_namespace":"{membrain_namespace}", "chunk_size":{chunk_size}}}}}')
    
    # Use a small open-source model that doesn't require authorization
    llm = LLM(model="facebook/opt-125m",
              kv_transfer_config=ktc,
              max_model_len=2048,
              gpu_memory_utilization=0.8,
              enforce_eager=True)

    print("Waiting for KV cache store to finish...")
    store_done.wait()
    time.sleep(timeout)

    outputs = llm.generate(prompts, sampling_params)
    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text from retrieve process: {generated_text!r}")


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

    store_done = Event()
    store_process = Process(target=run_store, args=(store_done, prompts))
    retrieve_process = Process(target=run_retrieve, args=(store_done, prompts))

    # Start KV cache store process
    print("Starting store process...")
    store_process.start()

    # Start KV cache retrieve process
    print("Starting retrieve process...")
    retrieve_process.start()

    # Clean up the processes
    store_process.join()
    retrieve_process.join()
    print("All processes finished.")


if __name__ == "__main__":
    main()