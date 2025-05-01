# SPDX-License-Identifier: Apache-2.0

import os
import asyncio
import torch
from vllm import LLM, SamplingParams
from vllm.distributed import cleanup_dist_env_and_memory
from vllm.v1.core.membrain import MembrainConfig, MembrainStore
from vllm.v1.core.membrain_kvmanager import MembrainKVConfig, MembrainKVCacheManager

# Ensure V1 is used
os.environ["VLLM_USE_V1"] = "1"

# Common prefix for testing cache hits
prefix = (
    "You are an expert school principal, skilled in effectively managing "
    "faculty and staff. Draft 10-15 questions for a potential first grade "
    "Head Teacher for my K-12, all-girls', independent school that emphasizes "
    "community, joyful discovery, and life-long learning."
)

# Test prompts with common prefix
prompts = [
    prefix + " Focus on classroom management experience.",
    prefix + " Focus on curriculum development experience.",
    prefix + " Focus on parent communication skills.",
]

async def test_membrain_store():
    """Test basic Membrain operations"""
    print("\nTesting Membrain Store operations...")
    
    config = MembrainConfig(
        endpoint="http://localhost:9201",
        namespace="test",
        enable_metrics=True
    )
    
    store = MembrainStore(
        config=config,
        node_id="test-node",
        block_size=16,
        dtype=torch.float16
    )

    # Test store/load
    test_tensor = torch.randn(1, 16, dtype=torch.float16)
    success = await store.store_block(
        "test-block-1",
        test_tensor,
        {"test": True}
    )
    print(f"Store success: {success}")

    loaded_tensor = await store.load_block("test-block-1")
    if loaded_tensor is not None:
        print("Load successful")
        print(f"Tensor match: {torch.allclose(test_tensor, loaded_tensor)}")
    
    # Test ref counting
    ref_count = await store.increment_ref("test-block-1")
    print(f"Ref count after increment: {ref_count}")
    
    ref_count = await store.decrement_ref("test-block-1")
    print(f"Ref count after decrement: {ref_count}")

    # Check metrics
    print("\nMetrics:", store.get_metrics())
    
    await store.close()

def main():
    """Run end-to-end test with prefix caching"""
    print("Starting end-to-end prefix caching test...")

    # Test basic Membrain operations first
    asyncio.run(test_membrain_store())

    # Create sampling params
    sampling_params = SamplingParams(temperature=0.0)

    print("\nTesting with Membrain disabled first...")
    
    # Create LLM without Membrain
    regular_llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True
    )

    # Generate baseline results
    outputs = regular_llm.generate(prompts, sampling_params)
    
    regular_outputs = []
    print("\nRegular outputs:")
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        regular_outputs.append(generated_text)
        print(f"Prompt: {prompt[:100]}...")
        print(f"Generated: {generated_text[:100]}...")
        print("-" * 50)

    # Cleanup
    del regular_llm
    cleanup_dist_env_and_memory()

    print("\nTesting with Membrain enabled...")

    # Create Membrain config
    membrain_config = MembrainKVConfig(
        membrain=MembrainConfig(
            endpoint="http://localhost:9201",
            namespace="vllm-cache",
            enable_metrics=True
        ),
        enable_metrics=True
    )

    # Create LLM with Membrain
    membrain_llm = LLM(
        model="facebook/opt-125m",
        gpu_memory_utilization=0.4,
        enable_prefix_caching=True,
        membrain_config=membrain_config  # Enable Membrain
    )

    # Warmup to cache the prefix
    membrain_llm.generate(prompts[0], sampling_params)

    # Generate with Membrain caching
    outputs = membrain_llm.generate(prompts, sampling_params)

    membrain_outputs = []
    print("\nMembrain outputs:")
    print("-" * 50)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        membrain_outputs.append(generated_text)
        print(f"Prompt: {prompt[:100]}...")
        print(f"Generated: {generated_text[:100]}...")
        print("-" * 50)

    # Compare outputs
    outputs_match = all(
        regular_outputs[i] == membrain_outputs[i]
        for i in range(len(prompts))
    )
    print(f"\nOutputs match: {outputs_match}")

    # Print Membrain metrics
    print("\nMembrain metrics:", membrain_llm.get_membrain_metrics())

if __name__ == "__main__":
    main()