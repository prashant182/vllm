# Membrain Distributed Caching for vLLM

This implementation adds support for Membrain distributed caching to vLLM v1, allowing you to share prefix caching across multiple vLLM instances.

## Overview

Membrain is a distributed key-value store that allows sharing prefix cache blocks between multiple vLLM instances. This implementation:

1. Works with the existing vLLM v1 interface
2. Can be enabled via environment variables
3. Falls back to regular prefix caching when not enabled
4. Requires a running Membrain server at the specified endpoint

## Usage

### Option 1: Use the run_with_membrain.sh script

```bash
./run_with_membrain.sh python -m vllm.entrypoints.llm --model facebook/opt-125m
```

### Option 2: Set environment variables manually

```bash
export VLLM_USE_V1=1
export VLLM_MEMBRAIN_ENABLED=1
export VLLM_MEMBRAIN_ENDPOINT="http://localhost:9201"
export VLLM_MEMBRAIN_NAMESPACE="default"

# Then run your vLLM command
python -m vllm.entrypoints.llm --model your-model-name
```

## Configuration

You can configure Membrain through these environment variables:

- `VLLM_USE_V1`: Must be set to "1" to use vLLM v1 (required for Membrain)
- `VLLM_MEMBRAIN_ENABLED`: Set to "1" to enable Membrain (default: "0")
- `VLLM_MEMBRAIN_ENDPOINT`: Membrain server URL (default: "http://localhost:9201")
- `VLLM_MEMBRAIN_NAMESPACE`: Namespace for storing blocks (default: "default")

## Testing

You can test the implementation with a small model:

```bash
./run_with_membrain.sh python run_membrain_test.py --model facebook/opt-125m
```

## Implementation Details

The implementation adds minimal changes to vLLM:

1. Modified `vllm/v1/core/sched/scheduler.py` to support initializing Membrain KV cache manager
2. Added proper handling for async methods in synchronous context
3. Used environment variables for configuration

## File Structure

- `vllm/v1/core/membrain.py`: Core Membrain store implementation
- `vllm/v1/core/membrain_kvmanager.py`: KV cache manager implementation for Membrain
- `vllm/v1/core/membrain_manager.py`: Manager for distributed memory tier
- `vllm/v1/core/sched/scheduler.py`: Modified to use Membrain when enabled

## Limitations

- Currently, async methods are wrapped in synchronous code
- Requires a running Membrain server
- Metrics reporting is limited

## Future Work

- Implement full async operation in a more elegant way
- Add better metrics and monitoring
- Optimize serialization/deserialization of tensors
- Add support for more advanced eviction policies