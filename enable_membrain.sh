#!/bin/bash
# Enable vLLM V1 and set up Membrain configuration
export VLLM_USE_V1=1

# Membrain configuration
export VLLM_MEMBRAIN_ENABLED=1
export VLLM_MEMBRAIN_ENDPOINT="http://localhost:9201"
export VLLM_MEMBRAIN_NAMESPACE="default"

echo "vLLM V1 and Membrain enabled with the following configuration:"
echo "VLLM_USE_V1=$VLLM_USE_V1"
echo "VLLM_MEMBRAIN_ENABLED=$VLLM_MEMBRAIN_ENABLED"
echo "VLLM_MEMBRAIN_ENDPOINT=$VLLM_MEMBRAIN_ENDPOINT"
echo "VLLM_MEMBRAIN_NAMESPACE=$VLLM_MEMBRAIN_NAMESPACE"