#!/bin/bash
# Script to run any vLLM command with Membrain distributed caching enabled

# Enable vLLM V1 and setup Membrain
export VLLM_USE_V1=1
export VLLM_MEMBRAIN_ENABLED=1
export VLLM_MEMBRAIN_ENDPOINT=${VLLM_MEMBRAIN_ENDPOINT:-"http://localhost:9201"}
export VLLM_MEMBRAIN_NAMESPACE=${VLLM_MEMBRAIN_NAMESPACE:-"default"}

# Activate virtual environment if it exists
if [ -d "myenv" ]; then
    source myenv/bin/activate
fi

echo "Running with Membrain distributed caching enabled:"
echo "VLLM_USE_V1=$VLLM_USE_V1"
echo "VLLM_MEMBRAIN_ENABLED=$VLLM_MEMBRAIN_ENABLED" 
echo "VLLM_MEMBRAIN_ENDPOINT=$VLLM_MEMBRAIN_ENDPOINT"
echo "VLLM_MEMBRAIN_NAMESPACE=$VLLM_MEMBRAIN_NAMESPACE"
echo ""
echo "Running command: $@"
echo "-------------------------------------"

# Run the command passed as arguments
"$@"