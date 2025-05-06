#!/bin/bash
# Start a vLLM OpenAI compatible server with Membrain connector

# Configuration
CUDA_DEVICE=${1:-0}  # Default to GPU 0
PORT=${2:-8000}      # Default to port 8000
MODE=${3:-"both"} # Default to producer mode (can be "producer", "consumer", or "both")

# Set environment variables
export CUDA_VISIBLE_DEVICES=$CUDA_DEVICE
export VLLM_USE_V1=1
export VLLM_WORKER_MULTIPROC_METHOD=spawn

# Create KV config based on mode
if [ "$MODE" = "producer" ]; then
  ROLE="kv_producer"
elif [ "$MODE" = "consumer" ]; then
  ROLE="kv_consumer"
elif [ "$MODE" = "both" ]; then
  ROLE="kv_both"
fi

KV_CONFIG="{\"kv_connector\":\"MembrainConnectorV1\", \"kv_role\":\"$ROLE\", \"kv_connector_extra_config\": {\"membrain_endpoint\":\"http://localhost:9201\", \"membrain_namespace\":\"vllm_kv\", \"chunk_size\":256}}"

# Run the server
echo "Starting vLLM server in $MODE mode on port $PORT using GPU $CUDA_DEVICE..."

python -m vllm.entrypoints.openai.api_server \
  --model facebook/opt-125m \
  --host 0.0.0.0 \
  --port $PORT \
  --kv-transfer-config "$KV_CONFIG" \
  --enable-prefix-caching \
  --max-model-len 2048