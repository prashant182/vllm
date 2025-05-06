#!/bin/bash
set -e

# This script should be run on all nodes in the cluster

# Create directory on the node
mkdir -p /mnt/nvme_raid/vllm-data

# Set permissions to allow all pods to read/write
chmod 777 /mnt/nvme_raid/vllm-data

# Ensure the directory is accessible across all nodes
echo "Created /mnt/nvme_raid/vllm-data with permissions 777"

# Note: You'll need to run this script on each node in your cluster,
# or use a DaemonSet to automate this process.

# After running on all nodes, apply the PV, PVC, and StorageClass resources
echo "Now apply the Kubernetes resources with:"
echo "kubectl apply -f local-pv-pvc.yaml"
echo "kubectl apply -f namespace.yaml"
echo "kubectl apply -f huggingface-credentials-secret.yaml"