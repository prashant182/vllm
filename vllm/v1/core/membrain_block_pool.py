# SPDX-License-Identifier: Apache-2.0

from typing import Dict, List, Optional, Tuple, Any, Callable
import torch
import logging
import time
import asyncio

from vllm.logger import init_logger
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.request import Request

logger = init_logger(__name__)

class MembrainBlockPool(BlockPool):
    """Extension of BlockPool that integrates with tiered caching system.
    
    This class extends the standard BlockPool to handle tiered caching across
    GPU, CPU, and remote (Membrain) tiers. It intercepts the cache_full_blocks 
    call and handles tiered caching after standard GPU caching.
    
    Args:
        num_gpu_blocks: Number of blocks in the GPU cache
        enable_caching: Whether to enable prefix caching
        enable_kv_cache_events: Whether to enable KV cache events
        cpu_cache: Optional CPU cache tier
        membrain_store: Optional remote Membrain store
    """
    
    def __init__(
        self,
        num_gpu_blocks: int,
        enable_caching: bool,
        enable_kv_cache_events: bool = False,
        cpu_cache = None,
        membrain_store = None,
    ):
        super().__init__(num_gpu_blocks, enable_caching, enable_kv_cache_events)
        self.cpu_cache = cpu_cache
        self.membrain_store = membrain_store
        
        # Tracking for CPU and remote tiers
        self.cpu_blocks = set()
        self.remote_blocks = {}
        
        # Stats for monitoring
        self.cpu_store_attempts = 0
        self.cpu_store_successes = 0
        self.remote_store_attempts = 0
        self.remote_store_successes = 0
        
        # Create a single shared event loop for all async operations
        self._event_loop = None
        try:
            self._event_loop = asyncio.new_event_loop()
        except Exception as e:
            logger.error(f"Failed to create event loop for MembrainBlockPool: {e}")
            
    def _extract_hash_key(self, block_hash) -> str:
        """Extract a stable key from various hash formats.
        
        Args:
            block_hash: Hash in various possible formats
            
        Returns:
            str: Consistent string key
        """
        hash_key = str(block_hash)
        
        if isinstance(block_hash, tuple) and hasattr(block_hash, "hash_value"):
            hash_key = str(block_hash.hash_value)
        elif "hash_value=" in hash_key:
            import re
            match = re.search(r"hash_value=(-?\d+)", hash_key)
            if match:
                hash_key = match.group(1)
                
        return hash_key
            
    def _should_store_cpu(self, request: Request, block_index: int) -> bool:
        """Determine if block should be stored in CPU tier based on policy.
        
        Args:
            request: The request context
            block_index: Position of the block in the sequence
            
        Returns:
            bool: True if block should be stored in CPU tier
        """
        # Check if we should force CPU caching (for testing)
        import os
        force_cpu = os.environ.get('VLLM_FORCE_CPU_CACHE', '0').lower() in ('1', 'true', 'yes')
        if force_cpu:
            logger.debug(f"Forcing CPU cache storage for block {block_index} due to VLLM_FORCE_CPU_CACHE=1")
            return True
            
        # Default policy: store everything in CPU cache
        # This is generally safe since CPU memory is much larger than GPU
        return True
        
    def _should_store_remote(self, request: Request, block_index: int) -> bool:
        """Determine if block should be stored in remote tier based on policy.
        
        Args:
            request: The request context
            block_index: Position of the block in the sequence
            
        Returns:
            bool: True if block should be stored in remote tier
        """
        # Check if we should force remote caching (for testing)
        import os
        force_remote = os.environ.get('VLLM_FORCE_REMOTE_CACHE', '0').lower() in ('1', 'true', 'yes')
        if force_remote:
            logger.debug(f"Forcing remote cache storage for block {block_index} due to VLLM_FORCE_REMOTE_CACHE=1")
            return True
            
        # Simple policy to limit remote storage to reduce overhead
        # Store blocks with more priority if they're early in the sequence
        if block_index < 4:
            # Store all first few blocks (most valuable for prefix caching)
            return True
        elif block_index < 16:
            # Store every other block
            return block_index % 2 == 0
        elif block_index < 64:
            # Store every 4th block 
            return block_index % 4 == 0
        else:
            # Store every 8th block for long sequences
            return block_index % 8 == 0
    
    def _get_block_tensor(self, block: KVCacheBlock) -> Optional[torch.Tensor]:
        """Get tensor from a block with multiple fallback methods.
        
        Args:
            block: The block to get tensor from
            
        Returns:
            torch.Tensor or None: The tensor if available
        """
        # First try direct attribute access (most common)
        if hasattr(block, 'tensor') and block.tensor is not None:
            return block.tensor
            
        # Then try via block_pool helper
        block_with_tensor = self.get_tensor_for_block(block)
        if block_with_tensor is not None and hasattr(block_with_tensor, 'tensor'):
            return block_with_tensor.tensor
            
        return None
            
    def cache_full_blocks(
        self,
        request: Request,
        blocks: List[KVCacheBlock], 
        block_hashes: List[BlockHashType],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        hash_fn: Callable,
    ) -> None:
        """Cache full blocks in tiered system
        
        Extends the base implementation to also cache blocks in CPU and Membrain.
        First calls the parent to handle GPU caching, then handles CPU and remote.
        
        Args:
            request: The request these blocks belong to
            blocks: The blocks to potentially cache
            block_hashes: The block hashes
            num_cached_blocks: Number of blocks already cached
            num_full_blocks: Number of full blocks that should be cached
            block_size: Size of each block
            hash_fn: Hash function to use
        """
        print(f"🔍 TIERED CACHE: Blocks for req {request.request_id}, cached={num_cached_blocks}, full={num_full_blocks}")
        
        # First, call the parent implementation to handle GPU caching
        super().cache_full_blocks(
            request,
            blocks,
            block_hashes,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            hash_fn
        )
        
        # Important: We process ALL full blocks for tiered caching, not just the newly cached ones
        # This ensures blocks are properly propagated to all tiers regardless of GPU caching status
        end_idx = min(num_full_blocks, len(blocks), len(block_hashes))
        
        if end_idx <= 0:
            logger.debug(f"MembrainBlockPool: No blocks to cache in tiers")
            return
            
        print(f"🔍 TIERED CACHE: Propagating {end_idx} blocks to CPU and remote tiers")
        logger.debug(f"MembrainBlockPool: Propagating {end_idx} blocks to CPU and remote tiers")
            
        # Process ALL blocks that should be cached (0 to end_idx)
        for i in range(end_idx):
            block = blocks[i]
            block_hash = block_hashes[i]
            
            # Skip if already in tracking
            hash_key = self._extract_hash_key(block_hash)
            
            # Get block tensor using helpers if needed
            block_tensor = self._get_block_tensor(block)
            
            if block_tensor is None:
                logger.debug(f"MembrainBlockPool: No tensor found for block {block.block_id}, skipping caching")
                continue
                
            # First try to cache in CPU if available
            if self.cpu_cache is not None and self._should_store_cpu(request, i):
                # Skip if already stored in CPU tier
                if hash_key in self.cpu_blocks:
                    print(f"💾 CPU CACHE: Block {block.block_id} already in CPU cache, skipping")
                    continue
                    
                print(f"💾 CPU CACHE: Storing block {block.block_id}, hash={hash_key}")
                logger.debug(f"MembrainBlockPool: Attempting to store block {block.block_id} in CPU tier")
                self.cpu_store_attempts += 1
                cpu_tensor = block_tensor.cpu() if block_tensor.device.type != "cpu" else block_tensor
                
                metadata = {
                    "request_id": request.request_id,
                    "block_id": block.block_id,
                    "block_index": i,
                    "timestamp": time.time(),
                }
                
                try:
                    if self.cpu_cache.store(hash_key, cpu_tensor, metadata):
                        print(f"💾 CPU CACHE: Successfully stored block {block.block_id} in CPU tier")
                        self.cpu_store_successes += 1
                        self.cpu_blocks.add(hash_key)
                        logger.debug(f"MembrainBlockPool: Stored block {block.block_id} in CPU cache with key {hash_key}")
                    else:
                        print(f"💾 CPU CACHE: Failed to store block {block.block_id} in CPU tier")
                except Exception as e:
                    print(f"💾 CPU CACHE: Error storing block {block.block_id}: {str(e)}")
                    logger.error(f"MembrainBlockPool: Error storing block {block.block_id} in CPU cache: {e}")
            
            # Then try to cache in remote tier if enabled and it passes policy
            if self.membrain_store and self._should_store_remote(request, i):
                # Skip if already stored in remote tier
                if hash_key in self.remote_blocks:
                    print(f"🌐 REMOTE CACHE: Block {block.block_id} already in remote cache, skipping")
                    continue
                    
                print(f"🌐 REMOTE CACHE: Storing block {block.block_id}, hash={hash_key}")
                logger.debug(f"MembrainBlockPool: Attempting to store block {block.block_id} in remote tier")
                self.remote_store_attempts += 1
                
                metadata = {
                    "request_id": request.request_id,
                    "block_id": block.block_id,
                    "block_index": i,
                    "tensor_shape": str(block_tensor.shape),
                    "tensor_type": str(block_tensor.dtype),
                    "timestamp": time.time(),
                }
                
                try:
                    # Use shared event loop for async operations
                    if self._event_loop and not self._event_loop.is_closed():
                        success = self._event_loop.run_until_complete(self.membrain_store.store_block(
                            hash_key,
                            block_tensor,
                            metadata=metadata
                        ))
                        
                        if success:
                            print(f"🌐 REMOTE CACHE: Successfully stored block {block.block_id} in remote tier")
                            logger.debug(f"MembrainBlockPool: Successfully stored block {block.block_id} in remote tier")
                            self.remote_store_successes += 1
                            self.remote_blocks[hash_key] = block
                        else:
                            print(f"🌐 REMOTE CACHE: Failed to store block {block.block_id} in remote tier")
                    else:
                        print(f"🌐 REMOTE CACHE: Event loop unavailable for block {block.block_id}")
                        logger.error(f"MembrainBlockPool: Event loop not available for remote storage")
                except Exception as e:
                    print(f"🌐 REMOTE CACHE: Error storing block {block.block_id}: {str(e)}")
                    logger.error(f"MembrainBlockPool: Error storing block {block.block_id} in remote tier: {e}")
                    
        print(f"🔍 TIERED CACHE: Finished propagating {end_idx} blocks to tiered caching system")
        logger.debug(f"MembrainBlockPool: Finished propagating {end_idx} blocks to tiered caching system")
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get metrics for the block pool and cache tiers"""
        metrics = {
            "cpu_tier": {
                "store_attempts": self.cpu_store_attempts,
                "store_successes": self.cpu_store_successes,
                "store_success_rate": self.cpu_store_successes / max(1, self.cpu_store_attempts),
                "tracked_blocks": len(self.cpu_blocks),
            },
            "remote_tier": {
                "store_attempts": self.remote_store_attempts,
                "store_successes": self.remote_store_successes,
                "store_success_rate": self.remote_store_successes / max(1, self.remote_store_attempts),
                "tracked_blocks": len(self.remote_blocks),
            }
        }
        
        # Add CPU cache metrics if available
        if self.cpu_cache and hasattr(self.cpu_cache, "get_metrics"):
            metrics["cpu_tier"].update(self.cpu_cache.get_metrics())
            
        # Add remote metrics if available
        if self.membrain_store and hasattr(self.membrain_store, "get_metrics"):
            remote_metrics = self.membrain_store.get_metrics()
            if remote_metrics:
                metrics["remote_tier"].update(remote_metrics)
                
        return metrics
        
    def __del__(self):
        """Clean up resources when the object is deleted"""
        if hasattr(self, '_event_loop') and self._event_loop is not None:
            if not self._event_loop.is_closed():
                self._event_loop.close()