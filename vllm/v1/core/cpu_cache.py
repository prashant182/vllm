#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

import torch
import time
from typing import Dict, Optional, Tuple, Any
from collections import OrderedDict
from vllm.logger import init_logger

logger = init_logger(__name__)

class CPUCacheTier:
    """CPU memory tier for caching blocks evicted from GPU but not yet sent to remote storage.
    
    This provides a fast intermediate cache between GPU memory and remote storage
    using a simple LRU (Least Recently Used) eviction policy.
    """
    
    def __init__(self, max_size_bytes: int = 4 * 1024 * 1024 * 1024):  # Default 4GB
        """Initialize CPU cache tier.
        
        Args:
            max_size_bytes: Maximum memory to use for CPU cache in bytes
        """
        self.max_size_bytes = max_size_bytes
        self.current_size_bytes = 0
        # OrderedDict to maintain LRU order
        self.cache: OrderedDict[str, Tuple[torch.Tensor, int, Dict]] = OrderedDict()
        # Track size of each tensor for easy eviction decisions
        self.tensor_sizes: Dict[str, int] = {}
        # Metrics
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        
    def store(self, block_hash: str, tensor: torch.Tensor, metadata: Optional[Dict] = None) -> bool:
        """Store tensor in CPU cache.
        
        Args:
            block_hash: Hash key for the block
            tensor: Tensor data to store
            metadata: Optional metadata to store with tensor
            
        Returns:
            bool: True if stored successfully, False otherwise
        """
        print(f"🔵 CPU CACHE: Attempting to store block {block_hash}, shape={tensor.shape}")
        
        if block_hash in self.cache:
            # Update existing entry - move to end of LRU
            current_tensor, ref_count, _ = self.cache.pop(block_hash)
            self.cache[block_hash] = (tensor, ref_count, metadata or {})
            # No need to update size as we're just replacing
            print(f"🔵 CPU CACHE: Updated existing block {block_hash}, ref_count={ref_count}")
            return True
            
        # Calculate tensor size in bytes
        tensor_size = tensor.element_size() * tensor.nelement()
        
        # Check if we need to make space
        if self.current_size_bytes + tensor_size > self.max_size_bytes:
            print(f"🔵 CPU CACHE: Need to evict for block {block_hash}, need {tensor_size/1024/1024:.2f}MB")
            self._evict_until_size(tensor_size)
            # If still not enough space, return False
            if self.current_size_bytes + tensor_size > self.max_size_bytes:
                logger.debug(f"CPU cache: Cannot store block {block_hash}, not enough space after eviction")
                print(f"🔵 CPU CACHE: FAILED to store block {block_hash}, not enough space after eviction")
                return False
        
        # Store tensor with initial ref_count=1
        self.cache[block_hash] = (tensor, 1, metadata or {})
        self.tensor_sizes[block_hash] = tensor_size
        self.current_size_bytes += tensor_size
        
        logger.debug(f"CPU cache: Stored block {block_hash}, size={tensor_size/1024/1024:.2f}MB, "
                   f"total={self.current_size_bytes/1024/1024:.2f}MB")
        print(f"🔵 CPU CACHE: Successfully stored block {block_hash}, size={tensor_size/1024/1024:.2f}MB, "
              f"total={self.current_size_bytes/1024/1024:.2f}MB")
        return True
        
    def load(self, block_hash: str) -> Optional[Tuple[torch.Tensor, Dict]]:
        """Load tensor from CPU cache.
        
        Args:
            block_hash: Hash key for the block
            
        Returns:
            tuple: (tensor, metadata) if found, None otherwise
        """
        print(f"🔵 CPU CACHE: Attempting to load block {block_hash}")
        
        if block_hash not in self.cache:
            self.misses += 1
            print(f"🔵 CPU CACHE: MISS for block {block_hash}")
            return None
            
        # Move to end of LRU (most recently used)
        tensor, ref_count, metadata = self.cache.pop(block_hash)
        self.cache[block_hash] = (tensor, ref_count, metadata)
        self.hits += 1
        
        logger.debug(f"CPU cache: Loaded block {block_hash}, ref_count={ref_count}")
        print(f"🔵 CPU CACHE: HIT for block {block_hash}, ref_count={ref_count}, shape={tensor.shape}")
        return tensor, metadata
        
    def increment_ref(self, block_hash: str) -> int:
        """Increment reference count for block.
        
        Args:
            block_hash: Hash key for the block
            
        Returns:
            int: New reference count or 0 if block not found
        """
        if block_hash not in self.cache:
            return 0
            
        tensor, ref_count, metadata = self.cache[block_hash]
        # Increase reference count
        ref_count += 1
        self.cache[block_hash] = (tensor, ref_count, metadata)
        
        logger.debug(f"CPU cache: Incremented ref for block {block_hash} to {ref_count}")
        return ref_count
        
    def decrement_ref(self, block_hash: str) -> int:
        """Decrement reference count for block.
        
        Args:
            block_hash: Hash key for the block
            
        Returns:
            int: New reference count or 0 if block was removed or not found
        """
        if block_hash not in self.cache:
            return 0
            
        tensor, ref_count, metadata = self.cache[block_hash]
        ref_count -= 1
        
        if ref_count <= 0:
            # Remove if no longer referenced
            tensor_size = self.tensor_sizes.pop(block_hash)
            self.current_size_bytes -= tensor_size
            self.cache.pop(block_hash)
            logger.debug(f"CPU cache: Removed unreferenced block {block_hash}")
            return 0
        else:
            # Update reference count
            self.cache[block_hash] = (tensor, ref_count, metadata)
            logger.debug(f"CPU cache: Decremented ref for block {block_hash} to {ref_count}")
            return ref_count
    
    def _evict_until_size(self, required_size: int) -> None:
        """Evict least recently used blocks until there's enough space.
        
        Args:
            required_size: Size in bytes needed
        """
        # Only evict blocks with ref_count=0
        blocks_to_evict = []
        for block_hash, (_, ref_count, _) in self.cache.items():
            if ref_count == 0:
                blocks_to_evict.append(block_hash)
                
        # Check unreferenced blocks first (those are safe to evict)
        for block_hash in blocks_to_evict:
            if self.current_size_bytes + required_size <= self.max_size_bytes:
                break
                
            _, _, _ = self.cache.pop(block_hash)
            tensor_size = self.tensor_sizes.pop(block_hash)
            self.current_size_bytes -= tensor_size
            self.evictions += 1
            
            logger.debug(f"CPU cache: Evicted block {block_hash}, size={tensor_size/1024/1024:.2f}MB")
            
    def has_block(self, block_hash: str) -> bool:
        """Check if block exists in cache.
        
        Args:
            block_hash: Hash key for the block
            
        Returns:
            bool: True if block exists, False otherwise
        """
        return block_hash in self.cache
        
    def get_metrics(self) -> Dict[str, Any]:
        """Get cache metrics.
        
        Returns:
            dict: Metrics about cache usage
        """
        hit_rate = self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0
        return {
            "size_bytes": self.current_size_bytes,
            "size_mb": self.current_size_bytes / 1024 / 1024,
            "max_size_mb": self.max_size_bytes / 1024 / 1024,
            "utilization": self.current_size_bytes / self.max_size_bytes,
            "entries": len(self.cache),
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": hit_rate,
            "evictions": self.evictions
        }