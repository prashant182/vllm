# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple, Any
import os
import uuid
import torch
import time
import logging
from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.membrain import MembrainConfig, MembrainStore
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig 
from vllm.v1.request import Request, RequestStatus
from vllm.v1.core.cpu_cache import CPUCacheTier
from vllm.v1.core.membrain_block_pool import MembrainBlockPool

logger = init_logger(__name__)


@dataclass
class MembrainKVConfig:
    """Configuration for Membrain-enabled KV Cache Manager"""
    membrain: MembrainConfig
    node_id: str = ""  # Will be auto-generated if empty
    enable_metrics: bool = False
    cpu_cache_size_bytes: int = 4 * 1024 * 1024 * 1024  # 4GB default
    cpu_cache_enabled: bool = True  # Enable CPU cache by default


class MembrainKVCacheManager(KVCacheManager):
    """KVCache Manager with Membrain distributed memory support
    
    This extends the standard KVCacheManager to add a distributed memory tier
    using Membrain. It maintains the same interface while transparently 
    handling distributed caching.
    """

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        membrain_config: Optional[MembrainKVConfig] = None,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
    ):
        """Initialize KV Cache Manager with Membrain support
        
        Args:
            kv_cache_config: KV cache configuration
            max_model_len: Maximum model sequence length
            membrain_config: Optional Membrain configuration
            enable_caching: Whether to enable prefix caching
            caching_hash_algo: Hash algorithm for prefix caching
            use_eagle: Whether to use eagle drafting head
            log_stats: Whether to log statistics
            enable_kv_cache_events: Whether to enable KV cache events
        """
        # Initialize CPU cache tier first
        self.cpu_cache = None
        if membrain_config and getattr(membrain_config, 'cpu_cache_enabled', False):
            cpu_cache_size = getattr(membrain_config, 'cpu_cache_size_bytes', 4 * 1024 * 1024 * 1024)
            self.cpu_cache = CPUCacheTier(cpu_cache_size)
            logger.info(f"Initialized CPU cache tier with {cpu_cache_size / 1024 / 1024 / 1024:.1f}GB")
        else:
            # Check environment variable if no config
            env_cpu_cache = os.environ.get('VLLM_MEMBRAIN_CPU_CACHE_SIZE_GB', '0')
            try:
                cpu_cache_size_gb = float(env_cpu_cache)
                if cpu_cache_size_gb > 0:
                    cpu_cache_bytes = int(cpu_cache_size_gb * 1024 * 1024 * 1024)
                    self.cpu_cache = CPUCacheTier(cpu_cache_bytes)
                    logger.info(f"Initialized CPU cache tier with {cpu_cache_size_gb}GB from env var")
            except ValueError:
                pass

        # Initialize Membrain if configured
        self.membrain = None
        self.membrain_config = membrain_config
        if membrain_config:
            # Auto-generate node ID if not provided
            if not membrain_config.node_id:
                membrain_config.node_id = str(uuid.uuid4())

            # Get dtype from the KV cache spec since it's not available directly in KVCacheConfig
            kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
            
            self.membrain = MembrainStore(
                config=membrain_config.membrain,
                node_id=membrain_config.node_id,
                block_size=self.block_size if hasattr(self, 'block_size') else kv_cache_spec.block_size,
                dtype=kv_cache_spec.dtype if hasattr(kv_cache_spec, 'dtype') else torch.float16
            )
            
            logger.info(f"Initialized Membrain KV cache manager with endpoint {membrain_config.membrain.endpoint}")
            logger.info(f"Using namespace: {membrain_config.membrain.namespace}")
            logger.info(f"Node ID: {membrain_config.node_id}")

        # Save block size before parent init
        self.block_size = kv_cache_spec.block_size if hasattr(kv_cache_spec, 'block_size') else None
        
        # Initialize parent after CPU and Membrain setup
        # Don't create the standard block pool - we'll replace it with our custom one
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            caching_hash_algo=caching_hash_algo,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
        )
        
        # Replace the standard BlockPool with our MembrainBlockPool
        # Save the num_gpu_blocks before replacing
        num_gpu_blocks = self.block_pool.num_gpu_blocks
        
        # Create the MembrainBlockPool with access to CPU and remote tiers
        self.block_pool = MembrainBlockPool(
            num_gpu_blocks=num_gpu_blocks,
            enable_caching=enable_caching,
            enable_kv_cache_events=enable_kv_cache_events,
            cpu_cache=self.cpu_cache,
            membrain_store=self.membrain
        )
        
        logger.info("Replaced standard BlockPool with MembrainBlockPool")
        
        # Create a single shared event loop for all async operations
        import asyncio
        self._event_loop = asyncio.new_event_loop()
        
        # Stats for monitoring - now handled by MembrainBlockPool
        self.store_attempts = 0
        self.store_successes = 0
        self.load_attempts = 0 
        self.load_successes = 0
        self.cpu_store_attempts = 0
        self.cpu_store_successes = 0
        self.cpu_load_attempts = 0
        self.cpu_load_successes = 0
        
        # Track blocks in different tiers - now handled by MembrainBlockPool
        self.remote_blocks: Dict[str, KVCacheBlock] = {}
        self.cpu_blocks: Set[str] = set()

    def get_computed_blocks(
        self,
        request: Request
    ) -> tuple[list[KVCacheBlock], int]:
        """Get computed blocks for a request from all tiers (local, CPU, remote)
        
        This extends the base implementation to check all cache tiers:
        1. Check local GPU cache first (via parent implementation)
        2. If not found locally, check CPU cache
        3. If not found in CPU, check Membrain remote tier
        
        Args:
            request: The request to get blocks for
            
        Returns:
            Tuple of (computed blocks, number of computed tokens)
        """
        # First try local cache
        print(f"🔄 CACHE FLOW: Looking up blocks for request {request.request_id}")
        
        local_blocks, num_local_tokens = super().get_computed_blocks(request)
        
        print(f"🔄 CACHE FLOW: GPU lookup found {len(local_blocks)} blocks")
        
        if local_blocks:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Found {len(local_blocks)} blocks in local cache")
            return local_blocks, num_local_tokens
        
        # If CPU cache is available, check it next
        if self.cpu_cache is not None:
            block_hashes = self.req_to_block_hashes.get(request.request_id, [])
            
            if not block_hashes:
                print(f"🔄 CACHE FLOW: No block hashes found for request {request.request_id}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"No block hashes found for request {request.request_id}")
                return [], 0
            
            print(f"🔄 CACHE FLOW: Checking CPU cache for {len(block_hashes)} blocks")
                
            # Try to load from CPU cache
            cpu_blocks = []
            
            for i, block_hash in enumerate(block_hashes):
                # Skip if we already have this block locally
                local_block = self.block_pool.get_cached_block(block_hash)
                if local_block:
                    print(f"🔄 CACHE FLOW: Block {i} already in GPU cache, skipping CPU lookup")
                    continue
                    
                # Extract a stable key from the hash
                hash_key = self._extract_hash_key(block_hash)
                
                # Count attempts for metrics
                self.cpu_load_attempts += 1
                
                # Check if in CPU cache
                if self.cpu_cache.has_block(hash_key):
                    result = self.cpu_cache.load(hash_key)
                    if result:
                        tensor, metadata = result
                        
                        # Allocate new block and populate with CPU tensor data
                        block = self._allocate_new_block()
                        
                        # Move tensor to GPU if needed
                        if tensor.device.type == "cpu" and torch.cuda.is_available():
                            tensor = tensor.cuda()
                            
                        # Set block tensor and hash
                        block.tensor = tensor
                        block.block_hash = block_hash
                        
                        # Mark as full since it's a complete block
                        if hasattr(block, 'mark_full'):
                            block.mark_full()
                            
                        cpu_blocks.append(block)
                        self.cpu_load_successes += 1
                        
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Loaded block {i} with key {hash_key} from CPU cache")
                    else:
                        # Break the chain - need all blocks in sequence
                        print(f"🔄 CACHE FLOW: Block {i} not found in CPU cache, breaking chain")
                        break
                else:
                    # Not in CPU cache, break the chain
                    print(f"🔄 CACHE FLOW: Block {i} not in CPU cache, breaking chain")
                    break
                    
            # If we found blocks in CPU cache, return them
            if cpu_blocks:
                print(f"🔄 CACHE FLOW: CPU lookup found {len(cpu_blocks)} blocks")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Found {len(cpu_blocks)} blocks in CPU cache")
                return cpu_blocks, len(cpu_blocks) * self.block_size
        
        # If we get here, check remote tier (Membrain) as last resort
        if not self.membrain:
            print(f"🔄 CACHE FLOW: Membrain not configured, skipping remote lookup")
            return [], 0
            
        block_hashes = self.req_to_block_hashes.get(request.request_id, [])
        
        if not block_hashes:
            print(f"🔄 CACHE FLOW: No block hashes for remote lookup")
            return [], 0
            
        print(f"🔄 CACHE FLOW: Checking remote tier (Membrain) for {len(block_hashes)} blocks")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Checking Membrain remote tier for {len(block_hashes)} blocks")
            
        remote_blocks = []
        
        for i, block_hash in enumerate(block_hashes):
            # Skip if we already have this block locally
            local_block = self.block_pool.get_cached_block(block_hash)
            if local_block:
                print(f"🔄 CACHE FLOW: Block {i} already in GPU cache, skipping remote lookup")
                continue
                
            # Extract a stable key from the hash
            hash_key = self._extract_hash_key(block_hash)
                
            # Count attempts for metrics
            self.load_attempts += 1

            # Try to load from remote using shared event loop
            try:
                print(f"🔄 CACHE FLOW: Loading block {i} with key {hash_key} from remote")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Loading block {i} with key {hash_key} from Membrain")
                    
                tensor = self._event_loop.run_until_complete(self.membrain.load_block(hash_key))
                
                if tensor is not None:
                    print(f"🔄 CACHE FLOW: Successfully loaded block {i} from remote")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Successfully loaded block {i} from Membrain")
                        
                    self.load_successes += 1
                    
                    # Also store in CPU cache if available (without using much overhead)
                    if self.cpu_cache is not None:
                        # Store CPU copy for faster access later
                        self.cpu_store_attempts += 1
                        print(f"🔄 CACHE FLOW: Storing remote block {i} in CPU cache")
                        cpu_tensor = tensor.cpu() if tensor.device.type != "cpu" else tensor
                        if self.cpu_cache.store(hash_key, cpu_tensor):
                            self.cpu_store_successes += 1
                            self.cpu_blocks.add(hash_key)
                else:
                    # Break the chain - we need all blocks in sequence for prefix caching
                    print(f"🔄 CACHE FLOW: Block {i} not found in remote, breaking chain")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Block {i} not found in Membrain, breaking chain")
                    break
            except Exception as e:
                print(f"🔄 CACHE FLOW: ERROR loading block {i} from remote: {e}")
                logger.error(f"Failed to load block {i} with key {hash_key} from Membrain: {e}")
                break

            # Allocate new block and populate it with the loaded tensor
            block = self._allocate_new_block()
            
            # Set block tensor and hash
            block.tensor = tensor
            block.block_hash = block_hash
            
            # Mark as full since we're loading a completed block
            if hasattr(block, 'mark_full'):
                block.mark_full()
                
            remote_blocks.append(block)

            # Track block in our remote blocks dictionary
            self.remote_blocks[block_hash] = block

        print(f"🔄 CACHE FLOW: Remote lookup found {len(remote_blocks)} blocks")
        if remote_blocks and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Loaded {len(remote_blocks)} blocks from Membrain")
            
        return remote_blocks, len(remote_blocks) * self.block_size

    def _should_store_cpu(self, request: Request, block: KVCacheBlock, block_index: int) -> bool:
        """Determine if block should be stored in CPU tier based on policy.
        
        This implements a policy for CPU caching, with the option to force caching for testing.
        
        Args:
            request: The request context
            block: The block being considered
            block_index: Position of the block in the sequence
            
        Returns:
            bool: True if block should be stored in CPU tier
        """
        # Check if we should force CPU caching (for testing)
        force_cpu = os.environ.get('VLLM_FORCE_CPU_CACHE', '0').lower() in ('1', 'true', 'yes')
        if force_cpu:
            print(f"💾 POLICY: Forcing CPU cache storage for block {block_index} due to VLLM_FORCE_CPU_CACHE=1")
            return True
            
        # Default policy: store everything in CPU cache
        # This is generally safe since CPU memory is much larger than GPU
        return True

    def cache_full_blocks(
        self,
        request: Request,
        blocks: list[KVCacheBlock], 
        block_hashes: list[BlockHashType],
        num_cached_blocks: int,
        num_full_blocks: int,
        block_size: int,
        hash_fn: callable,
    ) -> None:
        """Cache full blocks in tiered system
        
        Extends base implementation to also cache blocks in CPU and Membrain.
        The workflow is:
        1. First use the block pool implementation to handle local caching
        2. Then selectively cache to CPU tier
        3. Finally, selectively cache to remote tier
        
        Args:
            request: The request these blocks belong to
            blocks: The blocks to potentially cache
            block_hashes: The block hashes
            num_cached_blocks: Number of blocks already cached
            num_full_blocks: Number of full blocks that should be cached
            block_size: Size of each block
            hash_fn: Hash function to use
        """
        print(f"💾 BLOCK STORE: Request to cache blocks for {request.request_id}, cached={num_cached_blocks}, full={num_full_blocks}")
        
        # Calculate how many blocks we can safely cache based on token count
        safe_num_full_blocks = num_cached_blocks
        for i in range(num_cached_blocks, min(num_full_blocks, len(blocks))):
            # Calculate the start token index for this block
            block_idx = i
            start_token_idx = block_idx * block_size
            end_token_idx = (block_idx + 1) * block_size
            
            # Check if we have enough tokens to fill this block
            if end_token_idx <= len(request.all_token_ids):
                # This block can be safely cached
                safe_num_full_blocks = i + 1
            else:
                # This block doesn't have enough tokens
                print(f"💾 BLOCK STORE: Block {i} has incomplete tokens ({len(request.all_token_ids) - start_token_idx} < {block_size}), stopping")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Block {i} has incomplete tokens ({len(request.all_token_ids) - start_token_idx} < {block_size}), stopping")
                break
                
        if safe_num_full_blocks != num_full_blocks:
            print(f"💾 BLOCK STORE: Adjusted num_full_blocks: {num_full_blocks} → {safe_num_full_blocks}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"Adjusted num_full_blocks: {num_full_blocks} → {safe_num_full_blocks}")
            
        num_full_blocks = safe_num_full_blocks
        
        if num_cached_blocks >= num_full_blocks:
            print(f"💾 BLOCK STORE: No new blocks to cache for request {request.request_id}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"No new blocks to cache for request {request.request_id}")
            return
            
        # Call the block pool implementation to handle local caching
        print(f"💾 BLOCK STORE: Caching {num_full_blocks - num_cached_blocks} new blocks in GPU tier")
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Calling block_pool.cache_full_blocks for {num_full_blocks - num_cached_blocks} blocks")
            
        self.block_pool.cache_full_blocks(
            request,
            blocks,
            block_hashes,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            hash_fn
        )
        
        # Now handle CPU and remote caching for the newly cached blocks
        # Make sure we don't go out of bounds
        start_idx = num_cached_blocks
        end_idx = min(num_full_blocks, len(blocks), len(block_hashes))
        
        if start_idx >= end_idx:
            print(f"💾 BLOCK STORE: No blocks to cache in tiers after index check")
            return
            
        print(f"💾 BLOCK STORE: Beginning tier caching for blocks {start_idx}-{end_idx-1}")
            
        for i in range(start_idx, end_idx):
            block = blocks[i]
            block_hash = block_hashes[i]
            
            # Skip if already in tracking
            hash_key = self._extract_hash_key(block_hash)
            
            # Get block tensor using helpers if needed
            block_tensor = self._get_block_tensor(block)
            
            if block_tensor is None:
                print(f"💾 BLOCK STORE: No tensor found for block {block.block_id}, skipping")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"No tensor found for block {block.block_id}, skipping caching")
                continue
                
            # First try to cache in CPU if available and it passes policy
            if self.cpu_cache is not None and self._should_store_cpu(request, block, i):
                print(f"💾 BLOCK STORE: Attempting to store block {block.block_id} in CPU tier")
                self.cpu_store_attempts += 1
                cpu_tensor = block_tensor.cpu() if block_tensor.device.type != "cpu" else block_tensor
                
                metadata = {
                    "request_id": request.request_id,
                    "node_id": self.membrain_config.node_id if self.membrain_config else "local",
                    "block_id": block.block_id,
                    "block_index": i,
                    "timestamp": time.time(),
                }
                
                if self.cpu_cache.store(hash_key, cpu_tensor, metadata):
                    self.cpu_store_successes += 1
                    self.cpu_blocks.add(hash_key)
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Stored block {block.block_id} in CPU cache with key {hash_key}")
            
            # Then try to cache in remote tier if enabled and it passes policy
            if self.membrain and hash_key not in self.remote_blocks and self._should_store_remote(request, block, i):
                print(f"💾 BLOCK STORE: Attempting to store block {block.block_id} in remote tier")
                self.store_attempts += 1
                
                metadata = {
                    "request_id": request.request_id,
                    "node_id": self.membrain_config.node_id if self.membrain_config else "local",
                    "block_id": block.block_id,
                    "block_index": i,
                    "tensor_shape": str(block_tensor.shape),
                    "tensor_type": str(block_tensor.dtype),
                    "timestamp": time.time(),
                }
                
                try:
                    # Use shared event loop for async operations
                    print(f"💾 BLOCK STORE: Calling Membrain to store block {block.block_id} with key {hash_key}")
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Calling Membrain to store block {block.block_id} with key {hash_key}")
                        
                    success = self._event_loop.run_until_complete(self.membrain.store_block(
                        hash_key,
                        block_tensor,
                        metadata=metadata
                    ))
                    
                    if success:
                        print(f"💾 BLOCK STORE: Successfully stored block {block.block_id} in remote tier")
                        self.store_successes += 1
                        self.remote_blocks[block_hash] = block
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Successfully stored block {block.block_id} in Membrain")
                    else:
                        print(f"💾 BLOCK STORE: Failed to store block {block.block_id} in remote tier")
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Failed to store block {block.block_id} in Membrain")
                        
                except Exception as e:
                    print(f"💾 BLOCK STORE: Error storing block {block.block_id} in remote tier: {e}")
                    logger.error(f"Error storing block {block.block_id}: {e}")
                    
        print(f"💾 BLOCK STORE: Cached {end_idx - start_idx} blocks for request {request.request_id}")
        if logger.isEnabledFor(logging.DEBUG):
            num_to_cache = end_idx - start_idx
            logger.debug(f"Cached {num_to_cache} blocks for request {request.request_id}")

    def free(self, request: Request) -> None:
        """Free blocks for a request from all tiers
        
        This extends the base implementation to also handle CPU and remote blocks.
        When a request is completed or aborted, this method:
        1. Gets the list of blocks assigned to the request before freeing them
        2. Calls the parent implementation to free blocks locally
        3. Decrements reference counts for blocks in the CPU tier
        4. Decrements reference counts for blocks in the remote tier
        
        Args:
            request: The request whose blocks should be freed
        """
        # Get blocks before freeing locally so we have access to them for remote cleanup
        blocks = self.req_to_blocks.get(request.request_id, [])
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Freeing {len(blocks)} blocks for request {request.request_id}")

        # Free locally first through parent implementation
        super().free(request)

        # Get the list of block hashes that were computed for this request
        computed_hashes = self.req_to_block_hashes.get(request.request_id, [])
        
        # Handle CPU cache tier
        if self.cpu_cache is not None and computed_hashes:
            for block_hash in computed_hashes:
                hash_key = self._extract_hash_key(block_hash)
                if hash_key in self.cpu_blocks:
                    # Decrement reference count in CPU cache
                    ref_count = self.cpu_cache.decrement_ref(hash_key)
                    if ref_count <= 0:
                        self.cpu_blocks.discard(hash_key)
                        
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"Decremented CPU cache ref for {hash_key}, new count: {ref_count}")

        # Skip remote handling if Membrain is not enabled
        if not self.membrain:
            return

        # Track blocks successfully freed in remote tier
        freed_blocks = 0
        
        # Iterate through all blocks that were allocated to this request
        for block in blocks:
            # Skip blocks with no hash (they weren't cached)
            if not block.block_hash:
                continue

            # Check if this block is in our remote tracking
            if block.block_hash in self.remote_blocks:
                # Extract a clean hash key for logs and API calls
                hash_key = self._extract_hash_key(block.block_hash)
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"Decrementing reference for block {block.block_id} with hash {hash_key}")
                
                try:
                    # Use shared event loop for async operations to decrement reference count
                    ref_count = self._event_loop.run_until_complete(self.membrain.decrement_ref(hash_key))
                    freed_blocks += 1
                    
                    # Log result based on whether block was deleted or just decremented
                    if ref_count <= 0:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Deleted block {block.block_id} from remote tier (ref count = 0)")
                    else:
                        if logger.isEnabledFor(logging.DEBUG):
                            logger.debug(f"Decremented block {block.block_id} in remote tier (ref count = {ref_count})")
                        
                except Exception as e:
                    logger.error(f"Failed to decrement ref for block {block.block_id}: {e}")
                    
                # Remove from our tracking regardless of success
                # This ensures we don't leave stale references in memory
                del self.remote_blocks[block.block_hash]
        
        if freed_blocks > 0 and logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"Released {freed_blocks} remote blocks for request {request.request_id}")

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

    def _should_store_remote(self, request: Request, block: KVCacheBlock, block_index: int) -> bool:
        """Determine if block should be stored in remote tier based on policy.
        
        This is a simple policy function that can be expanded later.
        For now, it implements:
        1. Only store blocks at regular intervals to reduce overhead
        2. Store blocks with lower index more aggressively (more valuable prefixes)
        
        Args:
            request: The request context
            block: The block being considered
            block_index: Position of the block in the sequence
            
        Returns:
            bool: True if block should be stored in remote tier
        """
        # Check if we should force remote caching (for testing)
        force_remote = os.environ.get('VLLM_FORCE_REMOTE_CACHE', '0').lower() in ('1', 'true', 'yes')
        if force_remote:
            print(f"💾 POLICY: Forcing remote cache storage for block {block_index} due to VLLM_FORCE_REMOTE_CACHE=1")
            logger.warning(f"Forcing remote cache storage for block {block_index} due to VLLM_FORCE_REMOTE_CACHE=1")
            return True
            
        # Simple policy to limit remote storage to reduce overhead
        # Store blocks with more priority if they're early in the sequence
        if block_index < 4:
            # Store all first few blocks (most valuable for prefix caching)
            return True
        elif block_index < 16:
            # Store every other block
            result = block_index % 2 == 0
            print(f"💾 POLICY: Block {block_index} evaluated to {result} for remote storage (block_index < 16)")
            return result
        elif block_index < 64:
            # Store every 4th block 
            result = block_index % 4 == 0
            print(f"💾 POLICY: Block {block_index} evaluated to {result} for remote storage (block_index < 64)")
            return result
        else:
            # Store every 8th block for long sequences
            result = block_index % 8 == 0
            print(f"💾 POLICY: Block {block_index} evaluated to {result} for remote storage (block_index >= 64)")
            return result
            
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
        block_with_tensor = self.block_pool.get_tensor_for_block(block)
        if block_with_tensor is not None and hasattr(block_with_tensor, 'tensor'):
            return block_with_tensor.tensor
            
        # We need to add more sophisticated fallbacks here if needed
        return None
        
    def _allocate_new_block(self) -> KVCacheBlock:
        """Helper to allocate a new block"""
        return self.block_pool.get_new_blocks(1)[0]
        
    def get_metrics(self) -> Dict:
        """Get metrics for all cache tiers"""
        metrics = {}
        
        # Get base metrics from parent
        base_metrics = super().make_prefix_cache_stats()
        if base_metrics:
            metrics["gpu_tier"] = base_metrics
        
        # Get metrics from MembrainBlockPool if available
        if hasattr(self.block_pool, 'get_metrics'):
            tier_metrics = self.block_pool.get_metrics()
            # Merge with our metrics
            if "cpu_tier" in tier_metrics:
                if "cpu_tier" not in metrics:
                    metrics["cpu_tier"] = {}
                metrics["cpu_tier"].update(tier_metrics["cpu_tier"])
            
            if "remote_tier" in tier_metrics:
                if "remote_tier" not in metrics:
                    metrics["remote_tier"] = {}
                metrics["remote_tier"].update(tier_metrics["remote_tier"])
        else:
            # Use legacy metrics if block_pool doesn't have get_metrics
            
            # Add CPU tier metrics
            if self.cpu_cache:
                metrics["cpu_tier"] = self.cpu_cache.get_metrics()
            
            # Add local tracking metrics for remote tier
            remote_metrics = {
                "store_attempts": self.store_attempts,
                "store_successes": self.store_successes,
                "load_attempts": self.load_attempts,
                "load_successes": self.load_successes,
                "store_success_rate": self.store_successes / self.store_attempts if self.store_attempts > 0 else 0.0,
                "load_success_rate": self.load_successes / self.load_attempts if self.load_attempts > 0 else 0.0,
                "tracked_remote_blocks": len(self.remote_blocks)
            }
            
            # Add remote metrics if available
            if self.membrain and self.membrain_config and getattr(self.membrain_config, 'enable_metrics', False):
                remote_metrics.update(self.membrain.get_metrics())
            
            metrics["remote_tier"] = remote_metrics
        
        return metrics
        
    def force_cache_to_cpu(self, num_blocks=10) -> int:
        """Force caching of GPU blocks to CPU tier.
        
        This is primarily used for testing to make sure the CPU tier works.
        It iterates through recently cached blocks and copies them to CPU cache.
        
        Args:
            num_blocks: Number of blocks to try to move to CPU
            
        Returns:
            Number of blocks successfully cached to CPU
        """
        if not self.cpu_cache:
            print(f"⚡ FORCE CPU: CPU cache tier not available")
            return 0
        
        print(f"⚡ FORCE CPU: Starting forced migration of {num_blocks} blocks to CPU")
        success_count = 0
        
        # Get recent cached blocks from block pool
        cached_blocks = []
        for hash_key, blocks_dict in self.block_pool.cached_block_hash_to_block.items():
            for block_id, block in blocks_dict.items():
                if block and block.block_hash:
                    cached_blocks.append((block, block.block_hash))
                    if len(cached_blocks) >= num_blocks:
                        break
            if len(cached_blocks) >= num_blocks:
                break
                
        print(f"⚡ FORCE CPU: Found {len(cached_blocks)} candidate blocks in GPU cache")
        if not cached_blocks:
            logger.warning("No cached blocks found to move to CPU tier")
            return 0
        
        # Store them in the CPU cache
        for idx, (block, block_hash) in enumerate(cached_blocks):
            hash_key = self._extract_hash_key(block_hash)
            block_tensor = self._get_block_tensor(block)
            
            if block_tensor is None:
                print(f"⚡ FORCE CPU: No tensor found for block {block.block_id}")
                logger.debug(f"No tensor available for block {block.block_id}")
                continue
                
            print(f"⚡ FORCE CPU: Moving block {idx+1}/{len(cached_blocks)} (ID: {block.block_id}) to CPU")
            self.cpu_store_attempts += 1
            # Make sure tensor is on CPU
            cpu_tensor = block_tensor.cpu() if block_tensor.device.type != "cpu" else block_tensor
            metadata = {
                "source": "forced_cpu_migration",
                "block_id": block.block_id,
                "timestamp": time.time(),
            }
            
            if self.cpu_cache.store(hash_key, cpu_tensor, metadata):
                self.cpu_store_successes += 1
                self.cpu_blocks.add(hash_key)
                success_count += 1
                print(f"⚡ FORCE CPU: Successfully moved block {block.block_id} to CPU cache with key {hash_key}")
                logger.info(f"Forced block {block.block_id} to CPU cache with key {hash_key}")
            else:
                print(f"⚡ FORCE CPU: Failed to move block {block.block_id} to CPU cache")
        
        print(f"⚡ FORCE CPU: Migration completed - moved {success_count}/{len(cached_blocks)} blocks to CPU cache")
        return success_count
        
    def __del__(self):
        """Clean up resources when the object is deleted"""
        if hasattr(self, '_event_loop') and self._event_loop is not None:
            if not self._event_loop.is_closed():
                self._event_loop.close()