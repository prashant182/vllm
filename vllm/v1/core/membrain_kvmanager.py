# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Dict, List, Optional, Set
import uuid
import torch

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.membrain import MembrainConfig, MembrainStore
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.kv_cache_interface import KVCacheConfig 
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)


@dataclass
class MembrainKVConfig:
    """Configuration for Membrain-enabled KV Cache Manager"""
    membrain: MembrainConfig
    node_id: str = ""  # Will be auto-generated if empty
    enable_metrics: bool = False


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
        super().__init__(
            kv_cache_config=kv_cache_config,
            max_model_len=max_model_len,
            enable_caching=enable_caching,
            caching_hash_algo=caching_hash_algo,
            use_eagle=use_eagle,
            log_stats=log_stats,
            enable_kv_cache_events=enable_kv_cache_events,
        )

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
                block_size=self.block_size,
                dtype=kv_cache_spec.dtype if hasattr(kv_cache_spec, 'dtype') else torch.float16
            )
            
            logger.info(f"Initialized Membrain KV cache manager with endpoint {membrain_config.membrain.endpoint}")
            logger.info(f"Using namespace: {membrain_config.membrain.namespace}")
            logger.info(f"Node ID: {membrain_config.node_id}")

        # Track blocks in remote tier
        self.remote_blocks: Dict[str, KVCacheBlock] = {}
        
        # Create a single shared event loop for all async operations
        import asyncio
        self._event_loop = asyncio.new_event_loop()
        
        # Stats for monitoring
        self.store_attempts = 0
        self.store_successes = 0
        self.load_attempts = 0 
        self.load_successes = 0

    def get_computed_blocks(
        self,
        request: Request
    ) -> tuple[list[KVCacheBlock], int]:
        """Get computed blocks for a request from both local and remote tiers
        
        This extends the base implementation to check Membrain after local cache miss.
        
        Args:
            request: The request to get blocks for
            
        Returns:
            Tuple of (computed blocks, number of computed tokens)
        """
        # First try local cache
        local_blocks, num_local_tokens = super().get_computed_blocks(request)
        if local_blocks or not self.membrain:
            if local_blocks:
                logger.info(f"Found {len(local_blocks)} blocks in local cache for request {request.request_id}")
            return local_blocks, num_local_tokens

        # Check remote cache
        block_hashes = self.req_to_block_hashes[request.request_id]
        logger.info(f"Checking Membrain for {len(block_hashes)} blocks for request {request.request_id}")
        logger.debug(f"Block hashes: {block_hashes}")
        remote_blocks = []

        for block_hash in block_hashes:
            # Skip if we already have this block locally
            local_block = self.block_pool.get_cached_block(block_hash)
            if local_block:
                logger.info(f"Block {block_hash} already exists locally, skipping remote load")
                continue
                
            # Count attempts for metrics
            self.load_attempts += 1

            # Try to load from remote using shared event loop
            try:
                tensor = self._event_loop.run_until_complete(self.membrain.load_block(block_hash))
                
                if tensor is not None:
                    logger.debug(f"Successfully loaded block {block_hash} from Membrain")
                    self.load_successes += 1
                else:
                    logger.debug(f"Block {block_hash} not found in Membrain")
                    # Break the chain - we need all blocks in sequence
                    break
            except Exception as e:
                logger.error(f"Failed to load block {block_hash} from Membrain: {e}")
                tensor = None
                break

            # Allocate new block and populate
            block = self._allocate_new_block()
            if block.tensor is None:
                logger.warning(f"Block {block.block_id} has no tensor attribute")
            
            # Set block tensor and hash
            block.tensor = tensor  # type: ignore
            block.block_hash = block_hash
            remote_blocks.append(block)

            # Track block
            self.remote_blocks[block_hash] = block
            logger.info(f"Added remote block {block_hash} (block ID: {block.block_id}) to tracking")

        if remote_blocks:
            logger.info(f"Loaded {len(remote_blocks)} blocks from Membrain for request {request.request_id}")
        
        return remote_blocks, len(remote_blocks) * self.block_size

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
        """Cache full blocks in both local and remote tiers
        
        Extends base implementation to also cache blocks in Membrain.
        
        Args:
            request: The request these blocks belong to
            blocks: The blocks to potentially cache
            block_hashes: The block hashes
            num_cached_blocks: Number of blocks already cached
            num_full_blocks: Number of full blocks that should be cached
            block_size: Size of each block
            hash_fn: Hash function to use
        """
        # DEBUG: Log entry to cache_full_blocks
        logger.info(f"CACHE_FULL_BLOCKS ENTRY: request={request.request_id}, blocks={len(blocks)}, " 
                   f"num_cached_blocks={num_cached_blocks}, num_full_blocks={num_full_blocks}")
        
        # Force at least one block to be cached if we have blocks
        if len(blocks) > 0 and num_cached_blocks >= num_full_blocks:
            logger.info(f"FORCING block caching: Setting num_full_blocks={num_cached_blocks + 1}")
            num_full_blocks = num_cached_blocks + 1
        
        if num_cached_blocks == num_full_blocks:
            logger.debug(f"No new blocks to cache for request {request.request_id}")
            return
        
        # First do local caching
        super().cache_full_blocks(
            request,
            blocks,
            block_hashes,
            num_cached_blocks,
            num_full_blocks,
            block_size,
            hash_fn
        )
        
        # Debug log block states after super() call
        for i, block in enumerate(blocks):
            if i >= len(block_hashes):
                break
            logger.debug(f"Block {i} state: hash={block_hashes[i]}, is_full={(hasattr(block, 'is_full') and block.is_full())}")

        # Then cache in remote tier if enabled
        if not self.membrain:
            return

        logger.info(f"Caching blocks in Membrain: {num_cached_blocks} to {num_full_blocks-1} for request {request.request_id}")
        
        for i, (block, block_hash) in enumerate(zip(blocks[num_cached_blocks:num_full_blocks],
                                                 block_hashes[num_cached_blocks:])):
            idx = num_cached_blocks + i
            # Skip if already in remote
            if block_hash in self.remote_blocks:
                logger.debug(f"Block {block_hash} already in remote tracking, skipping")
                continue

            # Make sure block has tensor
            if getattr(block, 'tensor', None) is None:
                logger.warning(f"Block {block.block_id} has no tensor attribute, skipping remote caching")
                continue
            
            # Count attempts for metrics
            self.store_attempts += 1
                
            # Store block remotely
            try:
                block_tensor = block.tensor  # type: ignore
                if block_tensor is None:
                    logger.warning(f"Block tensor is None for block {block.block_id}, skipping")
                    continue
                
                logger.debug(f"Storing block {block_hash} to Membrain (block idx {idx})")
                
                # Create metadata for this block
                metadata = {
                    "request_id": request.request_id,
                    "node_id": self.membrain_config.node_id,  # type: ignore
                    "block_index": idx,
                    "timestamp": time.time()
                }
                
                # Use shared event loop for async operations
                success = self._event_loop.run_until_complete(self.membrain.store_block(
                    str(block_hash),
                    block_tensor,
                    metadata=metadata
                ))
                
                if success:
                    self.store_successes += 1
                    self.remote_blocks[block_hash] = block
                    logger.info(f"Successfully stored block {block_hash} to Membrain")
                else:
                    logger.warning(f"Failed to store block {block_hash} to Membrain")
                    
            except Exception as e:
                logger.error(f"Failed to store block {block_hash} to Membrain: {e}")
                import traceback
                logger.error(traceback.format_exc())

    def free(self, request: Request) -> None:
        """Free blocks for a request from both tiers
        
        This extends the base implementation to also handle remote blocks.
        
        Args:
            request: The request whose blocks should be freed
        """
        # Get blocks before freeing locally
        blocks = self.req_to_blocks.get(request.request_id, [])
        logger.debug(f"Freeing {len(blocks)} blocks for request {request.request_id}")

        # Free locally first
        super().free(request)

        # Then handle remote blocks if enabled
        if not self.membrain:
            return

        for block in blocks:
            if not block.block_hash:
                logger.debug(f"Block {block.block_id} has no hash, skipping remote cleanup")
                continue

            if block.block_hash in self.remote_blocks:
                logger.info(f"Decrementing reference count for remote block {block.block_hash}")
                try:
                    # Use shared event loop for async operations
                    self._event_loop.run_until_complete(self.membrain.decrement_ref(block.block_hash))
                    logger.debug(f"Successfully decremented ref count for block {block.block_hash}")
                except Exception as e:
                    logger.error(f"Failed to decrement ref count in Membrain: {e}")
                    
                # Remove from our tracking regardless of success
                del self.remote_blocks[block.block_hash]

    def _allocate_new_block(self) -> KVCacheBlock:
        """Helper to allocate a new block"""
        return self.block_pool.get_new_blocks(1)[0]

    def get_metrics(self) -> Dict:
        """Get metrics for both local and remote tiers"""
        metrics = super().make_prefix_cache_stats()
        
        # Add local tracking metrics
        membrain_metrics = {
            "store_attempts": self.store_attempts,
            "store_successes": self.store_successes,
            "load_attempts": self.load_attempts,
            "load_successes": self.load_successes,
            "store_success_rate": self.store_successes / self.store_attempts if self.store_attempts > 0 else 0.0,
            "load_success_rate": self.load_successes / self.load_attempts if self.load_attempts > 0 else 0.0,
            "tracked_remote_blocks": len(self.remote_blocks)
        }
        
        # Add remote metrics if available
        if self.membrain and self.membrain_config.enable_metrics:  # type: ignore
            membrain_metrics.update(self.membrain.get_metrics())
            
        metrics["membrain"] = membrain_metrics
        
        return metrics
        
    def __del__(self):
        """Clean up resources when the object is deleted"""
        if hasattr(self, '_event_loop') and self._event_loop is not None:
            if not self._event_loop.is_closed():
                self._event_loop.close()