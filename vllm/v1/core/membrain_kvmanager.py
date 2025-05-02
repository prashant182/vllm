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

        # Track blocks in remote tier
        self.remote_blocks: Dict[str, KVCacheBlock] = {}

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
        remote_blocks = []

        for block_hash in block_hashes:
            # Skip if we already have this block locally
            if self.block_pool.get_cached_block(block_hash):
                continue

            # Try to load from remote
            # We need to handle async methods differently in a sync context
            try:
                # Create a simple asyncio event loop to run the async code
                import asyncio
                loop = asyncio.new_event_loop()
                tensor = loop.run_until_complete(self.membrain.load_block(block_hash))
                loop.close()
            except Exception as e:
                logger.error(f"Failed to load block from Membrain: {e}")
                tensor = None
            if tensor is None:
                break

            # Allocate new block and populate
            block = self._allocate_new_block()
            block.tensor = tensor  # type: ignore
            block.block_hash = block_hash
            remote_blocks.append(block)

            # Track block
            self.remote_blocks[block_hash] = block

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

        # Then cache in remote tier if enabled
        if not self.membrain:
            return

        for block, block_hash in zip(blocks[num_cached_blocks:num_full_blocks],
                                   block_hashes[num_cached_blocks:]):
            # Skip if already in remote
            if block_hash in self.remote_blocks:
                continue

            # Store block remotely
            try:
                # Create a simple asyncio event loop to run the async code
                import asyncio
                loop = asyncio.new_event_loop()
                success = loop.run_until_complete(self.membrain.store_block(
                    block_hash,
                    block.tensor,  # type: ignore
                    metadata={
                        "request_id": request.request_id,
                        "node_id": self.membrain_config.node_id  # type: ignore
                    }
                ))
                loop.close()
            except Exception as e:
                logger.error(f"Failed to store block to Membrain: {e}")
                success = False

            if success:
                self.remote_blocks[block_hash] = block

    def free(self, request: Request) -> None:
        """Free blocks for a request from both tiers
        
        This extends the base implementation to also handle remote blocks.
        
        Args:
            request: The request whose blocks should be freed
        """
        # Get blocks before freeing locally
        blocks = self.req_to_blocks.get(request.request_id, [])

        # Free locally first
        super().free(request)

        # Then handle remote blocks if enabled
        if not self.membrain:
            return

        for block in blocks:
            if not block.block_hash:
                continue

            if block.block_hash in self.remote_blocks:
                try:
                    # Create a simple asyncio event loop to run the async code
                    import asyncio
                    loop = asyncio.new_event_loop()
                    loop.run_until_complete(self.membrain.decrement_ref(block.block_hash))
                    loop.close()
                except Exception as e:
                    logger.error(f"Failed to decrement ref count in Membrain: {e}")
                del self.remote_blocks[block.block_hash]

    def _allocate_new_block(self) -> KVCacheBlock:
        """Helper to allocate a new block"""
        return self.block_pool.get_new_blocks(1)[0]

    def get_metrics(self) -> Dict:
        """Get metrics for both local and remote tiers"""
        metrics = super().make_prefix_cache_stats()
        if self.membrain and self.membrain_config.enable_metrics:  # type: ignore
            metrics.update({
                "membrain": self.membrain.get_metrics()
            })
        return metrics