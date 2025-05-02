# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Dict, List, Optional, Tuple
import asyncio
import torch
import time

from vllm.logger import init_logger
from vllm.v1.core.membrain import MembrainConfig, MembrainStore
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.specialized_manager import SpecializedManager
from vllm.v1.core.kv_cache_utils import BlockHashType, KVCacheBlock
from vllm.v1.request import Request

logger = init_logger(__name__)


class MembrainManager:
    """Manager for distributed memory tier using Membrain
    
    This class coordinates between the local KVCache system and the distributed
    Membrain store. It handles:
    - Block allocation/deallocation between tiers
    - Cache lookups across tiers
    - Reference counting coordination
    - Cache eviction policies
    """

    def __init__(
        self,
        node_id: str,
        block_size: int,
        dtype: torch.dtype,
        block_pool: BlockPool,
        membrain_config: Optional[MembrainConfig] = None,
    ) -> None:
        """Initialize the Membrain manager
        
        Args:
            node_id: Unique identifier for this node
            block_size: Size of KV cache blocks
            dtype: Data type of tensors
            block_pool: Local block pool instance
            membrain_config: Optional Membrain configuration
        """
        self.node_id = node_id
        self.block_size = block_size
        self.dtype = dtype
        self.block_pool = block_pool

        # Initialize Membrain store if config provided
        self.membrain = None
        if membrain_config:
            self.membrain = MembrainStore(
                config=membrain_config,
                node_id=node_id,
                block_size=block_size,
                dtype=dtype
            )

        # Track blocks being managed
        self.remote_blocks: Dict[str, KVCacheBlock] = {}
        self.pending_ops: Dict[str, asyncio.Task] = {}

    async def get_cached_blocks(
        self,
        request: Request,
        specialized_manager: SpecializedManager,
        block_hashes: List[BlockHashType],
    ) -> Tuple[List[KVCacheBlock], int]:
        """Get cached blocks for a request from both local and remote tiers
        
        Args:
            request: The request to get blocks for
            specialized_manager: Manager for attention pattern
            block_hashes: List of block hashes to look up
            
        Returns:
            Tuple of (cached blocks, number of tokens)
        """
        if not self.membrain:
            return [], 0

        # First check local cache via specialized manager
        local_blocks = specialized_manager.find_longest_cache_hit(block_hashes)
        if local_blocks:
            return local_blocks, len(local_blocks) * self.block_size

        # Check remote cache for missing blocks
        remote_blocks = []
        for block_hash in block_hashes:
            # Skip if we already have this block locally
            if self.block_pool.get_cached_block(block_hash):
                continue

            # Try to load from remote
            tensor = await self.membrain.load_block(block_hash)
            if tensor is None:
                break

            # Allocate new block and populate
            block = self._allocate_new_block()
            block.tensor = tensor  # type: ignore
            block.block_hash = block_hash
            remote_blocks.append(block)

            # Track reference
            self.remote_blocks[block_hash] = block
            await self.membrain.increment_ref(block_hash)

        return remote_blocks, len(remote_blocks) * self.block_size

    async def cache_blocks(
        self,
        request: Request,
        blocks: List[KVCacheBlock],
        block_hashes: List[BlockHashType],
    ) -> None:
        """Cache blocks in remote tier
        
        Args:
            request: Request these blocks belong to
            blocks: List of blocks to cache
            block_hashes: Corresponding block hashes
        """
        if not self.membrain:
            return

        for block, block_hash in zip(blocks, block_hashes):
            if not block.is_full():
                continue

            # Skip if already in remote cache
            if block_hash in self.remote_blocks:
                continue

            # Store block remotely
            success = await self.membrain.store_block(
                block_hash,
                block.tensor,  # type: ignore
                metadata={
                    "request_id": request.request_id,
                    "node_id": self.node_id,
                    "timestamp": time.time()
                }
            )

            if success:
                self.remote_blocks[block_hash] = block

    async def free_blocks(self, blocks: List[KVCacheBlock]) -> None:
        """Free blocks from both tiers
        
        Args:
            blocks: List of blocks to free
        """
        if not self.membrain:
            return

        for block in blocks:
            if not block.block_hash:
                continue

            if block.block_hash in self.remote_blocks:
                await self.membrain.decrement_ref(block.block_hash)
                del self.remote_blocks[block.block_hash]

    def _allocate_new_block(self) -> KVCacheBlock:
        """Allocate a new block from the pool"""
        block = self.block_pool.get_new_blocks(1)[0]
        return block

    async def close(self) -> None:
        """Cleanup resources"""
        if self.membrain:
            await self.membrain.close()