# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple
import time
import asyncio
import aiohttp
import torch
import numpy as np
import struct
from urllib.parse import urljoin

from vllm.logger import init_logger

logger = init_logger(__name__)


class MembrainError(Exception):
    """Base exception for Membrain operations."""
    pass


@dataclass
class MembrainConfig:
    """Configuration for Membrain store"""
    endpoint: str = "http://localhost:9201"
    namespace: str = "default" 
    timeout: float = 30.0
    max_retries: int = 3
    retry_delay: float = 0.1
    enable_metrics: bool = False


@dataclass
class MembrainBlockMetadata:
    """Metadata for blocks stored in Membrain"""
    block_hash: str  # Block hash value 
    ref_count: int  # Global reference count
    last_access: float  # Timestamp for LRU
    node_id: str  # Owner node ID
    block_size: int  # Size of block
    dtype: str  # Data type of tensor
    tensor_shape: Tuple[int, ...]  # Shape of tensor


def _serialize_tensor(tensor: torch.Tensor) -> bytes:
    """Serialize tensor to bytes with metadata."""
    # Convert tensor to contiguous CPU numpy array
    tensor_cpu = tensor.cpu().detach().numpy()
    
    # Get tensor metadata
    dtype_str = str(tensor.dtype)
    shape = tensor.shape
    
    # Serialize metadata
    metadata = {
        "dtype": dtype_str,
        "shape": shape
    }
    metadata_bytes = json.dumps(metadata).encode()
    metadata_size = len(metadata_bytes)
    
    # Combine metadata size, metadata and tensor bytes
    size_bytes = struct.pack("!Q", metadata_size)  # 8 bytes for size
    return size_bytes + metadata_bytes + tensor_cpu.tobytes()


def _deserialize_tensor(data: bytes) -> torch.Tensor:
    """Deserialize bytes back to tensor."""
    # Extract metadata size
    metadata_size = struct.unpack("!Q", data[:8])[0]
    
    # Extract and parse metadata
    metadata_bytes = data[8:8+metadata_size]
    metadata = json.loads(metadata_bytes.decode())
    
    # Extract tensor data
    tensor_bytes = data[8+metadata_size:]
    
    # Reconstruct tensor
    tensor_np = np.frombuffer(tensor_bytes, dtype=np.dtype(metadata["dtype"]))
    tensor_np = tensor_np.reshape(metadata["shape"])
    return torch.from_numpy(tensor_np)


class MembrainStore:
    """Membrain-based distributed block storage for vLLM V1.
    
    This provides a distributed memory tier for KV cache blocks using the
    Membrain key-value store service. It handles:
    - Block storage/retrieval
    - Reference counting 
    - Cache metrics
    - Connection management
    """

    def __init__(
        self,
        config: Optional[MembrainConfig] = None,
        node_id: str = "default",  
        block_size: int = 16,
        dtype: torch.dtype = torch.float16,
    ) -> None:
        """Initialize Membrain store.
        
        Args:
            config: Optional Membrain configuration
            node_id: Unique ID for this node
            block_size: KV cache block size
            dtype: Tensor data type
        """
        self.config = config or MembrainConfig()
        self.node_id = node_id
        self.block_size = block_size
        self.dtype = dtype

        # Track in-flight operations
        self._pending_stores: Set[str] = set()
        self._pending_loads: Set[str] = set()
        
        # Metrics
        self.hits = 0
        self.misses = 0
        self.store_latencies: List[float] = []
        self.load_latencies: List[float] = []

        # Session state
        self._session: Optional[aiohttp.ClientSession] = None
        self._closed = False

    async def _ensure_session(self) -> None:
        """Ensure aiohttp session exists and is active."""
        if self._session is None or self._session.closed:
            connector = aiohttp.TCPConnector(limit=100, limit_per_host=100)
            self._session = aiohttp.ClientSession(connector=connector)

    async def _request(
        self,
        method: str,
        key: str,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None
    ) -> bytes:
        """Make HTTP request with retries."""
        if self._closed:
            raise MembrainError("Store is closed")

        timeout = timeout or self.config.timeout
        url = urljoin(self.config.endpoint, f"/memory/{self.config.namespace}/{key}")

        for attempt in range(self.config.max_retries):
            try:
                await self._ensure_session()
                async with self._session.request(
                    method=method,
                    url=url, 
                    data=data,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 404:
                        raise KeyError(f"Key not found: {key}")
                    elif response.status != 200:
                        raise MembrainError(f"HTTP {response.status}: {await response.text()}")
                    return await response.read()

            except asyncio.TimeoutError:
                if attempt == self.config.max_retries - 1:
                    raise MembrainError(f"Operation timed out after {timeout}s")
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))
            except aiohttp.ClientError as e:
                if attempt == self.config.max_retries - 1:
                    raise MembrainError(f"Connection failed: {e}")
                await asyncio.sleep(self.config.retry_delay * (2 ** attempt))

    async def store_block(
        self,
        block_hash: str,
        tensor: torch.Tensor,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """Store a block in Membrain asynchronously.
        
        Args:
            block_hash: Hash of the block
            tensor: Block tensor data
            metadata: Optional block metadata
            
        Returns:
            bool: Success status
        """
        if block_hash in self._pending_stores:
            logger.warning(f"Block {block_hash} already being stored, skipping")
            return False

        try:
            self._pending_stores.add(block_hash)
            start_time = time.time()
            logger.debug(f"Starting store operation for block {block_hash}")

            # Prepare metadata
            block_meta = MembrainBlockMetadata(
                block_hash=block_hash,
                ref_count=1,
                last_access=time.time(),
                node_id=self.node_id,
                block_size=self.block_size,
                dtype=str(tensor.dtype),
                tensor_shape=tensor.shape
            )
            
            # Store metadata with block key
            meta_key = f"{block_hash}:meta"
            await self._request(
                'PUT',
                meta_key,
                json.dumps(asdict(block_meta)).encode()
            )

            # Store tensor data with block hash as key
            tensor_bytes = _serialize_tensor(tensor)
            logger.info(f"MEMBRAIN STORE: Key {block_hash}, shape {tensor.shape}, size {len(tensor_bytes)} bytes")
            await self._request('PUT', block_hash, tensor_bytes)
            
            if self.config.enable_metrics:
                elapsed = time.time() - start_time
                self.store_latencies.append(elapsed)
                
            elapsed = time.time() - start_time
            logger.info(f"MEMBRAIN STORE: Stored block {block_hash[:8]}... in {elapsed:.3f}s, "
                       f"shape {tensor.shape}, node_id {self.node_id}")

            return True

        except Exception as e:
            logger.error(f"Failed to store block {block_hash}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        finally:
            if block_hash in self._pending_stores:
                self._pending_stores.remove(block_hash)

    async def load_block(
        self,
        block_hash: str
    ) -> Optional[torch.Tensor]:
        """Load a block from Membrain asynchronously.
        
        Args:
            block_hash: Hash of the block
            
        Returns:
            Optional[torch.Tensor]: Loaded tensor or None if not found
        """
        if block_hash in self._pending_loads:
            logger.warning(f"Block {block_hash} already being loaded, skipping")
            return None

        try:
            start_time = time.time()
            self._pending_loads.add(block_hash)
            logger.debug(f"Starting load operation for block {block_hash}")

            # First check metadata exists
            meta_key = f"{block_hash}:meta"
            try:
                logger.debug(f"Loading metadata for block {block_hash}")
                meta_bytes = await self._request('GET', meta_key)
                meta_dict = json.loads(meta_bytes.decode())
                block_meta = MembrainBlockMetadata(**meta_dict)
                logger.debug(f"Found metadata for block {block_hash}: ref_count={block_meta.ref_count}")
            except KeyError:
                if self.config.enable_metrics:
                    self.misses += 1
                logger.info(f"MEMBRAIN MISS: Block {block_hash[:8]}... not found in metadata")
                return None

            # Then load tensor data
            try:
                logger.debug(f"Loading tensor data for block {block_hash}")
                tensor_bytes = await self._request('GET', block_hash)
                logger.debug(f"Received {len(tensor_bytes)} bytes for block {block_hash}")
                
                tensor = _deserialize_tensor(tensor_bytes)
                logger.debug(f"Deserialized tensor for block {block_hash}, shape: {tensor.shape}")
                
                if self.config.enable_metrics:
                    self.hits += 1
                    self.load_latencies.append(time.time() - start_time)
                
                # Update access time
                block_meta.last_access = time.time()
                await self._request(
                    'PUT',
                    meta_key,
                    json.dumps(asdict(block_meta)).encode()
                )
                
                elapsed = time.time() - start_time
                logger.info(f"MEMBRAIN HIT: Loaded block {block_hash[:8]}... in {elapsed:.3f}s, " 
                           f"shape {tensor.shape}, from node {block_meta.node_id}")
                return tensor

            except KeyError:
                if self.config.enable_metrics:
                    self.misses += 1
                logger.info(f"MEMBRAIN MISS: Block data {block_hash[:8]}... not found")
                return None

        except Exception as e:
            logger.error(f"Failed to load block {block_hash}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

        finally:
            if block_hash in self._pending_loads:
                self._pending_loads.remove(block_hash)

    async def increment_ref(self, block_hash: str) -> int:
        """Increment block reference count.
        
        Args:
            block_hash: Block hash to increment
            
        Returns:
            New reference count
        """
        meta_key = f"{block_hash}:meta"
        try:
            # Get current metadata
            meta_bytes = await self._request('GET', meta_key)
            meta_dict = json.loads(meta_bytes.decode())
            block_meta = MembrainBlockMetadata(**meta_dict)
            
            # Increment ref count
            block_meta.ref_count += 1
            
            # Store updated metadata
            await self._request(
                'PUT',
                meta_key,
                json.dumps(asdict(block_meta)).encode()
            )
            
            return block_meta.ref_count
            
        except KeyError:
            return 0

    async def decrement_ref(self, block_hash: str) -> int:
        """Decrement block reference count and cleanup if zero.
        
        Args:
            block_hash: Block hash to decrement
            
        Returns:
            New reference count
        """
        meta_key = f"{block_hash}:meta"
        try:
            # Get current metadata
            meta_bytes = await self._request('GET', meta_key)
            meta_dict = json.loads(meta_bytes.decode())
            block_meta = MembrainBlockMetadata(**meta_dict)
            
            # Decrement ref count
            block_meta.ref_count -= 1
            
            if block_meta.ref_count <= 0:
                # Delete block data and metadata
                await self._request('DELETE', block_hash)
                await self._request('DELETE', meta_key)
                return 0
                
            # Store updated metadata
            await self._request(
                'PUT', 
                meta_key,
                json.dumps(asdict(block_meta)).encode()
            )
            
            return block_meta.ref_count
            
        except KeyError:
            return 0

    def get_metrics(self) -> Dict:
        """Get store metrics."""
        if not self.config.enable_metrics:
            return {}
            
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / (self.hits + self.misses) if (self.hits + self.misses) > 0 else 0,
            "avg_store_latency": np.mean(self.store_latencies) if self.store_latencies else 0,
            "avg_load_latency": np.mean(self.load_latencies) if self.load_latencies else 0,
            "pending_stores": len(self._pending_stores),
            "pending_loads": len(self._pending_loads)
        }

    async def close(self) -> None:
        """Close store and cleanup resources."""
        self._closed = True
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None

    async def __aenter__(self) -> 'MembrainStore':
        """Async context manager entry."""
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """Async context manager exit."""
        await self.close()