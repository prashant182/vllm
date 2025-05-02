# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Set, Tuple, Union
import time
import asyncio
import aiohttp
import torch
import numpy as np
import struct
from urllib.parse import urljoin
import threading

from vllm.logger import init_logger

logger = init_logger(__name__)


class MembrainError(Exception):
    """Base exception for Membrain operations."""
    pass


class MembrainConnectionError(MembrainError):
    """Raised when connection to Membrain fails."""
    pass


class MembrainKeyError(MembrainError):
    """Raised when a key operation fails."""
    pass


class MembrainTimeoutError(MembrainError):
    """Raised when an operation times out."""
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
    pool_connections: int = 10
    pool_maxsize: int = 10


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
    # Ensure tensor is on CPU and contiguous
    if tensor.is_cuda:
        tensor = tensor.cpu()
    if not tensor.is_contiguous():
        tensor = tensor.contiguous()
    tensor_cpu = tensor.detach().numpy()
    
    # Get tensor metadata - convert torch dtype to string representation
    dtype_str = str(tensor.dtype)
    shape = tensor.shape
    
    # Ensure shape is serializable (convert to list)
    shape_list = list(shape)
    
    # Serialize metadata
    metadata = {
        "dtype": dtype_str,
        "shape": shape_list,
        "version": "1.0",  # Add version for forward compatibility
    }
    
    # Convert metadata to JSON bytes
    metadata_bytes = json.dumps(metadata).encode('utf-8')
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
    
    # Convert torch dtype string to numpy dtype
    dtype_str = metadata["dtype"]
    if dtype_str.startswith("torch."):
        # Map torch dtypes to numpy ones
        dtype_map = {
            "torch.float16": np.float16,
            "torch.float32": np.float32,
            "torch.float64": np.float64,
            "torch.int32": np.int32,
            "torch.int64": np.int64,
            "torch.uint8": np.uint8,
            "torch.int8": np.int8,
            "torch.bool": np.bool_
        }
        np_dtype = dtype_map.get(dtype_str, np.float32)
    else:
        # Try to use the string directly
        try:
            np_dtype = np.dtype(dtype_str)
        except TypeError:
            logger.warning(f"Unknown dtype: {dtype_str}, falling back to float32")
            np_dtype = np.float32
    
    # Reconstruct tensor
    try:
        tensor_np = np.frombuffer(tensor_bytes, dtype=np_dtype)
        tensor_np = tensor_np.reshape(metadata["shape"])
        return torch.from_numpy(tensor_np.copy())  # Copy to make it writable
    except Exception as e:
        logger.error(f"Error deserializing tensor: {e}, shape={metadata['shape']}, dtype={dtype_str}")
        # Emergency fallback - try with float32
        tensor_np = np.frombuffer(tensor_bytes, dtype=np.float32)
        tensor_np = tensor_np.reshape(metadata["shape"])
        return torch.from_numpy(tensor_np.copy())


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
            connector = aiohttp.TCPConnector(
                limit_per_host=self.config.pool_maxsize,
                limit=self.config.pool_connections
            )
            self._session = aiohttp.ClientSession(connector=connector)

    async def _request(
        self,
        method: str,
        key: str,
        data: Optional[bytes] = None,
        timeout: Optional[float] = None,
        content_type: str = "application/octet-stream"
    ) -> bytes:
        """Make HTTP request with retries.
        
        This method handles API requests to the Membrain service with retry logic.
        Uses the endpoint format /memory/{namespace}/{key} as per the working client example.
        """
        if self._closed:
            raise MembrainError("Store is closed")

        timeout = timeout or self.config.timeout
        
        # Use the correct endpoint format from the working client example
        url = urljoin(self.config.endpoint, f"/memory/{self.config.namespace}/{key}")
        
        logger.info(f"Membrain API: {method} {url}")
        
        for attempt in range(self.config.max_retries):
            try:
                await self._ensure_session()
                
                headers = {"Content-Type": content_type} if data else None
                logger.debug(f"Request: {method} {url} (attempt {attempt+1}/{self.config.max_retries})")
                if data:
                    logger.debug(f"Data size: {len(data)} bytes, content-type: {content_type}")
                
                async with self._session.request(
                    method=method,
                    url=url, 
                    data=data,
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.read()
                        logger.debug(f"Success: {method} {url} returned {len(result)} bytes")
                        return result
                    elif response.status == 404:
                        # For GET, 404 is a KeyError
                        logger.warning(f"Key not found: {key}")
                        if method == 'GET':
                            raise MembrainKeyError(f"Key not found: {key}")
                        else:
                            error_text = await response.text()
                            raise MembrainError(f"HTTP 404 for {method}: {error_text}")
                    else:
                        # Any other non-200 response is an error
                        error_text = await response.text()
                        error_msg = f"HTTP {response.status}: {error_text}"
                        logger.warning(f"Error: {method} {url} - {error_msg}")
                        raise MembrainError(error_msg)
            
            except MembrainKeyError:
                # Pass MembrainKeyError through directly
                raise
                
            except asyncio.TimeoutError:
                logger.warning(f"Timeout: {method} {url} after {timeout}s")
                if attempt == self.config.max_retries - 1:
                    raise MembrainTimeoutError(f"Request timed out after {timeout}s")
                
            except aiohttp.ClientError as e:
                logger.error(f"Connection error: {method} {url} - {e}")
                if attempt == self.config.max_retries - 1:
                    raise MembrainConnectionError(f"Connection failed: {e}")
            
            # Only sleep between retries if we're going to retry
            if attempt < self.config.max_retries - 1:
                wait_time = self.config.retry_delay * (2 ** attempt)
                logger.debug(f"Retry: Waiting {wait_time:.2f}s before attempt {attempt+2}")
                await asyncio.sleep(wait_time)

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
        # Extract a stable key from BlockHashType objects
        # BlockHashType can come in various forms: as the object itself, as a string,
        # or already extracted as a string representation
        hash_key = str(block_hash)
        
        if isinstance(block_hash, tuple) and hasattr(block_hash, "hash_value"):
            # If it's the actual BlockHashType object (a NamedTuple), use hash_value directly
            hash_key = str(block_hash.hash_value)
        elif "hash_value=" in hash_key:
            # If it's a string representation like "BlockHashType(hash_value=123, ...)"
            import re
            match = re.search(r"hash_value=(-?\d+)", hash_key)
            if match:
                hash_key = match.group(1)
        
        logger.debug(f"Using simplified hash key: {hash_key} (from {block_hash})")
            
        if hash_key in self._pending_stores:
            logger.warning(f"Block {hash_key} already being stored, skipping")
            return False

        try:
            self._pending_stores.add(hash_key)
            start_time = time.time()
            logger.debug(f"Starting store operation for block {hash_key}")

            # Prepare metadata including original hash info
            block_meta = MembrainBlockMetadata(
                block_hash=hash_key,
                ref_count=1,
                last_access=time.time(),
                node_id=self.node_id,
                block_size=self.block_size,
                dtype=str(tensor.dtype),
                tensor_shape=tensor.shape
            )
            
            # Add any additional metadata
            meta_dict = asdict(block_meta)
            if metadata:
                meta_dict.update(metadata)
                
            # Include original hash for debugging
            meta_dict["original_hash"] = str(block_hash)
            
            # Store tensor data with block hash as key
            tensor_bytes = _serialize_tensor(tensor)
            data_key = f"data:{hash_key}"
            meta_key = f"meta:{hash_key}"
            
            logger.warning(f"MEMBRAIN STORE: Data key {data_key}, shape {tensor.shape}, size {len(tensor_bytes)} bytes")
            await self._request('PUT', data_key, tensor_bytes, content_type="application/octet-stream")
            
            # Store metadata with meta key prefix
            logger.warning(f"MEMBRAIN STORE: Storing metadata with key {meta_key}")
            await self._request(
                'PUT',
                meta_key,
                json.dumps(meta_dict).encode(),
                content_type="application/json"
            )
            
            if self.config.enable_metrics:
                elapsed = time.time() - start_time
                self.store_latencies.append(elapsed)
                
            elapsed = time.time() - start_time
            logger.warning(f"✅ MEMBRAIN STORE SUCCESS: Block {hash_key} stored in {elapsed:.3f}s, "
                       f"shape {tensor.shape}, node_id {self.node_id}")

            return True

        except Exception as e:
            logger.error(f"❌ Failed to store block {hash_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

        finally:
            # Always clean up the pending stores tracking
            if hash_key in self._pending_stores:
                self._pending_stores.remove(hash_key)

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
        # Extract a stable key from the hash, using the same logic as store_block
        hash_key = str(block_hash)
        
        if isinstance(block_hash, tuple) and hasattr(block_hash, "hash_value"):
            # If it's the actual BlockHashType object (a NamedTuple), use hash_value directly
            hash_key = str(block_hash.hash_value)
        elif "hash_value=" in hash_key:
            # If it's a string representation like "BlockHashType(hash_value=123, ...)"
            import re
            match = re.search(r"hash_value=(-?\d+)", hash_key)
            if match:
                hash_key = match.group(1)
                
        logger.warning(f"MEMBRAIN LOAD: Loading block with key: {hash_key}")
                
        # Prevent concurrent loads of the same block
        if hash_key in self._pending_loads:
            logger.warning(f"Block {hash_key} already being loaded, skipping duplicate request")
            return None

        try:
            # Track the load operation
            start_time = time.time()
            self._pending_loads.add(hash_key)
            
            # Prepare keys using same format as store_block
            data_key = f"data:{hash_key}"
            meta_key = f"meta:{hash_key}"

            # First check if metadata exists - this tells us if the block exists
            try:
                # Get block metadata
                logger.warning(f"MEMBRAIN LOAD: Fetching metadata with key {meta_key}")
                meta_bytes = await self._request('GET', meta_key, content_type="application/json")
                meta_dict = json.loads(meta_bytes.decode())
                
                # Parse metadata into structured object
                block_meta = MembrainBlockMetadata(
                    block_hash=meta_dict.pop("block_hash", hash_key),
                    ref_count=meta_dict.pop("ref_count", 1),
                    last_access=meta_dict.pop("last_access", time.time()),
                    node_id=meta_dict.pop("node_id", self.node_id),
                    block_size=meta_dict.pop("block_size", self.block_size),
                    dtype=meta_dict.pop("dtype", str(self.dtype)),
                    tensor_shape=tuple(meta_dict.pop("tensor_shape", ())),
                )
                logger.warning(f"MEMBRAIN LOAD: Found metadata for block {hash_key}: ref_count={block_meta.ref_count}")
                
            except MembrainKeyError:
                # Metadata not found = cache miss
                if self.config.enable_metrics:
                    self.misses += 1
                logger.warning(f"MEMBRAIN MISS: Block {hash_key} metadata not found")
                return None
            except Exception as e:
                logger.error(f"MEMBRAIN ERROR: Failed to load metadata for block {hash_key}: {e}")
                return None

            # If metadata found, try to load the actual tensor data
            try:
                # Get the tensor bytes
                logger.warning(f"MEMBRAIN LOAD: Fetching tensor data with key {data_key}")
                tensor_bytes = await self._request('GET', data_key, content_type="application/octet-stream")
                logger.warning(f"MEMBRAIN LOAD: Received {len(tensor_bytes)} bytes for block {hash_key}")
                
                # Deserialize to PyTorch tensor
                tensor = _deserialize_tensor(tensor_bytes)
                logger.warning(f"MEMBRAIN LOAD: Deserialized tensor for block {hash_key}, shape: {tensor.shape}")
                
                # Track metrics
                if self.config.enable_metrics:
                    self.hits += 1
                    self.load_latencies.append(time.time() - start_time)
                
                # Update access time in metadata to support LRU eviction
                block_meta.last_access = time.time()
                await self._request(
                    'PUT',
                    meta_key,
                    json.dumps(asdict(block_meta)).encode(),
                    content_type="application/json"
                )
                
                # Log success and timing
                elapsed = time.time() - start_time
                logger.warning(f"✅ MEMBRAIN HIT: Loaded block {hash_key} in {elapsed:.3f}s, " 
                           f"shape {tensor.shape}, from node {block_meta.node_id}")
                return tensor

            except MembrainKeyError:
                # Tensor data not found = inconsistent state (metadata exists but data doesn't)
                if self.config.enable_metrics:
                    self.misses += 1
                logger.error(f"MEMBRAIN ERROR: Block {hash_key} has metadata but data is missing")
                return None
            except Exception as e:
                # Other errors during tensor loading
                logger.error(f"MEMBRAIN ERROR: Failed to load tensor for block {hash_key}: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None

        except Exception as e:
            # General exception during the whole load process
            logger.error(f"MEMBRAIN ERROR: Failed to load block {hash_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None

        finally:
            # Always clean up the pending loads tracking
            if hash_key in self._pending_loads:
                self._pending_loads.remove(hash_key)

    async def increment_ref(self, block_hash: str) -> int:
        """Increment block reference count.
        
        Args:
            block_hash: Block hash to increment
            
        Returns:
            New reference count
        """
        # Extract stable hash key using same logic as other methods
        hash_key = str(block_hash)
        
        if isinstance(block_hash, tuple) and hasattr(block_hash, "hash_value"):
            hash_key = str(block_hash.hash_value)
        elif "hash_value=" in hash_key:
            import re
            match = re.search(r"hash_value=(-?\d+)", hash_key)
            if match:
                hash_key = match.group(1)
        
        meta_key = f"meta:{hash_key}"
        try:
            # Get current metadata
            logger.debug(f"MEMBRAIN: Incrementing ref count for {hash_key}")
            meta_bytes = await self._request('GET', meta_key)
            meta_dict = json.loads(meta_bytes.decode())
            
            # Create a new clean metadata without extra fields that might cause errors
            clean_meta = {
                "block_hash": meta_dict.get("block_hash", hash_key),
                "ref_count": meta_dict.get("ref_count", 1),
                "last_access": meta_dict.get("last_access", time.time()),
                "node_id": meta_dict.get("node_id", self.node_id),
                "block_size": meta_dict.get("block_size", self.block_size),
                "dtype": meta_dict.get("dtype", str(self.dtype)),
                "tensor_shape": meta_dict.get("tensor_shape", ())
            }
            
            # Now create the metadata object
            block_meta = MembrainBlockMetadata(**clean_meta)
            
            # Increment ref count
            block_meta.ref_count += 1
            logger.debug(f"MEMBRAIN: Incremented ref count for {hash_key} to {block_meta.ref_count}")
            
            # Store updated metadata
            await self._request(
                'PUT',
                meta_key,
                json.dumps(asdict(block_meta)).encode(),
                content_type="application/json"
            )
            
            return block_meta.ref_count
            
        except MembrainKeyError:
            logger.warning(f"Tried to increment ref count for non-existent block {hash_key}")
            return 0
        except Exception as e:
            logger.error(f"Error incrementing ref count for block {hash_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return 0

    async def decrement_ref(self, block_hash: str) -> int:
        """Decrement block reference count and cleanup if zero.
        
        Args:
            block_hash: Block hash to decrement
            
        Returns:
            New reference count (0 if block was deleted)
        """
        # Extract stable hash key using same logic as other methods
        hash_key = str(block_hash)
        
        if isinstance(block_hash, tuple) and hasattr(block_hash, "hash_value"):
            hash_key = str(block_hash.hash_value)
        elif "hash_value=" in hash_key:
            import re
            match = re.search(r"hash_value=(-?\d+)", hash_key)
            if match:
                hash_key = match.group(1)
        
        data_key = f"data:{hash_key}"
        meta_key = f"meta:{hash_key}"
        
        try:
            # Get current metadata
            logger.debug(f"MEMBRAIN: Decrementing ref count for {hash_key}")
            meta_bytes = await self._request('GET', meta_key)
            meta_dict = json.loads(meta_bytes.decode())
            
            # Create a new clean metadata without extra fields that might cause errors
            clean_meta = {
                "block_hash": meta_dict.get("block_hash", hash_key),
                "ref_count": meta_dict.get("ref_count", 1),
                "last_access": meta_dict.get("last_access", time.time()),
                "node_id": meta_dict.get("node_id", self.node_id),
                "block_size": meta_dict.get("block_size", self.block_size),
                "dtype": meta_dict.get("dtype", str(self.dtype)),
                "tensor_shape": meta_dict.get("tensor_shape", ())
            }
            
            # Now create the metadata object
            block_meta = MembrainBlockMetadata(**clean_meta)
            
            # Decrement ref count
            block_meta.ref_count -= 1
            logger.debug(f"MEMBRAIN: Decremented ref count for {hash_key} to {block_meta.ref_count}")
            
            if block_meta.ref_count <= 0:
                # Delete block data and metadata when ref count reaches zero
                logger.warning(f"MEMBRAIN DELETE: Block {hash_key} ref count is {block_meta.ref_count}, removing from cache")
                
                # Delete both data and metadata
                await self._request('DELETE', data_key)
                await self._request('DELETE', meta_key)
                return 0
                
            # Store updated metadata
            await self._request(
                'PUT', 
                meta_key,
                json.dumps(asdict(block_meta)).encode(),
                content_type="application/json"
            )
            
            return block_meta.ref_count
            
        except MembrainKeyError:
            logger.warning(f"Tried to decrement ref count for non-existent block {hash_key}")
            return 0
        except Exception as e:
            logger.error(f"Error decrementing ref count for block {hash_key}: {e}")
            import traceback
            logger.error(traceback.format_exc())
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