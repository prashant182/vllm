"""
This is a modified version of the membrain_impl.py file with fixes for BFloat16 serialization.
It can be used as a drop-in replacement for the original implementation.

To use it:
1. Copy this file to the appropriate location in your vLLM deployment
2. Update your imports to point to this implementation
"""

# SPDX-License-Identifier: Apache-2.0

import asyncio
import hashlib
import io
import threading
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, List, Optional, Any

import aiohttp
import torch

import vllm.envs as envs
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.base import (
    KVConnectorBase_V1, KVConnectorMetadata, KVConnectorRole)
from vllm.utils import cdiv
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.logger import init_logger

if TYPE_CHECKING:
    from vllm.attention.backends.abstract import AttentionMetadata
    from vllm.forward_context import ForwardContext
    from vllm.v1.core.sched.output import CachedRequestData, NewRequestData
    from vllm.v1.request import Request

logger = init_logger(__name__)


class MembrainError(Exception):
    """Base exception for all Membrain client errors."""
    pass


class MembrainKeyError(MembrainError):
    """Raised when a key operation fails."""
    pass


@dataclass
class MembrainConfig:
    """Configuration for Membrain client."""
    endpoint: str  # Base URL for Membrain (e.g., "http://localhost:9201")
    namespace: str = "vllm_kv"  # Namespace for keys
    timeout: float = 30.0  # Default operation timeout in seconds
    max_retries: int = 3  # Maximum number of retries for operations
    retry_delay: float = 0.1  # Base delay between retries in seconds


class MembrainClient:
    """Async client for Membrain operations."""

    def __init__(self, config: MembrainConfig):
        """Initialize the client with given configuration."""
        self._config = config
        self._session = None
        
    async def _ensure_session(self):
        """Ensure aiohttp session exists and is active."""
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=self._config.timeout)
            )
            
    async def put(self, key: str, value: bytes, timeout: Optional[float] = None) -> None:
        """Put a value into Membrain."""
        await self._ensure_session()
        
        url = f"{self._config.endpoint}/memory/{self._config.namespace}/{key}"
        logger.info(f"Attempting to put {key} into the store")
        for attempt in range(self._config.max_retries):
            try:
                async with self._session.put(url, data=value) as response:
                    if response.status != 200:
                        raise MembrainError(f"HTTP {response.status}: {await response.text()}")
                    return
            except Exception as e:
                if attempt == self._config.max_retries - 1:
                    raise MembrainError(f"Failed to put key {key}: {e}")
                await asyncio.sleep(self._config.retry_delay * (2 ** attempt))
                
    async def get(self, key: str, timeout: Optional[float] = None) -> bytes:
        """Get a value from Membrain."""
        await self._ensure_session()
        
        url = f"{self._config.endpoint}/memory/{self._config.namespace}/{key}"
        for attempt in range(self._config.max_retries):
            try:
                async with self._session.get(url) as response:
                    if response.status == 404:
                        raise MembrainKeyError(f"Key not found: {key}")
                    elif response.status != 200:
                        raise MembrainError(f"HTTP {response.status}: {await response.text()}")
                    return await response.read()
            except MembrainKeyError:
                raise
            except Exception as e:
                if attempt == self._config.max_retries - 1:
                    raise MembrainError(f"Failed to get key {key}: {e}")
                await asyncio.sleep(self._config.retry_delay * (2 ** attempt))
                
    async def exists(self, key: str, timeout: Optional[float] = None) -> bool:
        """Check if a key exists in Membrain."""
        try:
            await self.get(key, timeout)
            return True
        except MembrainKeyError:
            return False
        except Exception as e:
            logger.error(f"Error checking if key {key} exists: {e}")
            return False
                
    async def close(self) -> None:
        """Close the client and cleanup resources."""
        if self._session and not self._session.closed:
            await self._session.close()
            self._session = None


@dataclass
class LoadSpec:
    # Number of tokens cached in vLLM
    vllm_cached_tokens: int
    # Number of tokens that are cached in Membrain
    membrain_cached_tokens: int
    # Whether the scheduler allows us to load the tokens
    can_load: bool


@dataclass
class SaveSpec:
    # Skip already saved tokens
    skip_leading_tokens: int
    # Whether the scheduler allows us to save the tokens
    can_save: bool


@dataclass
class RequestTracker:
    # Request id
    req_id: str
    # The token ids that have been scheduled so far
    token_ids: list[int]
    # The block ids that have been allocated so far
    allocated_block_ids: list[int]
    # The number of tokens that have been saved
    num_saved_tokens: int = 0
    
    @staticmethod
    def from_new_request(
        new_request: "NewRequestData",
        num_tokens_to_compute: int,
    ) -> "RequestTracker":
        """Create the request tracker from a new request."""
        return RequestTracker(
            req_id=new_request.req_id,
            token_ids=new_request.prompt_token_ids[:num_tokens_to_compute].copy(),
            allocated_block_ids=new_request.block_ids.copy(),
            num_saved_tokens=0,
        )
    
    def update(self, cached_request: "CachedRequestData") -> None:
        """Update the request tracker when a running request is scheduled again"""
        self.token_ids.extend(cached_request.new_token_ids)
        self.allocated_block_ids.extend(cached_request.new_block_ids)


@dataclass
class ReqMeta:
    # Request id
    req_id: str
    # Request tokens
    token_ids: torch.Tensor
    # Slot mapping
    slot_mapping: torch.Tensor
    # Skip save or not
    save_spec: Optional[SaveSpec] = None
    # load_spec
    load_spec: Optional[LoadSpec] = None
    
    @staticmethod
    def from_request_tracker(
        tracker: RequestTracker,
        block_size: int,
        chunk_size: int = 256,
        load_spec: Optional[LoadSpec] = None,
        skip_save: bool = False,
        discard_partial_chunks: bool = True,
    ) -> Optional["ReqMeta"]:
        """Create the request metadata from a request tracker."""
        input_token_ids = tracker.token_ids
        input_token_len = len(input_token_ids)
        
        # For save operation: do not save if the following condition is met
        # 1. has already been saved before (num_saved_tokens > 0)
        # 2. number of unsaved tokens is not reached the chunk boundary
        skip_leading_tokens = tracker.num_saved_tokens
        chunk_boundary = cdiv(tracker.num_saved_tokens, chunk_size) * chunk_size
        skip_save = skip_save or (tracker.num_saved_tokens > 0 and input_token_len < chunk_boundary)
        
        if skip_save and load_spec is None:
            return None
            
        # Calculate number of tokens to save based on discard_partial_chunks setting
        num_tokens_to_save = (input_token_len // chunk_size * chunk_size) if discard_partial_chunks else input_token_len
        
        # If we need to save, update the number of saved tokens
        if not skip_save:
            tracker.num_saved_tokens = num_tokens_to_save
        save_spec = SaveSpec(skip_leading_tokens, not skip_save)
        
        # Calculate the token ids and slot mappings for load and save
        token_ids = torch.tensor(input_token_ids)[:num_tokens_to_save]
        num_blocks = len(tracker.allocated_block_ids)
        block_ids = torch.tensor(tracker.allocated_block_ids, dtype=torch.long)
        
        if len(token_ids) > num_blocks * block_size:
            logger.error(
                "The number of tokens is more than the number of blocks. "
                "Something might be wrong in scheduling logic!")
            logger.error(f"Num tokens: {len(token_ids)}, num blocks: {num_blocks}, block size: {block_size}")
        
        # Generate slot mapping
        block_offsets = torch.arange(0, block_size, dtype=torch.long)
        slot_mapping = block_offsets.reshape((1, block_size)) + block_ids.reshape((num_blocks, 1)) * block_size
        
        slot_mapping = slot_mapping.flatten()[:len(token_ids)]
        assert slot_mapping.dtype == torch.long
        
        # For load operation: check whether the request is scheduled to load
        if load_spec is not None and load_spec.can_load:
            logger.debug(f"Scheduled to load {load_spec.membrain_cached_tokens} tokens for request {tracker.req_id}")
        else:
            # Do not load if not in `can_load` state
            load_spec = None
            
        return ReqMeta(
            req_id=tracker.req_id,
            token_ids=token_ids,
            slot_mapping=slot_mapping,
            save_spec=save_spec,
            load_spec=load_spec,
        )


@dataclass
class MembrainConnectorMetadata(KVConnectorMetadata):
    requests: list[ReqMeta]
    
    def __init__(self):
        self.requests = []
        
    def add_request(self, req_meta: ReqMeta) -> None:
        """Add a request to the metadata."""
        self.requests.append(req_meta)


class MembrainConnectorV1Impl:
    """Implementation of the Membrain connector for vLLM v1."""

    def __init__(self, vllm_config: "VllmConfig", role: KVConnectorRole, 
                parent: KVConnectorBase_V1):
        self._parent = parent
        self.kv_role = vllm_config.kv_transfer_config.kv_role
        
        # Initialize Membrain client with configurable endpoints
        # Check environment variables first, then fallback to config
        import os
        
        membrain_endpoint = os.environ.get(
            "MEMBRAIN_ENDPOINT",
            vllm_config.kv_transfer_config.get_from_extra_config(
                "membrain_endpoint", "http://localhost:9201")
        )
        
        membrain_namespace = os.environ.get(
            "MEMBRAIN_NAMESPACE", 
            vllm_config.kv_transfer_config.get_from_extra_config(
                "membrain_namespace", "vllm_kv")
        )
        
        membrain_timeout = float(os.environ.get(
            "MEMBRAIN_TIMEOUT",
            vllm_config.kv_transfer_config.get_from_extra_config(
                "membrain_timeout", "30.0")
        ))
        
        membrain_retries = int(os.environ.get(
            "MEMBRAIN_MAX_RETRIES",
            vllm_config.kv_transfer_config.get_from_extra_config(
                "membrain_max_retries", "3")
        ))
        
        logger.info(f"Configuring Membrain connector with endpoint {membrain_endpoint}, namespace {membrain_namespace}")
        
        self._membrain_config = MembrainConfig(
            endpoint=membrain_endpoint,
            namespace=membrain_namespace,
            timeout=membrain_timeout,
            max_retries=membrain_retries
        )
        
        # We need to run a separate event loop for the async client
        self._event_loop = asyncio.new_event_loop()
        self._thread = threading.Thread(target=self._start_event_loop, daemon=True)
        self._thread.start()
        
        # Run in the event loop to create the client
        future = asyncio.run_coroutine_threadsafe(self._create_client(), self._event_loop)
        self._membrain_client = future.result()
        
        # Additional configuration
        self._block_size = vllm_config.cache_config.block_size
        
        import os
        self._chunk_size = int(os.environ.get(
            "MEMBRAIN_CHUNK_SIZE",
            vllm_config.kv_transfer_config.get_from_extra_config("chunk_size", "256"))
        )
        
        self._discard_partial_chunks = os.environ.get(
            "MEMBRAIN_DISCARD_PARTIAL_CHUNKS", 
            vllm_config.kv_transfer_config.get_from_extra_config("discard_partial_chunks", "False")
        ).lower() in ('true', '1', 'yes')
        
        self._model_name = vllm_config.model_config.model
        
        logger.info(f"Membrain connector configured with chunk_size={self._chunk_size}, "
                   f"discard_partial_chunks={self._discard_partial_chunks}")
        
        # State tracking
        self._request_trackers = {}  # request_id -> RequestTracker
        self._load_specs = {}  # request_id -> LoadSpec
        self.kv_caches = {}  # layer_name -> kv_tensor
        
        logger.info(f"Initialized MembrainConnectorV1Impl with role {self.kv_role}")
    
    def _start_event_loop(self):
        """Start the event loop in a separate thread."""
        asyncio.set_event_loop(self._event_loop)
        self._event_loop.run_forever()
    
    async def _create_client(self):
        """Create the Membrain client."""
        return MembrainClient(self._membrain_config)
    
    def _run_async(self, coro):
        """Run a coroutine in the event loop and wait for the result."""
        future = asyncio.run_coroutine_threadsafe(coro, self._event_loop)
        return future.result()
    
    def _generate_key(self, token_ids: torch.Tensor) -> str:
        """Generate a unique key for the token sequence."""
        # Format: namespace:model:tokens_hash
        token_hash = hashlib.md5(str(token_ids.tolist()).encode()).hexdigest()
        return f"{token_hash}"

    def _serialize_kv_data(self, request: ReqMeta) -> bytes:
        """Serialize KV cache data for all layers for the given request."""
        # Create a buffer to store the serialized data
        buffer = io.BytesIO()
        
        # Collect all layer KV caches
        kvcaches = list(self.kv_caches.values())
        if not kvcaches:
            logger.warning("No KV caches available for serialization")
            return b''
        
        # Get dimensions for metadata
        num_layers = len(kvcaches)
        
        logger.info(f"Serializing KV data for {num_layers} layers, {len(request.token_ids)} tokens")
        
        try:
            # For simplicity and robustness, use pickle instead of torch.save
            import pickle
            
            # Prepare the data structure to serialize
            serialized_data = {
                "metadata": {
                    "num_layers": num_layers,
                    "num_tokens": len(request.token_ids),
                    "token_ids": request.token_ids.tolist(),
                    "model_name": self._model_name,
                },
                "layers": []
            }
            
            # Add each layer's KV cache
            for layer_idx, kv_cache in enumerate(kvcaches):
                # Get the relevant slices from the KV cache
                indices = request.slot_mapping
                
                # Extract tensor for specific slots
                kv_slice = torch.index_select(kv_cache, 1, indices.cuda())
                
                # Instead of direct numpy conversion which fails for BFloat16, use PyTorch serialization
                tensor_buffer = io.BytesIO()
                torch.save(kv_slice.cpu(), tensor_buffer)
                tensor_buffer.seek(0)
                
                # Add to the layers list with serialized tensor data
                serialized_data["layers"].append({
                    "layer_idx": layer_idx,
                    "shape": list(kv_slice.shape),
                    "dtype": str(kv_slice.dtype),
                    "tensor_bytes": tensor_buffer.read(),
                })
            
            # Serialize the entire structure
            pickle.dump(serialized_data, buffer, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Get the bytes from the buffer
            buffer.seek(0)
            data = buffer.read()
            logger.info(f"Serialized KV data size: {len(data)} bytes")
            return data
            
        except Exception as e:
            logger.error(f"Error during serialization: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return b''

    def _deserialize_kv_data(self, data: bytes, kvcaches: list, slot_mapping: torch.Tensor) -> bool:
        """Deserialize KV cache data and load it into the provided kvcaches list."""
        if not data:
            logger.warning("Empty data provided for deserialization")
            return False
        
        # Create a buffer from the bytes
        buffer = io.BytesIO(data)
        
        try:
            logger.info(f"Attempting to deserialize data of size: {len(data)} bytes")
            
            # Use pickle for deserialization to match our serialization
            import pickle
            deserialized_data = pickle.load(buffer)
            
            # Extract metadata
            metadata = deserialized_data["metadata"]
            num_layers = metadata["num_layers"]
            num_tokens = metadata["num_tokens"]
            
            logger.info(f"Deserialized metadata: {num_layers} layers, {num_tokens} tokens")
            
            # Check if we have enough layers
            if num_layers != len(kvcaches):
                logger.warning(f"Layer count mismatch: {num_layers} vs {len(kvcaches)}")
                return False
            
            # Process each layer
            for layer_info in deserialized_data["layers"]:
                layer_idx = layer_info["layer_idx"]
                
                # Load the tensor from bytes
                tensor_buffer = io.BytesIO(layer_info["tensor_bytes"])
                kv_slice = torch.load(tensor_buffer).cuda()
                
                # Insert into the correct positions
                dst_cache = kvcaches[layer_idx]
                
                # Use index_copy_ to place the data in the right slots
                dst_cache.index_copy_(1, slot_mapping.cuda(), kv_slice)
                
                logger.debug(f"Successfully loaded layer {layer_idx}")
            
            logger.info(f"Successfully deserialized all {num_layers} layers")
            return True
            
        except Exception as e:
            logger.error(f"Error deserializing KV cache: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return False

    def _init_kv_caches_from_forward_context(self, forward_context: "ForwardContext"):
        """Initialize the KV caches from the forward context."""
        for layer_name in forward_context.no_compile_layers:
            attn_layer = forward_context.no_compile_layers[layer_name]
            if not hasattr(attn_layer, "kv_cache"):
                logger.debug(f"Layer {layer_name} does not have kv_cache, skipping")
                continue
                
            if layer_name not in self.kv_caches:
                self.kv_caches[layer_name] = attn_layer.kv_cache[forward_context.virtual_engine]
    
    # ==============================
    # Worker-side methods
    # ==============================
    
    def start_load_kv(self, forward_context: "ForwardContext", **kwargs) -> None:
        """
        Start loading the KV cache from Membrain to vLLM's paged KV buffer.
        
        Args:
            forward_context (ForwardContext): the forward context.
            **kwargs: additional arguments for the load operation
        """
        # Initialize KV caches if not done already
        if len(self.kv_caches) == 0:
            self._init_kv_caches_from_forward_context(forward_context)
            
        # Get metadata from parent connector
        metadata = self._parent._get_connector_metadata()
        assert isinstance(metadata, MembrainConnectorMetadata)
        
        # Ensure we have KV caches
        if len(self.kv_caches) == 0:
            logger.warning("No KV caches available in start_load_kv")
            return
            
        kvcaches = list(self.kv_caches.values())
        
        # Get attention metadata
        attn_metadata = forward_context.attn_metadata
        if attn_metadata is None:
            logger.warning("No attention metadata available in start_load_kv")
            return
        
        # Process each request that needs loading
        for request in metadata.requests:
            if request.load_spec is None or not request.load_spec.can_load:
                continue
                
            # Generate key and retrieve from Membrain
            key = self._generate_key(request.token_ids)
            
            try:
                # Retrieve KV cache data from Membrain
                kv_data = self._run_async(self._membrain_client.get(key))
                
                if kv_data:
                    # Deserialize and load into vLLM's KV cache
                    success = self._deserialize_kv_data(
                        kv_data, kvcaches, request.slot_mapping
                    )
                    
                    if success:
                        logger.info(
                            f"Loaded KV cache for request {request.req_id} with "
                            f"{len(request.token_ids)} tokens"
                        )
                    else:
                        logger.error(
                            f"Failed to load KV cache for request {request.req_id}"
                        )
            except Exception as e:
                logger.error(f"Error loading KV cache from Membrain: {e}")

    def wait_for_layer_load(self, layer_name: str) -> None:
        """
        Block until the KV for a specific layer is loaded into vLLM's paged buffer.
        
        Args:
            layer_name: the name of the layer
        """
        # Our implementation loads all layers at once, so nothing to do here
        return

    def save_kv_layer(self, layer_name: str, kv_layer: torch.Tensor, 
                    attn_metadata: "AttentionMetadata", **kwargs) -> None:
        """
        Save a layer of KV cache to the connector.
        
        Args:
            layer_name (str): the name of the layer.
            kv_layer (torch.Tensor): the paged KV buffer of the current layer.
            attn_metadata (AttentionMetadata): the attention metadata.
            **kwargs: additional arguments for the save operation.
        """
        # Store the layer's KV cache for later serialization in wait_for_save
        if layer_name not in self.kv_caches:
            self.kv_caches[layer_name] = kv_layer

    def wait_for_save(self):
        """
        Block until all the save operations are done.
        """
        # Skip if we're a consumer-only node
        if self.kv_role == "kv_consumer":
            return
            
        # Get connector metadata
        connector_metadata = self._parent._get_connector_metadata()
        assert isinstance(connector_metadata, MembrainConnectorMetadata)
        
        # Ensure we have KV caches to save
        if len(self.kv_caches) == 0:
            logger.warning("No KV caches available for saving")
            return
            
        # Process each request that needs saving
        for request in connector_metadata.requests:
            save_spec = request.save_spec
            if save_spec is None or not save_spec.can_save:
                continue
                
            # Serialize the KV cache data
            kv_data = self._serialize_kv_data(request)
            if not kv_data:
                logger.warning(f"No KV data to save for request {request.req_id}")
                continue
                
            # Generate key for storage
            key = self._generate_key(request.token_ids)
            
            try:
                # Save to Membrain
                self._run_async(self._membrain_client.put(key, kv_data))
                logger.info(
                    f"Saved KV cache for request {request.req_id} with "
                    f"{len(request.token_ids)} tokens"
                )
            except Exception as e:
                logger.error(f"Error saving KV cache to Membrain: {e}")
    
    # ==============================
    # Scheduler-side methods
    # ==============================
    
    def get_num_new_matched_tokens(
        self,
        request: "Request",
        num_computed_tokens: int,
    ) -> int:
        """
        Get number of new tokens that can be loaded from Membrain beyond the num_computed_tokens.
        
        Args:
            request (Request): the request object.
            num_computed_tokens (int): the number of locally computed tokens for this request
            
        Returns:
            int: the number of tokens that can be loaded from Membrain beyond what is already computed.
        """
        if self.kv_role == "kv_producer":
            return 0
            
        token_ids = torch.tensor(request.prompt_token_ids)
        key = self._generate_key(token_ids)
        
        # Check if the KV cache exists in Membrain
        try:
            exists = self._run_async(self._membrain_client.exists(key))
            if exists:
                # If exists, the entire token sequence is available
                num_external_hit_tokens = len(token_ids)
                
                # When prompt length is exactly divisible by the block size,
                # we need to recompute the last token to ensure correctness
                if num_external_hit_tokens == request.num_tokens:
                    num_external_hit_tokens -= 1
                    
                need_to_allocate = num_external_hit_tokens - num_computed_tokens
                
                logger.info(
                    f"Request {request.request_id}: Total tokens {request.num_tokens}, "
                    f"Membrain hit tokens: {num_external_hit_tokens}, "
                    f"Need to load: {need_to_allocate}"
                )
                
                if need_to_allocate <= 0:
                    return 0
                    
                # Store the load spec for later use
                self._load_specs[request.request_id] = LoadSpec(
                    vllm_cached_tokens=num_computed_tokens,
                    membrain_cached_tokens=num_external_hit_tokens,
                    can_load=False
                )
                
                return need_to_allocate
        except Exception as e:
            logger.error(f"Error checking Membrain for key {key}: {e}")
        
        return 0

    def update_state_after_alloc(self, request: "Request", num_external_tokens: int):
        """
        Update KVConnector state after temporary buffer alloc.
        
        For MembrainConnectorV1, update load specs if the CacheManager has allocated
        blocks for us.
        
        Args:
            request (Request): the request object.
            num_external_tokens (int): number of tokens to be loaded from external.
        """
        if request.request_id not in self._load_specs:
            # No KV tokens from external KV cache, return
            return
            
        if num_external_tokens == 0:
            # No need to load anything
            self._load_specs[request.request_id].can_load = False
            return
            
        # Sanity check
        expected_tokens = (
            self._load_specs[request.request_id].membrain_cached_tokens -
            self._load_specs[request.request_id].vllm_cached_tokens
        )
        assert num_external_tokens > 0 and num_external_tokens == expected_tokens, (
            f"Mismatch in number of tokens: {num_external_tokens} vs {expected_tokens} "
            f"for request {request.request_id}"
        )
        
        # Mark that we can load this request's data
        self._load_specs[request.request_id].can_load = True

    def build_connector_meta(self, scheduler_output: SchedulerOutput) -> KVConnectorMetadata:
        """
        Build the connector metadata for this step.
        
        This function resets the state of the connector and creates metadata
        for the worker to use for loading/saving KV caches.
        
        Args:
            scheduler_output (SchedulerOutput): the scheduler output object.
            
        Returns:
            MembrainConnectorMetadata: the connector metadata.
        """
        force_skip_save = self.kv_role == "kv_consumer"
        
        meta = MembrainConnectorMetadata()
        
        # Clean up finished requests
        for finished_req_id in scheduler_output.finished_req_ids:
            self._request_trackers.pop(finished_req_id, None)
        
        # Process new requests
        for request in scheduler_output.scheduled_new_reqs:
            # Check if we need to load KV for this request
            load_spec = self._load_specs.pop(request.req_id, None)
            
            # Calculate total tokens to compute
            num_tokens_to_compute = (
                request.num_computed_tokens + 
                scheduler_output.num_scheduled_tokens[request.req_id]
            )
            
            # Create a tracker for this request
            request_tracker = RequestTracker.from_new_request(
                request, num_tokens_to_compute
            )
            self._request_trackers[request.req_id] = request_tracker
            
            # Create metadata for this request
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._chunk_size,
                load_spec=load_spec,
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks
            )
            
            if req_meta is not None:
                meta.add_request(req_meta)
        
        # Process cached (continuing) requests
        for request in scheduler_output.scheduled_cached_reqs:
            # Update the request tracker
            request_tracker = self._request_trackers[request.req_id]
            request_tracker.update(request)
            
            # Create metadata for this request
            req_meta = ReqMeta.from_request_tracker(
                request_tracker,
                self._block_size,
                self._chunk_size,
                load_spec=None,  # We don't load KV for continuing requests
                skip_save=force_skip_save,
                discard_partial_chunks=self._discard_partial_chunks
            )
            
            if req_meta is not None:
                meta.add_request(req_meta)
        
        return meta