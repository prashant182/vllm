# SPDX-License-Identifier: Apache-2.0

import pytest
import torch

from vllm.v1.core.membrain import MembrainConfig, MembrainStore, MembrainBlockMetadata
from vllm.v1.core.membrain_kvmanager import MembrainKVConfig, MembrainKVCacheManager
from vllm.v1.kv_cache_interface import KVCacheConfig, FullAttentionSpec


@pytest.fixture
def membrain_config():
    return MembrainConfig(
        host="localhost",
        port=9201,
        enable_metrics=True
    )


@pytest.fixture
def kv_cache_config():
    return KVCacheConfig(
        num_blocks=100,
        tensors={},
        kv_cache_groups=[{
            "layer_names": ["layer0"],
            "kv_cache_spec": FullAttentionSpec(
                block_size=16,
                num_kv_heads=32,
                head_size=128,
                dtype=torch.float16,
                use_mla=False
            )
        }]
    )


def test_membrain_store_init(membrain_config):
    """Test MembrainStore initialization"""
    store = MembrainStore(
        config=membrain_config,
        node_id="test_node",
        block_size=16,
        dtype=torch.float16
    )
    
    assert store.node_id == "test_node"
    assert store.block_size == 16
    assert store.dtype == torch.float16
    assert store._pending_stores == set()
    assert store._pending_loads == set()


def test_membrain_store_block(membrain_config):
    """Test storing a block in Membrain"""
    store = MembrainStore(
        config=membrain_config,
        node_id="test_node", 
        block_size=16,
        dtype=torch.float16
    )

    tensor = torch.randn(1, 16, dtype=torch.float16)
    success = await store.store_block(  # type: ignore
        block_hash="test_hash",
        tensor=tensor
    )

    assert success
    assert len(store._pending_stores) == 0
    
    # Verify metrics
    metrics = store.get_metrics()
    assert "store_latencies" in metrics
    assert len(metrics["store_latencies"]) == 1


def test_membrain_load_block(membrain_config):
    """Test loading a block from Membrain"""
    store = MembrainStore(
        config=membrain_config,
        node_id="test_node",
        block_size=16, 
        dtype=torch.float16
    )

    tensor = await store.load_block("test_hash")  # type: ignore

    assert tensor is None  # Not implemented yet
    assert len(store._pending_loads) == 0

    # Verify metrics
    metrics = store.get_metrics()
    assert metrics["misses"] == 1


def test_membrain_kvcache_manager(membrain_config, kv_cache_config):
    """Test KVCacheManager with Membrain integration"""
    membrain_kv_config = MembrainKVConfig(
        membrain=membrain_config,
        enable_metrics=True
    )

    manager = MembrainKVCacheManager(
        kv_cache_config=kv_cache_config,
        max_model_len=2048,
        membrain_config=membrain_kv_config,
        enable_caching=True,
        log_stats=True
    )

    assert manager.membrain is not None
    assert manager.block_size == 16
    assert len(manager.remote_blocks) == 0

    # Test metrics
    metrics = manager.get_metrics()
    assert "membrain" in metrics