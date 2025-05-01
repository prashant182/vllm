# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass
from typing import Optional

from vllm.config import CacheConfig, LoRAConfig, ModelConfig, SchedulerConfig
from vllm.v1.core.membrain import MembrainConfig


@dataclass
class LLMEngineConfig:
    """Configuration for LLM Engine V1."""
    model_config: ModelConfig
    cache_config: CacheConfig
    scheduler_config: SchedulerConfig
    lora_config: Optional[LoRAConfig] = None
    membrain_config: Optional[MembrainConfig] = None  # Added Membrain support