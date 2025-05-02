# Membrain Tiered Caching Architecture

## System Overview

The Membrain tiered caching architecture for vLLM provides a hierarchical approach to KV cache management, leveraging the strengths of each storage tier:

```
┌─────────────────────────────────────────┐
│                                         │
│            vLLM LLM Engine              │
│                                         │
└────────────────┬────────────────────────┘
                 │
         ┌────────┴──────┐
         ▼               │
┌─────────────────┐      │
│                 │      │
│  KVCacheManager │◄─────┤
│                 │      │
└────────┬────────┘      │
         │               │
         ▼               │
┌─────────────────────────────────────────┐
│                                         │
│        MembrainKVCacheManager           │
│                                         │
├─────────┬─────────────────┬─────────────┤
│         │                 │             │
│ ┌───────┴───────┐ ┌───────┴───────┐     │
│ │               │ │               │     │
│ │  Block Pool   │ │ Policy Manager│     │
│ │  (GPU Tier)   │ │               │     │
│ │               │ │               │     │
│ └───────────────┘ └───────────────┘     │
│                                         │
└───────────────────┬─────────────────────┘
                    │
         ┌──────────┴─────────┐
         ▼                    ▼
┌─────────────────┐   ┌───────────────────┐
│                 │   │                   │
│   CPU Cache     │   │   Membrain Store  │
│   (Medium Tier) │   │   (Remote Tier)   │
│                 │   │                   │
└─────────────────┘   └───────────────────┘
```

## Data Flow

### Cache Hit Flow

```
┌─────────┐     ┌────────┐     ┌─────────┐     ┌──────────┐
│         │     │        │     │         │     │          │
│ Request │────▶│GPU Tier│────▶│CPU Tier │────▶│ Membrain │
│         │     │        │     │         │     │          │
└─────────┘     └────┬───┘     └────┬────┘     └────┬─────┘
                     │              │               │
                     │              │               │
                ┌────▼───┐      ┌───▼────┐      ┌───▼────┐
                │        │      │        │      │        │
                │  Hit?  │─Yes─▶│ Return │      │        │
                │        │      │ Blocks │      │        │
                └────────┘      └────────┘      │        │
                     │                          │        │
                    No                          │        │
                     │                          │        │
                     ▼                          │        │
                ┌────────┐      ┌────────┐      │        │
                │        │      │        │      │        │
                │  Hit?  │─Yes─▶│ Copy to│─────▶│        │
                │        │      │  GPU   │      │        │
                └────────┘      └────────┘      │        │
                     │                          │        │
                    No                          │        │
                     │                          │        │
                     ▼                          │        │
                ┌────────┐      ┌────────┐      │        │
                │        │      │ Copy to│      │        │
                │  Hit?  │─Yes─▶│CPU & GPU│◀────┘        │
                │        │      │        │               │
                └────────┘      └────────┘               │
                     │                                   │
                    No                                   │
                     │                                   │
                     ▼                                   │
              ┌─────────────┐                            │
              │             │                            │
              │Compute Block│───────────────────────────▶│
              │             │                            │
              └─────────────┘                            │
```

### Caching Flow

```
┌─────────────┐                         ┌───────────────┐
│             │                         │               │
│New Computed │                         │ Policy        │
│   Blocks    │────┐              ┌────▶│ Manager       │
│             │    │              │     │               │
└─────────────┘    │              │     └───────┬───────┘
                   │              │             │
                   ▼              │             ▼
              ┌────────────┐      │      ┌─────────────┐
              │            │      │      │             │
              │ GPU Tier   │──────┘      │Should Cache?│
              │            │             │             │
              └────────────┘             └──────┬──────┘
                                                │
                   ┌───────────────────────┐    │
                   │                       ◀────┘
                   ▼                       │
              ┌────────────┐          ┌────▼────┐
              │            │    No    │         │
              │ CPU Tier   │◀─────────┤ Tier?   │
              │            │          │         │
              └───────┬────┘          └────┬────┘
                      │                    │
                     Yes                  Yes
                      │                    │
                      ▼                    ▼
              ┌────────────┐          ┌────────────┐
              │            │          │            │
              │   Store    │          │   Store    │
              │  in CPU    │          │in Membrain │
              │            │          │            │
              └────────────┘          └────────────┘
```

## Component Interaction

The key components and their interactions in the tiered caching system:

1. **MembrainKVCacheManager**
   - Top-level coordinator for all cache tiers
   - Handles block lookup, storage, and eviction across tiers
   - Delegates policy decisions to Policy Manager

2. **Block Pool (GPU Tier)**  
   - Fast, limited-capacity storage on GPU
   - Manages block allocation and GPU memory
   - Existing implementation in vLLM

3. **CPU Cache**
   - Medium-speed, medium-capacity storage
   - Serves as intermediary between GPU and remote storage
   - Typically uses memory-mapped storage or direct RAM

4. **Membrain Store**
   - Remote, high-capacity storage  
   - Handles serialization, network communication
   - Provides resilience against failures

5. **Policy Manager**
   - Makes caching and eviction decisions
   - Uses different policies for different tiers
   - Collects and analyzes usage statistics
