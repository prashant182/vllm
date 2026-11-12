# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Copy checkpoint tensors to the GPU through pinned host memory.

Weight loaders copy slices of memory-mapped checkpoint tensors, which live in
pageable memory, to the GPU. A copy from pageable memory runs well below the
bandwidth of a copy from pinned memory, and a multithreaded CPU copy from the
page cache into pinned memory is faster than either, so staging through pinned
buffers shortens the load.
"""

import math
import threading
from collections.abc import Iterator
from contextlib import contextmanager

import torch

from vllm.config.load import LoadConfig
from vllm.logger import init_logger
from vllm.model_executor.utils import staging_weight_copies
from vllm.platforms import current_platform
from vllm.utils.platform_utils import is_pin_memory_available

logger = init_logger(__name__)

MIN_STAGED_BYTES = 1 << 20
BUFFER_BYTES = 64 << 20


class PinnedWeightStager:
    """Copies large pageable CPU tensors to the GPU through two pinned buffers.

    Inside `staging_weight_copies(stager)`, `copy_weight_` offers each copy to
    `stager.copy_`, which makes it and returns True when `dst` is a CUDA tensor,
    `src` is a CPU tensor that is not pinned and holds at least
    MIN_STAGED_BYTES, the shapes are equal, one dim-0 row fits a buffer, the
    caller is the thread that entered the stager, neither tensor overrides
    `__torch_function__`, and no TorchDispatchMode is active (layerwise loading
    counts copied elements with one, and staging adds copies). For any other
    copy it returns False and the caller copies as usual.

    A staged copy is split along dim 0 into chunks that fit a buffer. The CPU
    gathers a chunk (any strides, converted to `dst.dtype` on the CPU as a
    blocking `copy_` does) into one buffer while the previous chunk is copied
    from the other with `non_blocking=True` on the current stream of `dst`'s
    device, so later work on that stream is ordered after it. A non-contiguous
    `dst` therefore needs a device temporary of at most one chunk. Each buffer's
    event is waited on before the buffer is refilled, and exiting waits for
    both. The gather runs on torch's intra-op threads, and with one thread it is
    about as slow as the pageable copy it replaces, so the stager raises them to
    at least `num_threads` while active and restores them on exit.
    """

    def __init__(self, num_threads: int, buffer_bytes: int = BUFFER_BYTES) -> None:
        self.num_threads = num_threads
        self.buffer_bytes = buffer_bytes
        self.staged_bytes = 0
        self._owner: int | None = None
        self._saved_num_threads = 0
        self._device: torch.device | None = None
        self._buffers: list[torch.Tensor] = []
        self._events: list[torch.Event] = []
        self._pending = [False, False]
        self._next = 0

    def __enter__(self) -> "PinnedWeightStager":
        self._owner = threading.get_ident()
        self._saved_num_threads = torch.get_num_threads()
        if self.num_threads > self._saved_num_threads:
            torch.set_num_threads(self.num_threads)
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        try:
            for event, pending in zip(self._events, self._pending):
                if pending:
                    event.synchronize()
        finally:
            torch.set_num_threads(self._saved_num_threads)
            self._owner = None
            self._buffers = []
            self._events = []
            self._pending = [False, False]
        if self.staged_bytes:
            logger.info(
                "Copied %.2f GiB of weights to the GPU through pinned memory",
                self.staged_bytes / (1 << 30),
            )

    def copy_(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        """Stage `dst.copy_(src)` and return True, or return False untouched."""
        if not self.accepts(dst, src):
            return False
        if not self._buffers:
            self._device = dst.device
            self._buffers = [
                torch.empty(
                    self.buffer_bytes, dtype=torch.uint8, device="cpu", pin_memory=True
                )
                for _ in range(2)
            ]
            self._events = [torch.Event(device=dst.device) for _ in range(2)]
        row_bytes = math.prod(dst.shape[1:]) * dst.element_size()
        rows_per_chunk = self.buffer_bytes // row_bytes
        stream = torch.accelerator.current_stream(dst.device)
        num_rows = src.shape[0]
        for start in range(0, num_rows, rows_per_chunk):
            k = self._next
            self._next ^= 1
            if self._pending[k]:
                self._events[k].synchronize()
            rows = min(rows_per_chunk, num_rows - start)
            src_chunk = src.narrow(0, start, rows)
            staged = self._buffers[k][: rows * row_bytes]
            staged = staged.view(dst.dtype).view(src_chunk.shape)
            staged.copy_(src_chunk)
            dst.narrow(0, start, rows).copy_(staged, non_blocking=True)
            self._events[k].record(stream)
            self._pending[k] = True
        self.staged_bytes += src.nbytes
        return True

    def accepts(self, dst: torch.Tensor, src: torch.Tensor) -> bool:
        """Whether `copy_(dst, src)` would stage the copy."""
        return (
            threading.get_ident() == self._owner
            and dst.is_cuda
            and src.device.type == "cpu"
            and dst.shape == src.shape
            and src.nbytes >= MIN_STAGED_BYTES
            and math.prod(dst.shape[1:]) * dst.element_size() <= self.buffer_bytes
            and (self._device is None or dst.device == self._device)
            and _has_default_torch_function(dst)
            and _has_default_torch_function(src)
            and torch._C._len_torch_dispatch_stack() == 0
            and not src.is_pinned()
        )


def _has_default_torch_function(tensor: torch.Tensor) -> bool:
    """A Tensor, or a Parameter (subclass) that keeps Parameter's disabled
    `__torch_function__`."""
    return (
        type(tensor) is torch.Tensor
        or type(tensor).__torch_function__ is torch._C._disabled_torch_function_impl
    )


@contextmanager
def maybe_stage_weight_copies(load_config: LoadConfig) -> Iterator[None]:
    """Stage `copy_weight_` copies through pinned memory while the context is
    active, if `load_config.pinned_weight_staging` is set (CUDA and ROCm)."""
    if not (
        load_config.pinned_weight_staging
        and current_platform.is_cuda_alike()
        and is_pin_memory_available()
    ):
        yield
        return
    with (
        PinnedWeightStager(load_config.pinned_weight_staging_threads) as stager,
        staging_weight_copies(stager),
    ):
        yield
