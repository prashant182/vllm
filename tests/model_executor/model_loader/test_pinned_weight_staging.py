# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import hashlib
import logging
import threading
from contextlib import contextmanager

import pytest
import torch

from vllm.model_executor.model_loader.pinned_weight_staging import (
    MIN_STAGED_BYTES,
    PinnedWeightStager,
)
from vllm.model_executor.model_loader.reload.meta import CopyCounter
from vllm.model_executor.utils import copy_weight_, staging_weight_copies
from vllm.platforms import current_platform

requires_gpu = pytest.mark.skipif(
    not current_platform.is_cuda_alike(), reason="needs a CUDA or ROCm GPU"
)


@contextmanager
def _staging(num_threads: int = 1):
    # A 1 MiB buffer splits every staged copy below into several chunks, so
    # both buffers are refilled behind their events.
    stager = PinnedWeightStager(num_threads, buffer_bytes=MIN_STAGED_BYTES)
    with stager, staging_weight_copies(stager):
        yield stager


def _checkpoint(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.randn(*shape, generator=torch.Generator().manual_seed(0)).to(dtype)


def _gpu(*shape: int, dtype: torch.dtype = torch.bfloat16) -> torch.Tensor:
    return torch.zeros(*shape, dtype=dtype, device="cuda")


def _row_half():
    """`_load_w13`: a TP rank's rows into one half of the fused w13."""
    w13 = _gpu(2048, 1024)
    return w13.narrow(0, 1024, 1024), _checkpoint(2048, 1024).narrow(0, 1024, 1024)


def _strided_source():
    """`_load_w2`: a row-parallel shard is a dim-1 slice of the checkpoint."""
    return _gpu(1024, 512), _checkpoint(1024, 2048).narrow(1, 512, 512)


def _padded_destination():
    """A destination narrowed for padding is not contiguous."""
    return _gpu(1024, 1536).narrow(1, 0, 1024), _checkpoint(1024, 1024)


STAGED_CASES = {
    "contiguous": lambda: (_gpu(2048, 1024), _checkpoint(2048, 1024)),
    "row_half": _row_half,
    "strided_source": _strided_source,
    "padded_destination": _padded_destination,
    "dtype_conversion": lambda: (
        _gpu(1024, 1024),
        _checkpoint(1024, 1024, dtype=torch.float32),
    ),
    "one_dim": lambda: (_gpu(1 << 20), _checkpoint(1 << 20)),
}

PASSTHROUGH_CASES = {
    "below_threshold": lambda: (_gpu(256, 1024), _checkpoint(256, 1024)),
    "pinned_source": lambda: (
        _gpu(2048, 1024),
        _checkpoint(2048, 1024).pin_memory(),
    ),
    "broadcast": lambda: (_gpu(2, 512, 1024), _checkpoint(512, 1024)),
    "row_larger_than_buffer": lambda: (_gpu(2, 1 << 20), _checkpoint(2, 1 << 20)),
}


@pytest.mark.cpu_test
def test_threads_are_raised_only_while_staging():
    before = torch.get_num_threads()
    src = _checkpoint(2048, 1024)
    dst = torch.zeros_like(src)
    with _staging(num_threads=before + 1) as stager:
        assert torch.get_num_threads() == before + 1
        copy_weight_(dst, src)
    assert torch.get_num_threads() == before
    # A copy to the CPU is not staged.
    assert stager.staged_bytes == 0
    assert torch.equal(dst, src)


@requires_gpu
@pytest.mark.parametrize(
    "case,staged",
    [(name, True) for name in STAGED_CASES]
    + [(name, False) for name in PASSTHROUGH_CASES],
)
def test_staged_copy_matches_plain_copy(case: str, staged: bool):
    make = {**STAGED_CASES, **PASSTHROUGH_CASES}[case]
    expected_dst, src = make()
    expected_dst.copy_(src)
    dst, src = make()
    with _staging() as stager:
        copy_weight_(dst, src)
    assert stager.staged_bytes == (src.nbytes if staged else 0)
    assert torch.equal(dst, expected_dst)


@requires_gpu
def test_copies_on_other_threads_are_not_staged():
    dst, src = STAGED_CASES["contiguous"]()
    with _staging() as stager:
        worker = threading.Thread(target=copy_weight_, args=(dst, src))
        worker.start()
        worker.join()
    assert stager.staged_bytes == 0
    assert torch.equal(dst.cpu(), src)


@requires_gpu
def test_copies_under_a_dispatch_mode_are_not_staged():
    """Layerwise loading counts copied elements with CopyCounter; staging
    would add copies to the count."""
    dst, src = STAGED_CASES["contiguous"]()
    with _staging() as stager, CopyCounter() as counter:
        copy_weight_(dst, src)
    assert stager.staged_bytes == 0
    assert counter.copied_numel == src.numel()
    assert torch.equal(dst.cpu(), src)


def _weight_digests(worker) -> dict[str, str]:
    return {
        name: hashlib.sha256(
            tensor.detach().reshape(-1).view(torch.uint8).cpu().numpy()
        ).hexdigest()
        for name, tensor in worker.get_model().state_dict().items()
    }


@requires_gpu
def test_engine_loads_the_same_weights(
    vllm_runner, caplog_mp_spawn, monkeypatch: pytest.MonkeyPatch
):
    # Spawned workers let caplog_mp_spawn read their logs.
    monkeypatch.setenv("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    monkeypatch.setenv("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    digests = {}
    for pinned_weight_staging in (False, True):
        with (
            caplog_mp_spawn(logging.INFO) as log_holder,
            vllm_runner(
                "Qwen/Qwen3-0.6B",
                pinned_weight_staging=pinned_weight_staging,
                max_model_len=16,
                max_num_seqs=1,
                enforce_eager=True,
            ) as llm,
        ):
            digests[pinned_weight_staging] = llm.collective_rpc(_weight_digests)
        staged = "to the GPU through pinned memory" in log_holder.text
        assert staged == pinned_weight_staging

    assert digests[True] == digests[False]
