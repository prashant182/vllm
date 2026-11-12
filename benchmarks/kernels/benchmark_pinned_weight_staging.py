# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark `--pinned-weight-staging` on loader-shaped host-to-device copies.

Weight loaders copy slices of memory-mapped checkpoint files to the GPU with
`copy_weight_`. This times those copies from a file in the page cache, once as
plain pageable copies and once through `PinnedWeightStager`, for the slice
shapes the linear and MoE loaders produce, and checks that the staged copies
give the same bytes as the source.

Usage::

    python benchmarks/kernels/benchmark_pinned_weight_staging.py --threads 1 8

    # From a checkpoint shard instead of a generated file; read it into the
    # page cache first, as a repeated load would find it:
    cat model-00001-of-00030.safetensors > /dev/null
    python benchmarks/kernels/benchmark_pinned_weight_staging.py \
        --file model-00001-of-00030.safetensors
"""

import logging
import mmap
import os
import tempfile
import time
from collections.abc import Callable, Iterator
from contextlib import ExitStack

import torch
from tabulate import tabulate

from vllm.model_executor.model_loader.pinned_weight_staging import PinnedWeightStager
from vllm.model_executor.utils import copy_weight_, staging_weight_copies
from vllm.utils.argparse_utils import FlexibleArgumentParser

GB = 1e9
LARGE_ROWS, LARGE_COLS = 8192, 32768


def _write_file(directory: str, nbytes: int) -> str:
    """A file of random bytes, left in the page cache by the write."""
    block = 64 << 20
    with tempfile.NamedTemporaryFile(dir=directory, delete=False) as f:
        for start in range(0, nbytes, block):
            size = min(block, nbytes - start)
            f.write(torch.randint(0, 256, (size,), dtype=torch.uint8).numpy())
    return f.name


def _map_file(path: str) -> torch.Tensor:
    """The file as a pageable uint8 tensor, the way safetensors maps it."""
    with open(path, "rb") as f:
        mapped = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_COPY)
    return torch.frombuffer(memoryview(mapped), dtype=torch.uint8)


def _workloads(
    buf: torch.Tensor, total: int
) -> Iterator[tuple[str, list[torch.Tensor]]]:
    """Checkpoint slices shaped like the copies the weight loaders make."""
    piece = 4 << 20
    yield (
        "contiguous 4 MiB",
        [buf[i * piece : (i + 1) * piece] for i in range(total // piece)],
    )
    rows, cols = 2048, 1536
    size = rows * cols
    experts = [
        buf[i * size : (i + 1) * size].view(rows, cols) for i in range(total // size)
    ]
    # `_load_w13`: one TP2 rank's rows of an expert's gate or up projection.
    yield "w13-like row half", [x[: rows // 2] for x in experts]
    # `_load_w2`: one TP2 rank's columns of an expert's down projection.
    yield "w2-like column half (strided)", [x[:, : cols // 2] for x in experts]
    large = buf[: LARGE_ROWS * LARGE_COLS].view(LARGE_ROWS, LARGE_COLS)
    yield "large strided quarter", [large[:, LARGE_COLS // 4 : LARGE_COLS // 2]]


def _copy_all(
    dsts: list[torch.Tensor],
    srcs: list[torch.Tensor],
    stager: PinnedWeightStager | None,
) -> None:
    with ExitStack() as stack:
        if stager is not None:
            stack.enter_context(stager)
            stack.enter_context(staging_weight_copies(stager))
        for dst, src in zip(dsts, srcs):
            copy_weight_(dst, src)
    torch.accelerator.synchronize()


def _seconds(fn: Callable[..., object], *args: object) -> float:
    torch.accelerator.synchronize()
    start = time.perf_counter()
    fn(*args)
    torch.accelerator.synchronize()
    return time.perf_counter() - start


def _compare(
    srcs: list[torch.Tensor], threads: int, iters: int, device: torch.device
) -> tuple[float, float, bool]:
    """Best pageable and staged GB/s, and whether staging copied exact bytes.

    The two variants alternate which runs first in each iteration, so drift in
    machine state does not land on one of them. The first iteration warms up.
    """
    dsts = [torch.empty(src.shape, dtype=src.dtype, device=device) for src in srcs]
    nbytes = sum(src.nbytes for src in srcs)
    times: dict[bool, list[float]] = {False: [], True: []}
    for i in range(iters + 1):
        for staged in (False, True) if i % 2 else (True, False):
            stager = PinnedWeightStager(threads) if staged else None
            seconds = _seconds(_copy_all, dsts, srcs, stager)
            if i:
                times[staged].append(seconds)

    for dst in dsts:
        dst.zero_()
    stager = PinnedWeightStager(threads)
    _copy_all(dsts, srcs, stager)
    if stager.staged_bytes != nbytes:
        raise RuntimeError(
            f"Staged {stager.staged_bytes} of {nbytes} bytes; "
            "this workload does not measure staging."
        )
    exact = all(torch.equal(dst.cpu(), src) for dst, src in zip(dsts, srcs))
    return nbytes / min(times[False]) / GB, nbytes / min(times[True]) / GB, exact


def _reference_rates(
    buf: torch.Tensor, nbytes: int, iters: int, device: torch.device
) -> tuple[float, float]:
    """Best GB/s of pinned to GPU, and of page cache to pinned memory."""
    pinned = torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)
    dst = torch.empty(nbytes, dtype=torch.uint8, device=device)
    to_gpu = min(_seconds(dst.copy_, pinned, True) for _ in range(iters + 1))
    to_pinned = min(_seconds(pinned.copy_, buf[:nbytes]) for _ in range(iters + 1))
    return nbytes / to_gpu / GB, nbytes / to_pinned / GB


def main(args) -> None:
    logging.getLogger(PinnedWeightStager.__module__).setLevel(logging.WARNING)
    device = torch.accelerator.current_accelerator()
    if device is None or device.type != "cuda":
        raise RuntimeError("Pinned weight staging needs a CUDA or ROCm GPU.")

    path = args.file or _write_file(args.tmp_dir, args.size_gib << 30)
    buf = _map_file(path)
    if not args.file:
        # The mapping keeps the pages; the name is no longer needed.
        os.unlink(path)
    total = min(args.size_gib << 30, buf.numel())
    if total < LARGE_ROWS * LARGE_COLS:
        raise ValueError(f"needs at least {LARGE_ROWS * LARGE_COLS} bytes of file")
    print(f"source={path} bytes={total} iters={args.iters} (best of)")

    for threads in args.threads:
        torch.set_num_threads(threads)
        rows = []
        for name, srcs in _workloads(buf, total):
            pageable, staged, exact = _compare(srcs, threads, args.iters, device)
            rows.append([name, len(srcs), pageable, staged, staged / pageable, exact])
        print(f"\ntorch intra-op threads: {threads}")
        print(
            tabulate(
                rows,
                headers=[
                    "copy",
                    "copies",
                    "pageable GB/s",
                    "staged GB/s",
                    "speedup",
                    "exact",
                ],
                floatfmt=".2f",
            )
        )
        to_gpu, to_pinned = _reference_rates(
            buf, min(total, 1 << 30), args.iters, device
        )
        print(
            f"reference: pinned to GPU {to_gpu:.2f} GB/s, "
            f"page cache to pinned {to_pinned:.2f} GB/s"
        )


if __name__ == "__main__":
    parser = FlexibleArgumentParser()
    parser.add_argument(
        "--file",
        help="Read from this file (e.g. a checkpoint shard) instead of "
        "generating one. Read it into the page cache first.",
    )
    parser.add_argument(
        "--size-gib",
        type=int,
        default=2,
        help="Bytes to copy per workload, and the size of a generated file.",
    )
    parser.add_argument(
        "--tmp-dir",
        default=tempfile.gettempdir(),
        help="Directory for the generated file.",
    )
    parser.add_argument(
        "--threads",
        type=int,
        nargs="+",
        default=[1, 8],
        help="torch intra-op thread counts to sweep; also the stager's floor.",
    )
    parser.add_argument("--iters", type=int, default=3)
    main(parser.parse_args())
