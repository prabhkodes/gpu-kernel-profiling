# gpu-kernel-profiling

Reading Nsight Systems traces to find out why a CUDA kernel is slow — and turning the answer into a
code change. Three kernels that move identical amounts of memory, profiled on an A100 at two block
sizes, plus a tool that extracts the numbers from the trace databases instead of clicking through the
GUI.

**Stack:** CUDA · Nsight Systems · NVTX · SQLite · Python · SLURM

**Device:** NVIDIA A100-SXM-64GB, compute 8.0, 124 SMs, 2.039 TB/s peak HBM2e

**Workload:** 1000 × 1000 `int32` matrix (4 MB). Every kernel reads it once and writes it once — 8 MB
of traffic — so time is directly comparable and any difference is *access pattern*, not work done.

**What the traces showed**

- **The naive transpose reaches 11% of peak bandwidth; a plain copy reaches 60%.** Same bytes moved.
- **Shared-memory tiling recovers almost all of it — 97% of copy speed** — but only at the right block size.
- **Block size mattered more than the algorithm.** The same shared kernel is 1.85× slower at 32×32 than at 16×16, and the cause is a bank-conflict pattern visible in the source.
- **Two of the four configurations never ran at all.** The traces record API calls and zero kernels.
- **Host↔device transfers cost 100× the kernel.** 4 MB each way at ~12 GB/s against a 6.6 µs kernel.

## Results

Best of two runs per kernel, from [`transpose/traces/`](transpose/traces/).

| Block | Kernel | Time | Achieved | % of peak | % of copy |
|---|---|---:|---:|---:|---:|
| 16×16 | copy (reference) | 6.78 µs | 1179 GB/s | 57.8% | 100% |
| 16×16 | transpose, naive | 12.22 µs | 654 GB/s | 32.1% | 55% |
| 16×16 | **transpose, shared** | **6.98 µs** | **1147 GB/s** | 56.2% | **97%** |
| 32×32 | copy (reference) | 6.56 µs | 1220 GB/s | 59.8% | 100% |
| 32×32 | transpose, naive | 34.50 µs | 232 GB/s | 11.4% | 19% |
| 32×32 | transpose, shared | 12.93 µs | 619 GB/s | 30.3% | 51% |

The copy kernel is the control. It does the same reads and writes with a perfectly coalesced pattern,
so it sets the practical ceiling — **60% of theoretical peak**, not 100%, which is the honest number to
measure against.

→ **A transpose is not intrinsically slower than a copy.** Done right it lands within 3% of it.

## Why naive is slow

```cuda
out[col * height + row] = in[row * width + col];
```

- Consecutive threads have consecutive `col`
- **Reads** are contiguous → coalesced into few transactions
- **Writes** are strided by `height` → each thread lands in a different cache line
- One warp's write becomes 32 separate memory transactions instead of a handful

That is the whole 5.3× gap at 32×32.

## Why 32×32 is slower than 16×16 — even with shared memory

The tile is declared unpadded:

```cuda
__shared__ int shTile[TILE_DIM][TILE_DIM];
...
out[idx] = shTile[threadIdx.x][threadIdx.y];    // column-wise read
```

Shared memory has 32 banks of 4 bytes. On that column-wise read, consecutive threads in a warp are
`TILE_DIM` ints apart:

| TILE_DIM | Offset between threads | Bank = offset mod 32 | Conflict |
|---:|---:|---|---|
| 32 | 32 ints | identical for every thread | **32-way — fully serialised** |
| 16 | 16 ints | alternates between two banks | 2-way |

→ At 32×32 every thread in the warp hits the **same bank**, so the read serialises 32 ways. At 16×16 it
only 2-ways. That is the 1.85× difference, and it is predictable from the declaration alone.

**The fix** is one character — pad the tile so the stride becomes coprime with the bank count:

```cuda
__shared__ int shTile[TILE_DIM][TILE_DIM + 1];
```

> Not yet measured. The prediction is that 32×32 shared should land near copy speed, as 16×16 already
> does. Verifying it needs GPU time.

## Two configurations that never ran

`TILE_DIM` 64 and 128 were profiled too. Both traces contain **15 runtime API calls and zero kernels**:

```
## mat_transpose_report_64
   no kernels recorded, 15 runtime API calls -> launches failed
```

64 × 64 = 4096 threads per block, and 128 × 128 = 16384 — both above CUDA's hard limit of **1024
threads per block**. The launches returned an error that the program never checked, so it exited
cleanly having computed nothing.

→ **A run that produces no output is not obviously a failed run.** The profiler is what makes it
obvious. The published source now catches it at compile time instead:

```cuda
static_assert(TILE_DIM * TILE_DIM <= 1024,
              "TILE_DIM*TILE_DIM exceeds the 1024 threads-per-block limit");
```

## Transfers dominate

| Direction | Size | Time | Achieved |
|---|---:|---:|---:|
| Host → device | 4 MB | 330 µs | 12.1 GB/s |
| Device → host | 4 MB | 326 µs | 12.3 GB/s |
| Best kernel | — | 6.6 µs | 1220 GB/s |

→ **The transfers cost 100× the compute.** At this size the choice of kernel is irrelevant to
end-to-end time; keeping data resident on the device is the only optimisation that matters. It becomes
worth tuning the kernel only when the data already lives on the GPU across many iterations — which is
exactly the case in the [Jacobi](https://github.com/prabhkodes/jacobi-poisson-solver) and
[miniWeather](https://github.com/prabhkodes/miniWeather-mpi-openacc) solvers.

## The tool

[`analysis/nsys_kernels.py`](analysis/nsys_kernels.py) reads Nsight's SQLite exports and prints one row
per kernel launch with achieved bandwidth, so runs can be diffed without opening each trace in the GUI.

```bash
./analysis/nsys_kernels.py transpose/traces/*.sqlite \
    --bytes-moved 8000000 --peak-gbs 2039 --csv analysis/results/kernel_summary.csv
```

- Joins `CUPTI_ACTIVITY_KIND_KERNEL` against `StringIds` for demangled names
- Reports grid and block dimensions as launched, not as intended
- Reports `CUPTI_ACTIVITY_KIND_MEMCPY` with direction and achieved rate
- **Treats "zero kernels, non-zero API calls" as a finding**, not as an empty file

Output is committed at [`analysis/results/kernel_summary.csv`](analysis/results/kernel_summary.csv).

## Reproducing

```bash
# build — TILE_DIM is a compile-time constant because it sizes the shared tile
nvc++ -O3 -DTILE_DIM=16 transpose/src/transpose.cu -o transpose.x
./transpose.x 1000                      # matrix edge length

# profile
nsys profile --trace=cuda,nvtx --stats=true -o report_16 ./transpose.x 1000
nsys export --type sqlite report_16.nsys-rep
```

[`transpose/slurm/profile.sh`](transpose/slurm/profile.sh) is the Leonardo batch script.

## Layout

```
transpose/
  src/transpose.cu       copy, naive transpose, shared-memory transpose
  slurm/profile.sh       SLURM batch script
  traces/                .nsys-rep and .sqlite for TILE_DIM 16, 32, 64, 128
analysis/
  nsys_kernels.py        trace database → table
  results/               extracted CSV
kernels/                 the standalone kernels these were built from
  array_reverse.cu  mat_copy.cu  mat_transpose_shared.cu  mat_transpose_unshared.cu
```

## Caveats

| Caveat | Detail |
|---|---|
| **Two runs per kernel, not a statistical sample** | The first is consistently slower (cold caches). Tables use the second |
| **The padding fix is untested** | Diagnosed from the source and the bank arithmetic, not measured |
| **`int32`, not `float`/`double`** | Bank behaviour is the same at 4 bytes; absolute bandwidth would differ for fp64 |
| **Source was traced at N = 1000** | The committed source now defaults to 1000 to match. The original had been shrunk to 10 for debugging after the traces were taken |
| **Nsight Systems, not Compute** | `nsys` gives timeline and bandwidth. Bank-conflict *counters* would need `ncu`, which would confirm the diagnosis directly |

## Where this came from

| | |
|---|---|
| Course | *P1.7 — GPU Programming*, MHPC, ICTP / SISSA Trieste |
| Traced | November 2025, Leonardo Booster |
| Related | [`jacobi-poisson-solver`](https://github.com/prabhkodes/jacobi-poisson-solver), [`miniWeather-mpi-openacc`](https://github.com/prabhkodes/miniWeather-mpi-openacc) and [`fft-gpu-programming-models`](https://github.com/prabhkodes/fft-gpu-programming-models) carry the same profiling approach into full applications |
| Course repository | Belongs to SISSA, private |
