# Inspecting kernels: PTX, SASS, ncu, Godbolt

Everything here was run on this machine (RTX 3050 Laptop, sm_86, CUDA 13.1)
against kernel 3 (`src/kernels/3_shared_mem.cuh`), except `ncu`, which needs
sudo (see below).

## The compilation pipeline

```
kernel.cu ──nvcc front end──> PTX ──ptxas──> SASS (inside a .cubin)
                              │                │
                virtual ISA, text,       the real machine code
                same for every GPU       the GPU executes (sm_86-specific)
```

- **PTX**: portable virtual assembly. Infinite virtual registers, no scheduling decisions yet.
- **SASS**: what really runs. Real registers, real instruction order, vectorized loads.
  Optimizations that matter for performance (unrolling, `LDS.128`, FMA scheduling) show up here.
- A release build embeds **both**: `cuobjdump --list-elf/--list-ptx build/release/sgemm`
  lists `main.sm_86.cubin` and `main.sm_86.ptx`.

## Commands (all tested)

| Question | Command |
|---|---|
| How many registers / how much shared memory? Any spills? | `cuobjdump --dump-resource-usage build/release/sgemm` (or compile with `-Xptxas -v`) |
| Show the SASS | `cuobjdump -sass build/release/sgemm` (restrict to one kernel: `-fun <mangled name>`) |
| Show the PTX | `cuobjdump -ptx build/release/sgemm` |
| Decode a mangled kernel name | `cu++filt _Z22sgemm_shared_mem_blockILi32EEviiifPKfS1_fPf` |
| SASS interleaved with your source lines | `cuobjdump -xelf all build/release/sgemm` then `nvdisasm -gi main.sm_86.cubin` |
| Only list instruction counts | pipe SASS through `grep`/`sort`/`uniq -c` (see below) |

Notes:
- Use the **release** build (`-O3 -lineinfo`). The debug build uses `-G`, which disables optimization, so its SASS is not what you benchmark.
- `nvdisasm -fun` takes a symbol *index*, not a name, and then ignores `-g`/`-gi`. Disassemble the whole cubin instead.
- `nvdisasm`'s `//## File ... line N` markers need `-lineinfo`, which the release preset has.

## What kernel 3 looks like (real output)

```
Used 37 registers, 8192 bytes smem, 0 bytes spill stores / loads
```
8192 bytes = two 32×32 float tiles (`As`, `Bs`) × 4096 bytes each.

Instruction counts in the SASS of the whole kernel:

| SASS | Count | What it is |
|---|---|---|
| `LDG` | 3 | loads from global memory (tile of A, tile of B, original C) |
| `STS` | 2 | stores into shared memory (the two tile elements) |
| `BAR.SYNC` | 2 | the two `__syncthreads()` |
| `LDS` | 32 | shared-memory loads of `Bs` (stride 32 floats, can't be vectorized) |
| `LDS.128` | 8 | shared-memory loads of `As`, 4 floats at a time |
| `FFMA` | 33 | 32 for the unrolled inner `dotIdx` loop + 1 for `alpha * tmp + beta * C` |

**Reading it:**
- The compiler **fully unrolled** the 32-iteration inner loop: 32 `FFMA` in a row.
- In PTX (before `ptxas`) there are **64** `ld.shared.f32` for 32 `fma.rn.f32`: 2 loads per FMA, exactly as the source reads.
  `ptxas` merged the 32 `As` loads into 8 `LDS.128`, but couldn't merge the `Bs` loads because
  `Bs[dotIdx * 32 + col]` jumps 128 bytes per step (`[R20.X4+0x1000]`, `+0x1080`, `+0x1100`, ...).
- Still: **40 shared-memory load instructions for 32 FMAs**. Shared memory moves about 128 bytes per clock per SM,
  while the FP32 units can do 128 FMAs per clock per SM. A load/FMA ratio above ~0.25 means the
  kernel waits on shared memory, not on math. That is the "MIO throttle" below.

## Nsight Compute (`ncu`): why is the kernel slow?

Needs access to GPU counters: `sudo (which ncu) ...` in fish (sudo has its own PATH, so the plain
name isn't found). Reports written with sudo are owned by root.

```fish
# one-shot text report for just kernel 3 (not cuBLAS's kernel)
sudo (which ncu) --section SpeedOfLight --section WarpStateStats --section InstructionStats \
     --section MemoryWorkloadAnalysis --kernel-name regex:shared_mem ./build/release/sgemm

# everything, saved for the GUI
sudo (which ncu) --set full --kernel-name regex:shared_mem -o k3 ./build/release/sgemm
ncu-ui k3.ncu-rep
```

Section names (from `ncu --list-sections`): `SpeedOfLight`, `WarpStateStats`, `SchedulerStats`,
`InstructionStats`, `MemoryWorkloadAnalysis`, `Occupancy`, `LaunchStats`, `SourceCounters`.

**Use a big problem for profiling.** The harness currently uses 128×256×64: far too small to fill the
GPU (16 SMs), so the numbers are meaningless. Temporarily set `M = N = K = 1024` (all multiples of 32).
The CPU reference check gets slow above ~2048.

### Walkthrough: profile kernel 3 and open it in `ncu-ui`

```
cmake --preset release && cmake --build --preset release
scripts/profile.sh sgemm_shared 1024 k3     # asks for your sudo password
ncu-ui build/k3.ncu-rep
```

`scripts/profile.sh` runs `sudo ncu --set full ...` on `./sgemm 1024` for the first kernel whose name
matches `sgemm_shared.*` (so cuBLAS's kernel is skipped), then gives the report file back to you.
The harness accepts one optional argument, a square size: `./build/release/sgemm 1024`.

Facts measured on this machine, to compare with the report:

- 37 registers per thread and 8192 bytes of shared memory per block.
- 1 active block per SM (Streaming Multiprocessor): 1024 threads of a possible 1536, so **66.7% theoretical occupancy**.
  The register file (65,536 registers per SM) fits only one 1024-thread block.
  With one block per SM, there is no other block to run while this one waits at `__syncthreads()`
  (the BAR.SYNC instruction, BARrier SYNChronization), so a **Barrier** stall is a likely suspect. This is a prediction; check it.

Pages in `ncu-ui` (names from memory of the GUI; I could not open it from here):

1. **Summary / Details**: start with **GPU Speed Of Light Throughput**: how close memory and compute are to their peak.
2. **Details, Launch Statistics and Occupancy**: grid size, registers, shared memory, theoretical vs achieved occupancy.
3. **Details, Warp State Statistics**: the stall reasons. The tallest bar is your bottleneck.
4. **Details, Memory Workload Analysis**: the memory chart shows traffic between global memory, L2 cache, L1 cache and shared memory.
5. **Source page**: the **View** dropdown selects SASS (Streaming ASSembler, the real GPU machine code), PTX (Parallel Thread Execution,
   the readable virtual assembly) or your CUDA source, alone or side by side (e.g. source on the left, PTX on the right).
   Clicking a line on one side highlights the matching lines on the other side. Each instruction shows how many warp-stall samples it collected.
   Find the shared-memory loads (LDS, LoaD from Shared memory) and see what the warps wait for.
   This build embeds PTX (`nvdisasm -gp` shows `.nv_debug_ptx_txt`), so the PTX view should be available.
   Nsight Compute shows no PTX when code is compiled with separate linking (`-rdc`); this project doesn't use it.
   PTX is easier to read, but stalls happen on SASS, so use PTX to understand and SASS to confirm.
6. The **OPT** (optimization) messages in Details are hints written by NVIDIA's rules. Treat them as ideas, not orders.

`ncu-ui` can also compare reports: open kernel 3's report, then add kernel 4's as a **baseline** to see every metric side by side.

### Reading SASS: the five instructions in kernel 3

| Instruction | Full form | Meaning |
|---|---|---|
| `LDG.E R17, [R2.64]` | LoaD from Global memory | `R17 = memory[address in R2]`. `R` = a per-thread register; `.E` = 64-bit address |
| `STS [R6.X4], R17` | STore to Shared memory | `shared[R6 * 4] = R17` (`.X4` = multiply the index by 4 bytes, one float) |
| `BAR.SYNC` | BARrier SYNChronization | `__syncthreads()`: wait until every thread of the block arrives |
| `LDS R24, [R20.X4+0x1000]` | LoaD from Shared memory | read one float. Offset `0x1000` = 4096 bytes = start of `Bs` (after `As`) |
| `LDS.128 R12, [R0]` | LoaD from Shared memory, 128 bits | read 4 floats at once into R12..R15 |
| `FFMA Rd, Ra, Rb, Rc` | Float Fused Multiply-Add | `Rd = Ra * Rb + Rc`, i.e. `tmp += As * Bs` |

The long run of `LDS` and `FFMA` is the unrolled inner loop. Everything else is setup.

### Warp State Statistics: the stall reasons

Every cycle, a warp that can't issue is "stalled" for a reason. The biggest bar tells you the bottleneck:

| Stall reason | Meaning | Typical fix |
|---|---|---|
| **Long Scoreboard** | waiting for a global-memory (or local) load | coalescing (kernel 2), more reuse, tiling |
| **MIO Throttle** | the queue of memory-I/O instructions (**shared memory**, special math) is full | fewer / wider shared-memory loads per FMA: more work per thread (kernels 4, 5), `float4` |
| **Short Scoreboard** | waiting for a shared-memory result or special function | reorder, more independent work |
| **Barrier** | waiting at `__syncthreads()` | less imbalance, double buffering |
| **LG Throttle** | global-memory queue full | fewer, wider global loads |
| **Math Pipe Throttle** | the FMA pipe is busy | you're compute-bound: the goal |

### The article's reasoning for kernel 4

1. Kernel 3 reads all its data from shared memory, so global-memory stalls are gone.
2. The profile shows **MIO Throttle** as the dominant stall: too many shared-memory loads per FMA (above: 40 per 32).
3. Fix: let each thread compute `TM = 8` outputs. For each `dotIdx` it loads **one** `Bs` value into a
   register and reuses it for 8 FMAs (each with its own `As` value): `(1 + TM) / TM = 1.125` loads per FMA,
   instead of 2 (source level).
4. Kernel 5 (2D tiling) goes further: `(TM + TN) / (TM * TN) = 16 / 64 = 0.25` loads per FMA.

**A good habit:** before profiling a new kernel, count `LDS` vs `FFMA` in its SASS and predict
which stall will drop. Then confirm with `ncu`.

### Source page in `ncu-ui`

Open the report, choose the **Source** page, switch the view to **SASS** (or SASS + source).
Each instruction shows how many warp-stall samples it received. You can see which `LDS` the
warps are waiting at. This relies on `-lineinfo`.

## Godbolt (Compiler Explorer)

Good for quick experiments without building the project: change a line, see the PTX/SASS change.

- Go to godbolt.org, set the language to **CUDA C++**, pick an **NVCC 13.x** compiler
  (the API lists NVCC 13.1.0, 13.1.1 up to 13.3.0, and reports device-assembly support for them).
- Compiler options: `-arch=sm_86 -O3 -std=c++20`
- Paste only the kernel plus `#include <cuda_runtime.h>`. You don't need `main` or cuBLAS to look at the assembly.
- Look for the **device code** (PTX / SASS) view. I couldn't open the web UI from here, so the exact menu
  names may differ from this note.
- It compiles for whatever `-arch` you give it, so keep `sm_86` to match your GPU.

## Other tools worth knowing

| Tool | Use |
|---|---|
| `cuda-gdb` / Nsight VS Code extension | step through a kernel, inspect threads (`.vscode/launch.json`) |
| `compute-sanitizer --tool memcheck/racecheck/synccheck/initcheck` | memory bugs, shared-memory races, bad `__syncthreads()` |
| `nsys profile` / `nsys-ui` | timeline of the whole program: copies, launches, gaps |
| `ncu-ui` | GUI for `.ncu-rep` reports, roofline chart, source page |
| `nvidia-smi dmon -s pc` | power, temperature, clocks while a benchmark runs; shows laptop throttling |
| VS Code command palette (Nsight extension) | "Change CUDA debug focus", "Generate .clangd", "Configure CUDA C++ Language Support" |

## Suggested loop for each new kernel

1. Build release. Check it against cuBLAS and run `compute-sanitizer`.
2. `cuobjdump --dump-resource-usage`: registers, shared memory, spills.
3. Count `LDG` / `LDS` / `FFMA` in the SASS.
4. `ncu` on a large size: speed of light, then the stall reasons.
5. Write the result down (GFLOPS, top stall, registers), then decide what the next kernel should fix.
