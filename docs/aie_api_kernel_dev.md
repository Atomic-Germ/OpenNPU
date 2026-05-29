# AIE API — Kernel Development Reference for NPU2

Source: AMD AIE API User Guide (UG1529) 2025.1  
https://xilinx.github.io/aie_api/topics.html

This document extracts and annotates the API sections directly relevant to
implementing a Q4NX-compatible GEMM kernel on XDNA 2 (NPU2). Cross-references
to the binary analysis in `llama_npu2_kernel_analysis.md` are noted where
applicable.

---

## Toolchain Position

```
C++ kernel source (using aie::mmul, aie::lut, etc.)
        │
        ▼
    Peano (llvm-aie)           ← compiles to VLIW machine code
        │
        ▼
  VLIW object code             ← 62KB seen in Group A BLOCKWRITE
        │
        ▼
  MLIR-AIE / xclbinutil        ← packages into CDO + xclbin
        │
        ▼
    layer.xclbin
```

The AIE API is a header-only C++ template library. It is architecture-tagged
(`aie::arch::XDNA2`) so the same source compiles correctly for each tile
generation. Peano resolves the templates to native VLIW instructions for the
target architecture.

---

## XDNA 2 Specifics

### Vector element types

| Type      | Native sizes (element count)         | Notes                     |
|-----------|--------------------------------------|---------------------------|
| `int4`    | 32/64/128/256                        |                           |
| `uint4`   | 32/64/128/256                        |                           |
| `int8`    | 16/32/64/128                         |                           |
| `uint8`   | 16/32/64/128                         |                           |
| `int16`   | 8/16/32/64                           |                           |
| `uint16`  | 8/16/32/64                           |                           |
| `int32`   | 4/8/16/32                            |                           |
| `uint32`  | 4/8/16/32                            |                           |
| `bfloat16`| 8/16/32/64                           | Primary FP type on NPU2   |
| `float`   | 4/8/16/32                            |                           |
| `bfp16ebs8`/`bfp16ebs16` | 32/64/128/256 | Block floating-point; XDNA 2 only |

`cbfloat16` and `cfloat` are **not** supported on XDNA 2 (dropped vs XDNA 1).

### Accumulator types on XDNA 2

| Tag         | Lanes               | Native accumulation |
|-------------|---------------------|---------------------|
| `acc32`     | 8/16/32/64/128      | 32b                 |
| `acc40–64`  | 4/8/16/32/64        | 64b                 |
| `accfloat`  | 4/8/16/32/64/128    | 32b                 |
| `cacc32–64` | 2/4/8/16/32         | 64b                 |

`acc72`, `acc80`, `cacc72`, `cacc80`, `caccfloat` are **absent** on XDNA 2.
Use `acc64` for integer chains and `accfloat` for BF16 output.

### Memory alignment

XDNA 2 supports 128b, 256b, and **512b** vector load/store, vs 256b max on
XDNA 1. This is the widest access tier and directly corresponds to the 64-byte
(512-bit) DMA BD granularity observed in the Group D topology.

Use `alignas(aie::vector_decl_align)` for all static tile DM buffers.
Minimum alignment that covers all vector sizes:

```cpp
alignas(aie::vector_decl_align) static bfloat16 weight_buf[BUF_ELEMS];
```

---

## Matrix Multiplication — `aie::mmul`

```cpp
template<unsigned M, unsigned K, unsigned N,
         ElemBaseType TypeA, ElemBaseType TypeB = TypeA,
         AccumElemBaseType AccumTag = accauto>
struct aie::mmul;
```

An `mmul` object holds the running accumulator for a tiled C = A × B.

- `.mul(a, b)` — initialize result from first tile  
- `.mac(a, b)` — accumulate subsequent tiles  
- `.to_vector<T>(shift)` — downshift-round-saturate to output type  
- `.to_accum()` — pass accumulator to next stage without converting

Input vectors: `a` is `M×K` elements, `b` is `K×N` elements, both row-major.

### XDNA 2 supported shapes (dense, real types)

| TypeA × TypeB          | Supported MxKxN shapes                                              |
|------------------------|----------------------------------------------------------------------|
| `int8 × int8`          | 4x8x8, 8x8x8                                                        |
| `int16 × int8`         | 4x4x8, 8x4x8, 4x8x8, 2x8x8ᵇ                                       |
| `int8 × int16`         | 8x2x8ᵇ, 4x4x8ᵇ                                                     |
| `int16 × int16`        | 4x2x8, 8x2x8, 2x4x8, 4x4x8, 8x1x8ᵇ                               |
| `int32 × int16`        | 4x2x8, 2x4x8ᵃᵇ, 4x4x8ᵃᵇ, 4x1x8ᵇ                                  |
| `bfloat16 × bfloat16`  | **8x8x4**ᵃᵇ, **4x8x8**ᵃᵇᶜ, **4x8x4**ᵃᵇ, 8x8x8ᵉ, 8x1x8ᵇ          |
| `int8 × int4`          | 4x16x16                                                              |

Notes:  
ᵃ Emulated using multiple intrinsic calls  
ᵇ Require additional data manipulation  
ᶜ Available also as block-floating-point emulation for higher throughput  
ᵉ BFP16 emulation mode; enable with `AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16`

### XDNA 2 sparse matrix shapes (B sparse)

| TypeA × TypeB    | Shapes                       |
|------------------|------------------------------|
| `int8 × int8`    | 4x16x8, 8x16x8               |
| `int16 × int8`   | 2x16x8, 4x16x8               |
| `int16 × int16`  | 2x8x8, 4x8x8                 |

Sparse B requires 50% minimum sparsity; data layout is 64b mask + compressed
values. On XDNA 2 hardware computes this natively (not emulated like XDNA 1).

### Relevance to Q4NX GEMM

The Q4NX kernel weights are stored as Q4_1 blocks: 16 4-bit values + 1 `d`
(scale) + 1 `m` (min) per 16-element group. The tile kernel must dequantize
to BF16 before the `mmul`. The natural path:

1. Load packed int4 weight block from DM via `aie::vector<int4, 32>` (or
   `aie::vector<uint8, 16>` reinterpreted)
2. Unpack to `int8` with `.unpack()`, extract scale/min from the block header
3. Dequantize: `bf16_val = int8_val * d + m` using `aie::mul` + `aie::mac`
   or scalar multiply
4. Feed dequantized BF16 tiles into `aie::mmul<4, 8, 4, bfloat16, bfloat16>`

The 80KB streaming stride observed in the weight DMA BDs corresponds to the
number of weight blocks that fit in the double-buffer window. 80KB / (16 elements
× 4 bits + 2 × 16b scale/min) ≈ approx 3640 Q4_1 blocks per 80KB chunk.

The selection of 4x8x4 (or 4x8x8) as the inner mmul shape for BF16 is
constrained by what XDNA 2 natively supports. The 80KB buffer sizing likely
derives from tiling these shapes across the 17-tile column topology.

---

## Memory and Buffer Streams

### Data Memory model

Each AIE tile has access to up to 4 Data Memories (DM). Static data (lookup
tables, ping-pong buffers) is placed in DM by the linker or via CDO WRITE32
at kernel load time. The CDO WRITE32 commands in Group A/B load exactly this
static DM content — the values written there are the dequant lookup tables
and/or tile-size constants the kernel uses.

**Bank conflict avoidance:** Multiple simultaneous DM accesses from the same
tile must target different memory banks. The `aie_dm_resource` enum tags
pointers to virtual resources:

```cpp
void fn(int __aie_dm_resource_a * A,
        int                     * B) {
    auto v1 = aie::load_v<8>(A);  // resource a
    auto v2 = aie::load_v<8>(B);  // can issue same cycle as v1
}
```

This maps directly to the two DM ports observed in the XDNA 2 tile —
the DMA BD topology programs MM2S_0/MM2S_1 separately to land data in
different banks (input activation in one, weight in the other), enabling
the tile to issue both loads in the same VLIW slot.

### Tensor buffer streams (4D addressing)

Introduced in AIE-ML/XDNA 1. This is the mechanism behind the multi-level
DMA BD strides decoded in Group D:

```cpp
// Describe weight tensor as 4D in tile DM, block=32 BF16 elements
auto desc = aie::make_tensor_descriptor<bfloat16, 32>(
    aie::tensor_dim(M_tiles, K_stride),     // dim 0: rows
    aie::tensor_dim(K_tiles, N_stride),     // dim 1: cols
    aie::tensor_dim(N_tiles, 0),            // dim 2: iterate N times (repeat)
    aie::tensor_dim(outer,   outer_stride)  // dim 3: outer loop
);
auto stream = aie::make_tensor_buffer_stream(weight_ptr, desc);
aie::vector<bfloat16, 32> w_tile = stream.pop();
```

The `aie::tensor_dim(size, step)` parameters correspond directly to the
DMA BD `count`/`stride` fields decoded in the Group D 40-group structure.
The `step=0` trick (repeat current position) is the mechanism behind the
ping-pong double-buffering observed in the BD chain: one BD iterates over
the same 80KB window while the next queues the DMA to the next chunk.

For tensors needing >3 dimensions, the stream decomposes recursively: read
inner streams with `.pop()` from the outer stream. Up to 3 native dimensions
per level.

### Vector load/store functions

```cpp
// Aligned load (pointer must meet alignment)
aie::vector<bfloat16, 32> v = aie::load_v<32>(ptr);

// Unaligned load
aie::vector<bfloat16, 32> v = aie::load_unaligned_v<32>(ptr, aligned_elems);

// Floor-aligned load (rounds pointer down to n-element boundary)
aie::vector<bfloat16, 32> v = aie::load_floor_v<32>(ptr, n);

// Store
aie::store_v(ptr, v);
```

The 512b (64-byte) access capability of XDNA 2 enables loading 32 BF16
elements or 64 int8 elements in a single instruction.

### Circular iterators for ping-pong

```cpp
// Iterate over a ping-pong buffer that wraps after 2×BUF_ELEMS
auto it = aie::begin_vector_circular<32, 2*BUF_ELEMS>(ping_pong_base);
```

This is the natural API equivalent of the alternating BD chain (BD0 → BD1
→ BD0 → ...) that the DPU instruction buffer programs in the DMA controller.

---

## Lookup Tables — `aie::lut` / `aie::parallel_lookup` / `aie::linear_approx`

LUT functionality is XDNA 1+ only (present on NPU2). This is likely what
the static DM WRITE32 constants in the CDO are setting up.

### Memory layout requirement

For 4 parallel accesses (the only mode supported on XDNA 2), the LUT data
must appear **4 times** in memory: two pointers `LUT_ab` and `LUT_cd`, each
containing 2× repeated values interleaved at 128b (bank-width) granularity.

```cpp
constexpr unsigned size = 256;  // e.g. 256 dequant output values
alignas(aie::vector_decl_align) const bfloat16 lut_ab[size * 2] = { ... };
alignas(aie::vector_decl_align) const bfloat16 lut_cd[size * 2] = { ... };
aie::lut<4, bfloat16> dequant_table(size, lut_ab, lut_cd);
```

This duplication pattern is exactly the kind of structure that would appear
as large repeated blocks in the static DM content written by the Group A/B
CDO. If the Group A WRITE32 constants show an interleaved-repeat pattern at
128b boundaries, they are LUT data.

### Direct lookup

```cpp
aie::parallel_lookup<int8, aie::lut<4, bfloat16>> lookup(dequant_table, step_bits, bias);
auto bf16_vals = lookup.fetch(int8_input_vec);  // returns aie::vector<bfloat16, N>
```

`step_bits` discards the N lowest bits of the index (for sub-index precision).
`bias` shifts the zero point (relevant for asymmetric Q4_1 where `m ≠ 0`).

For Q4_1 dequant the table would be 16 entries per scale group. But because
`d` and `m` vary per block, a static LUT can only apply if the kernel uses a
normalized form. This is exactly the `q4nx.unscaled` → `q4nx` conversion step
observed in the FastFlowLM weight preparation pipeline: scales are pre-baked
so the kernel sees a fixed-scale quantization compatible with a static LUT.

### Linear approximation (for SiLU / activation functions)

```cpp
// LUT holds (slope, offset) pairs for piecewise-linear approximation
aie::lut<4, float, bfloat16> silu_lut(lut_size, lut_ab, lut_cd);
aie::linear_approx<bfloat16, decltype(silu_lut)> silu(silu_lut, step_bits, bias);
auto result_acc = silu.compute(input_vec);
```

For `bfloat16` input the computation is:
- `index = floor(input) >> step_bits + bias`
- `output = slope * input + offset`

The `layer.xclbin` kernel computes SiLU as part of the FFN (SwiGLU variant).
This linear approximation path is almost certainly how it is implemented —
slope/offset pairs stored in static DM, indexed by the BF16 activation value
after the first FFN linear projection.

| Input    | Offset | Slope     | Accumulator | Lanes |
|----------|--------|-----------|-------------|-------|
| bfloat16 | float  | bfloat16  | accfloat    | 16    |

Note: slope must be stored as `float` with the low 16 mantissa bits zeroed,
despite being BF16 in value — this is the memory layout constraint for the
hardware LUT interface.

---

## Block Floating Point — `bfp16ebs8` / `bfp16ebs16`

XDNA 2 only. Block vectors where multiple elements share a common exponent,
stored as a `aie::block_vector<bfp16ebs8, 64>`. Cannot be loaded via normal
`aie::load_v`; requires `aie::block_vector_input_buffer_stream` which uses a
separate FIFO memory interface to handle the size/alignment mismatch.

This is unrelated to Q4_1 / Q4NX but relevant if future kernels target the
BFP16 matrix multiplication path for higher throughput
(`AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16` / 8x8x8 shape).

---

## Implementation Sketch — Q4NX GEMM Tile Kernel

Rough structure of one compute tile in the `mm.xclbin` / `layer.xclbin`
kernel (before detailed validation):

```cpp
#include <aie_api/aie.hpp>
#include <aie_api/aie_adf.hpp>

// Q4NX block: 16 int4 weights + 1 bf16 scale (d) + 1 bf16 min (m)
// Pre-baked: d and m are folded in at conversion time (see model.q4nx pipeline)
// Kernel sees: 16 int4 values whose dequant is a fixed-scale LUT lookup

constexpr unsigned BLOCK  = 16;   // Q4_1 group size
constexpr unsigned M_TILE = 4;    // mmul row tile
constexpr unsigned K_TILE = 8;    // mmul inner tile  (= BLOCK/2 if int4 packed)
constexpr unsigned N_TILE = 4;    // mmul col tile

// Static DM: dequant LUT (scale pre-baked per model, 16 entries × 4 parallel copies)
alignas(aie::vector_decl_align) bfloat16 lut_ab[BLOCK * 2];
alignas(aie::vector_decl_align) bfloat16 lut_cd[BLOCK * 2];

void gemm_q4nx_tile(
    const bfloat16 * __restrict act,      // bo0: input activation (A matrix)
    const uint8_t  * __restrict weights,  // bo1/bo5: Q4NX weight stream
    bfloat16       * __restrict out       // bo2: output
) {
    aie::lut<4, bfloat16> dq_lut(BLOCK, lut_ab, lut_cd);
    aie::parallel_lookup<uint8, decltype(dq_lut)> dequant(dq_lut);

    using MMUL = aie::mmul<M_TILE, K_TILE, N_TILE, bfloat16, bfloat16>;
    MMUL acc;

    auto a_stream = aie::make_tensor_buffer_stream(act, /* ... */);
    auto w_stream = aie::make_tensor_buffer_stream(weights, /* ... */);

    for (unsigned k = 0; k < K_blocks; ++k) {
        // Load packed int4 → treat as uint8 (2 nibbles per byte)
        auto w_raw  = w_stream.pop().cast_to<uint8>();
        // Parallel dequant to BF16 via LUT
        auto w_bf16 = dequant.fetch(w_raw).cast_to<bfloat16>();
        auto a_tile = a_stream.pop();

        if (k == 0) acc.mul(a_tile, w_bf16);
        else        acc.mac(a_tile, w_bf16);
    }

    aie::store_v(out, acc.to_vector<bfloat16>());
}
```

The actual kernel adds SiLU (via `aie::linear_approx`) and the SwiGLU
pointwise multiply for the FFN path, plus the attention QKVO projections.
The per-tile work division across all 17 tiles is what determines the
BD count and stride values in the DPU instruction buffer.

---

## Key Constraints for xclbin Generation

| Parameter                     | Constraint                                              |
|-------------------------------|---------------------------------------------------------|
| Inner mmul tile (K dimension) | Must match Q4_1 group size or a divisor of it          |
| LUT layout                    | 4× repeated at 128b granularity, in aligned static DM  |
| Weight DMA stride             | 80KB per chunk (observed); changing requires new BDs   |
| Activation alignment          | 32-byte (256b) minimum; 64-byte (512b) for max perf    |
| Output buffer                 | Must be in different DM bank from activation input      |
| Tile count                    | Fixed at 17 for 8B; must be rederived for other sizes  |

Changing `intermediate_size` (e.g. 5120→32768 for a 30B FFN) requires
recomputing: tile count, per-tile K/N split, DMA BD count, and DPU instruction
buffer. The mmul shape itself is a fixed hardware capability — only the
loop trip counts and buffer descriptors change.
