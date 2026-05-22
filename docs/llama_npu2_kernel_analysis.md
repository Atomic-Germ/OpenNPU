# Llama 3.2-1B NPU2 Kernel Analysis

Captured from FastFlow runtime via LD_PRELOAD intercept of `xrt::elf::elf(const char*, size_t)`.
25 ELF files intercepted, each containing a `.ctrltext` section with the DPU instruction stream.

## Kernel Groups

### Group A — Main Transformer Layer (ELFs #0,2,3,16-24 × 12 copies, 66KB)

**Tile layout:** shim(0,0) + compute tiles (1,2), (2,3-5), (3,2-5), (4,2-5), (5,2-5) = 17 tiles

**Instruction structure:**
- 42 WRITEs: enable compute tiles (reg=0x08200 val=1) and stream ports (reg=0x1f060 val=1)
- 1 `UNK_0x60` (opcode 0x60) — purpose unknown, appears after initial WRITE block
- 1 BLOCKWRITE to shim(0,0) reg=0 with 15500 words (62KB): DPU instruction program loaded to shim

**7 Kernel Arguments (from RELA DDR patches, 380 total per ELF):**

| Arg | Patches | Stride | Total Size | Role |
|-----|---------|--------|------------|------|
| 1   | 2       | —      | small      | Input activation or output buffer |
| 2   | 1       | —      | single     | Layer norm weight (ln_weight) |
| 3   | 1       | —      | single     | Layer norm bias (ln_bias) |
| 4   | 4       | 64MB   | ~256MB     | KV-cache write buffer (grows per token) |
| 5   | 336     | 80KB   | ~26.9MB    | Weight tensor stream (Q/K/V/O/FFN chunks) |
| 6   | 4       | 64MB   | ~256MB     | KV-cache read buffer (attention scores) |
| 7   | 32      | 320KB  | ~10MB      | Output activation / scratch buffer |

**Autoregressive decode observation (ELFs #16-24):**
- arg4's `max_off` increases by 0x200 (512 bytes) per ELF step
- Confirms 9 autoregressive decode steps with KV-cache advancing 512 bytes/token
- KV cache: 512 bytes/token = 256 float16 values → 8 heads × 32 = 256 per K or V

**Weight streaming pattern:**
- 336 DMA patches into arg5 buffer, stride = 0x14000 = 80KB
- NPU double-buffers 80KB chunks: load next chunk while computing on current
- 336 × 80KB = 26.9MB weight data accessed per forward pass

### Group B — Init/LOADPDI Kernel (ELF #1, 348KB)

**Same tile layout as Group A.**

**Function:** Loads the AI core program (PDI) into all compute tiles. Called once at model load time.

**4 kernel arguments:**
- arg4: 2016 patches, stride=80KB, max_off=163MB → full model weight loading

### Group C — Embedding/Small Kernels (ELFs #4-7, 5-19KB)

**Tile layout:** shim(0,0) only (no compute tiles).

**2 kernel arguments:**

| ELF | Patches | arg1 stride | arg2 stride | Role |
|-----|---------|-------------|-------------|------|
| #4  | 80      | 160KB       | 512KB       | Token embedding (smaller model portion) |
| #5  | 128     | 160KB       | 512KB       | Token embedding (larger portion) |
| #6  | 128     | 160KB       | 512KB       | Token embedding (larger portion, different offsets) |
| #7  | 32      | 640KB       | 2MB         | LM head output projection |

**Unknown opcodes:** `UNK_0x7f`, `UNK_0xff`, `UNK_0x18` — likely packed DMA BD configuration

### Group D — All-Column DMA Setup (ELFs #8,12,15 × 3 copies, 12KB)

**Tile layout:** shim tiles of all 8 columns (0-7), plus tile (0,16) = system-level.

**483 instructions:** 437 WRITEs + 40 `UNK_0x30` + 3 `UNK_0x10` + 1 MASKPOLL + 1 TCT + 1 BLOCKWRITE

**3 kernel arguments (40 patches) — FULLY DECODED:**

| Arg | Patches | Stride   | Total Size | Role              | DMA dir | Queue       |
|-----|---------|----------|------------|-------------------|---------|-------------|
| 1   | 8       | 256KB    | ~1MB       | Input activations | MM2S_0  | Even cols only |
| 2   | 16      | 512KB    | ~8MB       | Weight matrix     | MM2S_1  | All 8 cols  |
| 3   | 16      | 256B     | ~4KB       | Output config     | S2MM_0  | All 8 cols  |

**40 DMA BD Group Structure (complete topology):**

Each `UNK_0x30` (constant value `0x00000030`, a DMA fence) is followed by exactly 6 WRITEs
that program one shim DMA BD and start the DMA transfer. The 37-word cycle repeats 40 times.

BD ping-pong pairs per column:
- **arg1 (activation, MM2S_0):** BD0 (pass 0) + BD1 (pass 1) — **even columns only** (0,2,4,6)
- **arg2 (weights, MM2S_1):** BD2 (pass 0) + BD3 (pass 1) — all 8 columns
- **arg3 (config out, S2MM_0):** BD14 (pass 0) + BD15 (pass 1) — all 8 columns

Full 40-group listing (col = shim column, BD = buffer descriptor index):

```
Grp  Arg  Addend       Col  BD   AddrHi  BD_ctrl      DMA Queue
[00] arg1 +0x00000000  col0  BD 0  hi=1  0xc40003ff  MM2S_0_Ctrl
[01] arg2 +0x00000000  col0  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[02] arg3 +0x00000000  col0  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
[03] arg2 +0x00080000  col1  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[04] arg3 +0x00000100  col1  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
[05] arg1 +0x00040000  col2  BD 0  hi=1  0xc40003ff  MM2S_0_Ctrl
[06] arg2 +0x00100000  col2  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[07] arg3 +0x00000200  col2  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
[08] arg2 +0x00180000  col3  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[09] arg3 +0x00000300  col3  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
[10] arg1 +0x00080000  col4  BD 0  hi=1  0xc40003ff  MM2S_0_Ctrl
[11] arg2 +0x00200000  col4  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[12] arg3 +0x00000400  col4  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
[13] arg2 +0x00280000  col5  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[14] arg3 +0x00000500  col5  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
[15] arg1 +0x000c0000  col6  BD 0  hi=1  0xc40003ff  MM2S_0_Ctrl
[16] arg2 +0x00300000  col6  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[17] arg3 +0x00000600  col6  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
[18] arg2 +0x00380000  col7  BD 2  hi=2  0xc0000000  MM2S_1_Ctrl
[19] arg3 +0x00000700  col7  BD14  hi=0  0xd00003ff  S2MM_0_Ctrl
--- second pass (BD1/BD3/BD15) ---
[20] arg1 +0x00000000  col0  BD 1  hi=1  0xc40003ff  MM2S_0_Ctrl
[21] arg2 +0x00400000  col0  BD 3  hi=2  0xc0000000  MM2S_1_Ctrl
[22] arg3 +0x00000800  col0  BD15  hi=0  0xd00003ff  S2MM_0_Ctrl
...same pattern col1-7, arg2 addend +0x480000..+0x780000, arg3 +0x900..+0xF00
[38] arg2 +0x00780000  col7  BD 3  hi=2  0xc0000000  MM2S_1_Ctrl
[39] arg3 +0x00000f00  col7  BD15  hi=0  0xd00003ff  S2MM_0_Ctrl
```

**Key findings:**
- `AddrHi` is the upper 8 bits of the 40-bit DDR physical address:
  - arg1: hi=1 → DDR at 0x1_XXXXXXXX (4–8 GB range)
  - arg2: hi=2 → DDR at 0x2_XXXXXXXX (8–12 GB range)
  - arg3: hi=0 → DDR at 0x0_XXXXXXXX (0–4 GB range, output/config may be normal VA range)
- arg1 is **only sent to even columns (0,2,4,6)**; odd columns receive their input via stream switches
  (inter-tile routing from adjacent even columns), not directly from DDR
- `BD_ctrl` values encode BD chaining and valid bits:
  - `0xc40003ff` (arg1/MM2S): valid+use_next_bd set, additional tiling flags
  - `0xc0000000` (arg2/MM2S): valid only, weight streaming mode
  - `0xd00003ff` (arg3/S2MM): valid+use_next_bd, output capture mode

**CRITICAL: RELA patching mechanism (empirically confirmed):**

The 6-WRITE group structure for each BD:
```
WRITE 1 [RELA patches w[0]]: w[2]=0x0 (reg=0 placeholder), w[4]=BD_ctrl
                              → w[0] gets overwritten with arg[N]_base + addend
WRITE 2: reg=0x81 (or col-encoded), val=0  → DMA size/config register
WRITE 3: reg=0x1dXXX|col_bits, val=addr_hi → BD word1 (addr_high field)
WRITE 4: reg=0x3, val=0x1d210/0x1d218/...  → DMA channel start queue address
WRITE 5: reg=0x1c, val=0                   → Status/lock clear
WRITE 6: reg=0x18, val=0                   → DMA channel enable/control
```

txn.py currently misidentifies WRITE 1 as `WriteInstr tile(0,0) reg=0x0` because
the placeholder w[0] has opcode byte 0x00 (= WRITE) and w[2]=0x0 gives reg=0.
After RELA patching, w[0] = actual DDR physical address (lower 32 bits).

**BLOCKWRITE at end:** 128 words to col=0,row=16,reg=0x10100 — stream switch configuration table
(row=16 decoded via decode_addr from address word 0x01010100)

### Group E — 6-Column DMA Setup (ELFs #9,10 × 2 copies, 7KB)

**Tile layout:** 6 of 8 shim columns + tile(0,16).

**3 args (12 patches):**
- arg1: 4 × 256KB, arg2: 4 × 512KB, arg3: 4 × 256B

### Group F — Full-Grid Init (ELF #11, 24KB)

**Tile layout:** All 32 compute tiles (8 cols × 4 rows = rows 2-5) + shim(0,0).

**3 args (144 patches):**
- arg1: 64 patches stride=128B → packed data (K/V cache with per-head stride?)
- arg2: 64 patches stride=128B → same
- arg3: 16 patches, max_off=0xc000180 (3GB+ range?) → special mapping

**Role:** Grid-wide initialization: sets up all 32 compute tiles and memory modules.

### Group G — Full-Column DMA (ELFs #13,14 × 2 copies, 32KB)

**Tile layout:** 9 columns (0-8)! Col 8 > NPU2's 8 physical columns (0-7).

**3 args (160 patches):**
- arg1: 32 patches, max_off=768KB
- arg2: 64 × 512KB
- arg3: 64 × 256B

**Note:** Col 8 in the address space may refer to a special AIE NOC/MMIO space.

---

## DMA Topology Summary

```
DDR Memory Layout (inferred):
  arg1 [~small]: Activation input buffer (~512KB typical)
  arg2 [~1MB]:   Layer norm weight
  arg3 [~1MB]:   Layer norm bias
  arg4 [~256MB]: KV-cache buffer A (query attention write)
  arg5 [~27MB]:  Weight tensor stream (Q/K/V/O + FFN weights, all layers)
  arg6 [~256MB]: KV-cache buffer B (attention score read)
  arg7 [~10MB]:  Output activation buffer

NPU2 Grid Usage (Llama 3.2-1B):
  Shim row 0:    DMA controllers for all 8 columns
  Mem row 1:     Memory tiles (not explicitly addressed)
  Compute rows 2-5, cols 1-5: GEMM/attention cores (17 of 32 tiles)
  Compute rows 2-5, cols 6-7: IDLE (unused for 1B model)
```

## RELA Patch Format

The `.rela.dyn` section uses **12-byte RELA entries** (not 8-byte REL):
- `r_offset` (4B): Byte offset within `.ctrltext` where the DDR address is patched
- `r_info` (4B): `{sym[31:8] | type[7:0]}` — sym = kernel arg index (1-indexed), type = 5 (AMD_R_AIEBU_BD_PATCH)
- `r_addend` (4B): Byte offset within the arg buffer for this DMA chunk

At runtime, XRT patches: `ctrltext[r_offset] = (arg_base_addr[sym-1] + r_addend) & 0xFFFFFFFF`

**CRITICAL: RELA patches `w[0]` (the first word), NOT `w[4]` (the value word).**

This was confirmed empirically:
```python
r_off → instr_by_offset[r_off] → instr[N]
word_in_instr = (r_off - instr[N].offset) // 4  # = 0 for ALL patches
```
`word_in_instr = 0` for every RELA entry across all 25 ELFs. The opcode byte of the
patched instruction is 0x00 (= WRITE) because DDR addresses are 4-byte aligned
(lower 2 bits = 0, and the aiebu format has opcode in bits[7:0]).

**Before patching:** `w[0]` = placeholder encoding `{BD_slot_hint[31:16] | 0x0000}`, e.g.:
- `0x00010000` (arg1, BD0/1 slot)
- `0x00020000` (arg2, BD2/3 slot)  
- `0x00004000` (arg3, BD14/15 slot)

**After patching:** `w[0]` = actual DDR physical address lower 32 bits of `arg[sym] + r_addend`

The upper 32 bits (addr_high) are written separately by the adjacent WRITE to `BD_N.word1`
(e.g., reg=`0x1d004` = BD0.word1 with value=1 → DDR bits[39:32]=1, i.e., 4-8 GB range).

## Unknown Opcodes

| Opcode | Context | Confirmed/Likely Function |
|--------|---------|--------------------------|
| 0x30   | Group D (40×) | **CONFIRMED: DMA_BD_FENCE** — constant `0x00000030`, 1-word fence preceding each BD setup group |
| 0x10   | Group D BLOCKWRITE payload | Stream switch connection config |
| 0x60   | Group A outer txn | Unknown, appears once after initial WRITEs |
| 0x18   | Groups C,F | DMA BD config (in conjunction with 0x7f/0xff) |
| 0x7f   | Groups C shim-only | Packed shim DMA BD configure (3-word compact format) |
| 0xff   | Groups C shim-only | Packed shim DMA BD second field |

## BD Control Word Values (empirically observed)

| Value      | Used for | Direction | Decoded meaning (partial) |
|------------|----------|-----------|--------------------------|
| 0xc40003ff | arg1 (input activation) | MM2S | valid_bd=1, use_next_bd=1, additional tiling |
| 0xc0000000 | arg2 (weights) | MM2S | valid_bd=0/simple mode, weight streaming |
| 0xd00003ff | arg3 (output config) | S2MM | valid_bd=1, use_next_bd=1, output capture |

Full BD control word decoding requires the AIE2 shim DMA BD7 register spec (not yet available).

## txn.py API

```python
import txn

instrs = txn.from_bytes(data)         # parse ctrltext bytes → list[Instr]
txn.describe_all(instrs)              # → multiline string of all instructions
txn.find_dma_bd_groups(instrs)        # → list[DmaBdGroup] for DMA BD analysis
txn.extract_dma_topology(instrs)      # → list[DmaTransfer] (requires DDR_PATCH instrs)

# DmaBdGroup fields:
#   .fence_instr_idx   — index of DMA_BD_FENCE in instrs list
#   .addr_write_idx    — index of the WRITE whose w[0] is the RELA relocation target
#   .ddr_addr_lo_placeholder  — pre-patch placeholder value at w[0]
#   .bd_ctrl           — BD control word (w[4] of the addr WRITE)
#   .col               — shim column (0-7)
#   .bd_id             — BD index (0-15)
#   .addr_hi           — DDR addr bits[39:32] (from BD.word1 WRITE)
#   .direction         — "MM2S" or "S2MM"
#   .channel           — 0 or 1
```

```bash
# Capture kernels for any model
mkdir -p /tmp/ff_instr
flatpak-spawn --host bash -c "
  timeout 120 bash -c '
    FF_DUMP_INSTR_PATH=/tmp/ff_instr \
    LD_PRELOAD=/home/atomic-germ/Code/open-npu2/tools/ff_instr_dump.so \
      /opt/fastflowlm/bin/flm run llama3.2:1b <<< \"Hello\" 2>&1
  '
"

# Extract .ctrltext from captured ELFs (use Python ELF parser, not objcopy)
# See /home/atomic-germ/Code/open-npu2/tools/elf_extract_ctrltext.py

# Analyze instruction stream
python3 -c "
import sys; sys.path.insert(0, 'tools')
import txn
data = open('/tmp/ff_txn/ctrltext_000.bin','rb').read()
instrs = txn.from_bytes(data)
print(txn.describe_all(instrs))
"
```
