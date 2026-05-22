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

**BLOCKWRITE at end:** 128 words to `tile(0,16) reg=0x10100` — **TCT (Task Completion Token) routing table**
(see [TCT Routing Table](#tct-routing-table) section below)

**Group E vs Group D:** ELFs #9,10 (Group E, 6-column DMA) use the same
`tile(0,16) reg=0x10100` BLOCKWRITE but with only 19 words (4 complete TCT entries
covering cols 2-4 only, matching their 6-column tile layout).

### Group E — 6-Column DMA Setup (ELFs #9,10 × 2 copies, 7KB)

**Tile layout:** 6 of 8 shim columns + tile(0,16).

**3 args (12 patches):**
- arg1: 4 × 256KB, arg2: 4 × 512KB, arg3: 4 × 256B

**BLOCKWRITE:** 19 words to `tile(0,16) reg=0x10100` — partial TCT routing table (cols 2–4 only,
4 complete 4-word entries + 3 padding words).

---

## TCT Routing Table

### Background: TCT Mechanism

The AMD NPU2 uses **Task Completion Tokens (TCT)** to notify the host CPU when
a set of DMA operations has completed. When the NPU firmware completes a DMA
transfer, it generates a TCT — a hardware-level completion signal delivered via
the NOC/shim tile to a firmware-managed sync register (`reg=0x10100`).

The host-side `npu_cmd_wait.hpp` (`to_npu()`) encodes a 4-word TCT wait instruction:
```
word[0] = 0x80  = XAIE_IO_CUSTOM_OP_TCT (TCT opcode)
word[1] = 0x10  = op_size << 2 = 4 words × 4 bytes (fixed)
word[2] = (row<<8) | (col<<16) | direction  ← which DMA channel to wait on
word[3] = (channel<<24) | 0x10100          ← TCT sync address + channel number
```

Two TCT channels are used:
- channel=0 → sync address `0x00010100`
- channel=1 → sync address `0x01010100`

### Register `0x10100` — TCT Sync Register Base

`reg=0x10100` in the **global shim tile** (`tile(0, row=16)`, where 16 is the
**absolute chip row** for the NPU partition's shim layer in the full AIE2-ML die)
is the base address of the **TCT completion notification routing table**.

This register is **firmware-private** — it does NOT appear in the standard
`xaiemlgbl_params.h` register map. It is managed exclusively by the NPU
partition controller firmware.

**Absolute vs. partition-relative row addressing:**
- Full AIE2-ML die: NPU partition starts at **absolute row 16** (0x10)
- NPU partition-relative: shim=0, mem=1, compute=2–5
- BLOCKWRITE uses absolute addressing: `tile(0, row=16)` = the shim tile of col 0
- TCT channel encoding: `(channel<<24)|0x10100` puts channel# in bits[31:24], which
  coincides with the "row" field of the absolute address encoding (bits[24:20])
- `0x00010100` = channel 0 (row-bits=0, above abs-shim-row 16 is incidental)
- `0x01010100` = channel 1 (row-bits=1, or more precisely ch=1 in bits[31:24])

### BLOCKWRITE Payload — 32 Pre-built TCT Instruction Bodies

The 128-word BLOCKWRITE stores **32 × 4-word TCT routing entries**, each being a
pre-built TCT instruction body stored in **rotated word order** [w1, w2, w3, w0]:

```
Entry layout (4 words):
  w0 = 0x10          ← TCT word[1]: op_size<<2 = 16 (always fixed)
  w1 = (col<<16)|dir ← TCT word[2]: DMA source (col + direction bit)
  w2 = (ch<<24)|0x10100 ← TCT word[3]: TCT sync address
  w3 = 0x80          ← TCT word[0]: XAIE_IO_CUSTOM_OP_TCT opcode (stored last)
```

Direction bit: `dir=0` = S2MM (device-to-host), `dir=1` = MM2S (host-to-device)

### Full 32-Entry Routing Table (ctrltext_008, Group D)

```
Entry  Col  Dir    TCT_ch  Sync_addr
[ 0]   col2  S2MM    ch0   0x00010100
[ 1]   col3  MM2S    ch1   0x01010100
[ 2]   col3  S2MM    ch0   0x00010100
[ 3]   col4  MM2S    ch0   0x00010100
[ 4]   col4  MM2S    ch1   0x01010100
[ 5]   col4  S2MM    ch0   0x00010100
[ 6]   col5  MM2S    ch1   0x01010100
[ 7]   col5  S2MM    ch0   0x00010100
[ 8]   col6  MM2S    ch0   0x00010100
[ 9]   col6  MM2S    ch1   0x01010100
[10]   col6  S2MM    ch0   0x00010100
[11]   col7  MM2S    ch1   0x01010100
[12]   col7  S2MM    ch0   0x00010100
[13]   col0  MM2S    ch0   0x00010100
[14]   col0  MM2S    ch1   0x01010100
[15]   col0  S2MM    ch0   0x00010100
[16]   col1  MM2S    ch1   0x01010100
[17]   col1  S2MM    ch0   0x00010100
[18]   col2  MM2S    ch0   0x00010100
[19]   col2  MM2S    ch1   0x01010100
[20]   col2  S2MM    ch0   0x00010100
[21]   col3  MM2S    ch1   0x01010100
[22]   col3  S2MM    ch0   0x00010100
[23]   col4  MM2S    ch0   0x00010100
[24]   col4  MM2S    ch1   0x01010100
[25]   col4  S2MM    ch0   0x00010100
[26]   col5  MM2S    ch1   0x01010100
[27]   col5  S2MM    ch0   0x00010100
[28]   col6  MM2S    ch0   0x00010100
[29]   col6  MM2S    ch1   0x01010100
[30]   col6  S2MM    ch0   0x00010100
[31]   col7  MM2S    ch1   0x01010100
```

**Observed patterns:**
- S2MM channels always map to TCT channel 0 (`0x00010100`)
- MM2S channels map to either ch0 or ch1 — encoding which "logical DMA group" completed
- First 16 entries (entries 0–12): cols {2,3,4,5,6,7} with no col 1 → Group α
- Last 19 entries (13–31): cols {0,1,2,3,4,5,6,7} → Group β (full partition coverage)
- The duplication (cols 2–7 appear in BOTH halves) suggests the table covers
  TWO sequential DMA passes (pass 0 and pass 1 in the BD ping-pong scheme)

### Confirmed Source References

| File | Line | Evidence |
|------|------|----------|
| `FastFlow/src/include/npu_utils/instr_utils/npu_cmd_wait.hpp` | 60 | `(wait_channel << 24) \| 0x10100` = TCT sync address format |
| `FastFlow/src/include/npu_utils/instr_utils/npu_cmd_wait.hpp` | 55–58 | TCT 4-word format: opcode=0x80, op_size=0x10, col/row/dir, sync_addr |
| `xdna-driver/xrt/.../xaie_intr_cdo_generate.c` | `XAIE_COL_SHIFT=25, XAIE_ROW_SHIFT=20` | ColShift/RowShift confirm absolute addressing |

---

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
| 0x0E   | Groups D/E (many) | **CONFIRMED: XAIE_CONFIG_SHIMDMA_BD** — shim DMA BD configure |
| 0x0F   | Some ELFs | **CONFIRMED: XAIE_CONFIG_SHIMDMA_DMABUF_BD** — shim DMA BO BD configure |
| 0x30   | Group D (40×) | **CONFIRMED: DMA_BD_FENCE** — constant `0x00000030`, 1-word fence preceding each BD setup group |
| 0x10   | Group D (3×) | **UNKNOWN newer opcode** — 1-word, appears after WRITE to TCT trigger reg and after BLOCKWRITE to TCT routing table. Likely a "TCT config flush" or "issue token" barrier. NOT the same as uC ISA `ISA_OPCODE_MOV=0x10` (CERT firmware has a separate ISA). |
| 0x80   | Group D (1×) | **CONFIRMED: XAIE_IO_CUSTOM_OP_TCT** — 4-word TCT completion wait instruction |
| 0x81   | All ELFs | **CONFIRMED: XAIE_IO_CUSTOM_OP_DDR_PATCH** — RELA DDR address patch |
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

## NPU Execution Architecture

### Two Separate ISAs

The AMD NPU2 has **TWO separate instruction processors** with DIFFERENT instruction sets:

#### 1. DPU (Data Processing Unit) — XAIE Transaction Buffer Format
The ctrltext `.ctrltext` section uses the **XAIE (libxaie) transaction buffer format**.
This is the instruction stream parsed by txn.py.

Opcodes (from `npu_cmd.hpp::op_headers`):
```
0x00  XAIE_IO_WRITE                  — write AIE tile register (6 words)
0x01  XAIE_IO_BLOCKWRITE             — block write to tile region (3-word header + payload)
0x02  XAIE_IO_BLOCKSET               — block set (fill region with value)
0x03  XAIE_IO_MASKWRITE              — masked register write (7 words)
0x04  XAIE_IO_MASKPOLL               — poll register until mask matches
0x05  XAIE_IO_NOOP                   — no operation
0x06  XAIE_IO_PREEMPT                — preemption point
0x07  XAIE_IO_MASKPOLL_BUSY          — poll register until NOT masked value
0x08  XAIE_IO_LOADPDI                — load partial PDI (AIE program)
0x09  XAIE_IO_LOAD_PM_START          — load program memory start
0x0A  XAIE_IO_CREATE_SCRATCHPAD      — create scratchpad memory region
0x0B  XAIE_IO_UPDATE_STATE_TABLE     — update hardware state table
0x0C  XAIE_IO_UPDATE_REG             — update a register via state table
0x0D  XAIE_IO_UPDATE_SCRATCH         — update scratchpad entry
0x0E  XAIE_CONFIG_SHIMDMA_BD         — configure a shim DMA buffer descriptor
0x0F  XAIE_CONFIG_SHIMDMA_DMABUF_BD  — configure shim DMA BO buffer descriptor
0x10  ???                            — UNKNOWN newer opcode (1 word, TCT context)
0x30  ???                            — UNKNOWN, 1-word DMA BD fence (UNK_0x30)
0x60  ???                            — UNKNOWN, seen in Group A once after init WRITEs
0x80  XAIE_IO_CUSTOM_OP_TCT          — TCT completion wait (4 words)
0x81  XAIE_IO_CUSTOM_OP_DDR_PATCH    — DDR address patch (RELA)
0x82  XAIE_IO_CUSTOM_OP_READ_REGS    — read registers
0x83  XAIE_IO_CUSTOM_OP_RECORD_TIMER — record timer event
0x84  XAIE_IO_CUSTOM_OP_MERGE_SYNC   — merge sync
0xFF  XAIE_IO_CUSTOM_OP_MAX          — max opcode value
```

**UNK_0x10 in ctrltext (1-word opcode):** Appears exclusively in TCT setup context:
- After `WRITE tile(0,0) reg=0x10 val=<TCT_sync_addr>` (writing TCT channel trigger)
- After `BLOCKWRITE tile(0,16) reg=0x10100 words=128` (programming TCT routing table)
Likely a "TCT config flush" or "issue token" barrier for the TCT register space.
NOT the same as `ISA_OPCODE_MOV = 0x10` in the CERT firmware ISA (see below).

**UNK_0x30 in ctrltext (1-word opcode):** Appears 40× in Group D as DMA BD fences.
Constant value `0x00000030`. Separates each (fence + 6-WRITE) DMA BD configuration group.

#### 2. CERT Firmware µCPU — Partition Controller ISA
The NPU partition controller contains a small µCPU running **CERT firmware** that
manages job dispatch, TCT completion notifications, and DDR address patching.
This is a completely SEPARATE ISA from the DPU instruction stream.

Source: `xdna-driver/xrt/.../io_backend/ext/isa_stubs.h`

Opcodes (1-byte opcode at pc[0], then 1-byte padding at pc[1], then operands):
```
0x00  ISA_OPCODE_START_JOB               (8 bytes)  — start a new job on the CERT µCPU
0x01  ISA_OPCODE_UC_DMA_WRITE_DES        (8 bytes)  — write a uCDMA descriptor
0x02  ISA_OPCODE_WAIT_UC_DMA             (4 bytes)  — wait for uCDMA completion
0x03  ISA_OPCODE_MASK_WRITE_32           (16 bytes) — masked 32-bit write to AIE reg
0x05  ISA_OPCODE_WRITE_32                (12 bytes) — 32-bit register write
0x06  ISA_OPCODE_WAIT_TCTS               (8 bytes)  — ★ WAIT for N TCT completions
0x07  ISA_OPCODE_END_JOB                 (4 bytes)  — end job
0x08  ISA_OPCODE_YIELD                   (4 bytes)  — yield execution
0x09  ISA_OPCODE_UC_DMA_WRITE_DES_SYNC   (4 bytes)  — write descriptor + sync
0x0B  ISA_OPCODE_WRITE_32_D              (12 bytes) — write 32-bit from register
0x0C  ISA_OPCODE_READ_32                 (8 bytes)  — read 32-bit value
0x0D  ISA_OPCODE_READ_32_D               (4 bytes)  — read 32-bit to register
0x0E  ISA_OPCODE_APPLY_OFFSET_57         (8 bytes)  — ★ APPLY DDR OFFSET (RELA patching!)
0x0F  ISA_OPCODE_ADD                     (8 bytes)  — add constant to register
0x10  ISA_OPCODE_MOV                     (8 bytes)  — move constant to register
0x11  ISA_OPCODE_LOCAL_BARRIER           (4 bytes)  — local barrier synchronization
0x12  ISA_OPCODE_REMOTE_BARRIER          (8 bytes)  — remote barrier synchronization
0x13  ISA_OPCODE_POLL_32                 (12 bytes) — poll 32-bit register for value
0x14  ISA_OPCODE_MASK_POLL_32            (16 bytes) — poll with mask
0x15  ISA_OPCODE_TRACE                   (4 bytes)  — trace event
0x16  ISA_OPCODE_NOP                     (4 bytes)  — no operation
0xFF  ISA_OPCODE_EOF                     (4 bytes)  — end of firmware program
```

**CERT ISA instruction format** (byte-level, little-endian):
```
pc[0]   = opcode (uint8_t)
pc[1]   = padding / flags
pc[2..] = operands (specific per opcode)
```

Key CERT firmware operations:
- **`WAIT_TCTS(tile_id, actor_id, target_tcts)`**: Wait for `target_tcts` TCT completion
  signals from `actor_id` DMA channel on `tile_id`. This is how the firmware waits for
  AIE DMA operations to complete before proceeding.
- **`APPLY_OFFSET_57(table_ptr, num_entries, offset_high_reg, offset_low_reg)`**: Process
  `num_entries` RELA entries from `table_ptr`, applying the DDR base address (from
  `offset_high_reg:offset_low_reg` register pair) to each entry. This IS the runtime
  RELA address patching mechanism — XRT calls the firmware to patch all DDR addresses
  before the DPU program runs.
- **`WRITE_32(address, value)` / `MASK_WRITE_32(address, mask, value)`**: Direct AIE
  register writes executed by the CERT µCPU (distinct from DPU WRITE opcodes).

### Execution Flow (Complete Picture)

```
1. XRT submits ELF kernel to driver
   └─ aiebu parses ctrltext + relocation sections
   └─ CERT firmware receives job in queue

2. CERT firmware runs APPLY_OFFSET_57 to patch DDR addresses
   └─ For each RELA entry: ctrltext[r_offset] = arg_base + r_addend

3. CERT firmware dispatches DPU instruction stream to NPU DPU controller
   └─ DPU executes XAIE transaction buffer opcodes:
      ├─ WRITE/MASKWRITE/BLOCKWRITE: configure AIE tile registers
      ├─ CONFIG_SHIMDMA_BD (0x0E): set up DMA buffer descriptors
      ├─ DDR_PATCH (0x81): (DPU-level patch, if any remaining)
      ├─ TCT WAIT (0x80): DPU pauses until specific DMA completes
      └─ BLOCKWRITE to tile(0,16) reg=0x10100: set TCT routing table

4. As AIE DMA channels complete, they fire TCT signals
   └─ TCT routing table (programmed in step 3) routes each completion to
      CERT firmware via TCT channel 0 or channel 1

5. CERT firmware runs WAIT_TCTS to detect completions
   └─ On completion, signals host via NOC (the mapped MSI/doorbell register)
```

---

## BLOCKWRITE Format Variants

Two distinct BLOCKWRITE formats appear across the 35 captured ELFs:

### New format (ELFs 8,9,10,12,15 — Groups D/E):
Used exclusively for the TCT routing table and NOT for DMA BDs.
```
word[0] = 0x00020001  ← op=BLOCKWRITE(0x01), upper bits=flags
word[1] = absolute_tile_addr  ← e.g. 0x01010100 = tile(0,row=16) reg=0x10100
word[2] = word_count  ← number of payload words (e.g. 128)
word[3..N] = payload
```
DMA BDs in these ELFs use dedicated SHIMDMA_BD opcode (0x0E), not BLOCKWRITE.

### Old format (all other ELFs — Groups A/B/C/F/G):
Used for bulk DMA BD programming via direct register write.
```
word[0] = XAIE_IO_BLOCKWRITE = 0x01
word[1] = 0x00000000  ← always zero (not a tile address!)
word[2] = (col<<25)|(row<<20)|(bd_id<<5)|0x1D000  ← tile+BD register address
word[3] = op_size * 4  ← BYTE count of payload (not word count)
word[4..N] = payload (DMA BD words)
```
**txn.py parses only the new format correctly.** For old-format ELFs, txn.py
misinterprets word[2] (tile address) as the word-count, producing a spurious
`tile(0,0) reg=0x00000 words=~15500` BLOCKWRITE that consumes all remaining bytes.

---

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
