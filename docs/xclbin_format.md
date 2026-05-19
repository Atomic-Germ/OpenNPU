# xclbin / AXLF Format Reference — AMD XDNA NPU2

## Overview

An `.xclbin` file is an AXLF (AMD Xilinx Loadable Binary) container.
It bundles kernel metadata, memory topology, and the AIE partition blob
(a CDO-bearing PDI) that configures the NPU tile array.

---

## AXLF Container Format

### File Header (`axlf` struct, 448 bytes)

| Offset | Size | Field                      |
|--------|------|----------------------------|
| 0x000  | 8    | magic `"xclbin2\0"`        |
| 0x008  | 4    | cipher_flags               |
| 0x00C  | 60   | key_block                  |
| 0x048  | 8    | uniqueTimestamp             |
| 0x050  | 20   | auth_signature             |
| 0x064  | 4    | header offset (→ 0x1B8)    |
| 0x1A0  | 16   | **UUID** (little-endian)   |
| 0x1C0  | 4    | **num_sections**           |
| 0x1C8  | ...  | **section table** starts   |

### Section Entry (`axlf_section_header` struct, **40 bytes**)

Confirmed by XRT static assert: `sizeof(axlf_section_header) == 40`.

| Offset | Size | Field            |
|--------|------|------------------|
| +0x00  | 4    | `m_sectionKind`  |
| +0x04  | 16   | `m_sectionName`  |
| +0x14  | 4    | implicit padding (for uint64 alignment) |
| +0x18  | 8    | `m_sectionOffset` (from file start)     |
| +0x20  | 8    | `m_sectionSize` (bytes)                 |

### Section Kinds (NPU2 xclbins)

| Kind  | Name                   | Contents                              |
|-------|------------------------|---------------------------------------|
| 0x02  | CONNECTIVITY           | XML: kernel args, memory connections  |
| 0x06  | MEM_TOPOLOGY           | Memory region descriptors             |
| 0x07  | IP_LAYOUT              | IP block layout                       |
| 0x08  | DEBUG_IP_LAYOUT        | Debug IP layout                       |
| 0x19  | ASK_GROUP_CONNECTIVITY | Compute unit group connections        |
| 0x1A  | ASK_GROUP_TOPOLOGY     | Compute unit group topology           |
| 0x20  | **AIE_PARTITION_PDI**  | AIE tile partition blob (CDO inside)  |

---

## CONNECTIVITY XML (kernel interface)

The `CONNECTIVITY` section (kind=0x02) is XML defining the MLIR_AIE kernel:

```xml
<CONNECTIVITY>
  <kernel name="MLIR_AIE" dpu_kernel_id="0x901">
    <arg id="0" name="opcode"  type="uint64_t"  .../>
    <arg id="1" name="instr"   type="char *"    .../>  <!-- DPU instr buffer -->
    <arg id="2" name="ninstr"  type="uint32_t"  .../>  <!-- instr word count -->
    <arg id="3" name="bo0"     type="void*"     .../>  <!-- weight/act buffer -->
    ...
    <arg id="7" name="bo4"     type="void*"     .../>
  </kernel>
</CONNECTIVITY>
```

---

## AIE Partition PDI Section (kind=0x20)

### PDI Wrapper (~540 byte header)

The AIE partition section starts with a Xilinx PDI image header.
The CDO payload begins at offset **0x21C** from the section start.

```
[+0x21C]  0xfdfb4175   framing begin marker
[+0x220]  0x00000004   4 more header words follow
[+0x224]  0x004F4443   CDO magic "CDO\0"
[+0x228]  0x00000200   CDO version 2.0
[+0x22C]  <N>          total CDO body word count
[+0x230]  <checksum>   XOR of body words
[+0x234]  ...          CDO command stream (body)
```

---

## CDO v2 Command Format

Source: `xrt/src/runtime_src/tools/xclbinutil/aie-pdi-transform/libinclude/cdo_cmd.h`

### Command Word Layout

```
bits[ 7: 0]  opcode
bits[15: 8]  module_id
bits[23:16]  payload_length  (number of 32-bit words that follow)
             If == 0xFF → long command: next word = real payload length
bits[31:24]  reserved
```

### Opcodes (`XCDO_CMD_*`)

| Opcode | Name         | Payload words              |
|--------|--------------|----------------------------|
| 0x02   | MASK_WRITE   | `[addr, mask, value]`      |
| 0x03   | WRITE        | `[addr, value]`            |
| 0x05   | DMAWRITE     | `[hi_addr, lo_addr, data*N]` |
| 0x07   | MASKWRITE64  | `[hi_addr, lo_addr, mask, value]` |
| 0x08   | WRITE64      | `[hi_addr, lo_addr, value]` |
| 0x11   | NOP          | `[]`                        |
| 0x01FF | END          | (terminates command stream) |

---

## AIE2 Tile Address Encoding (NPU2 / Phoenix SoC)

```
physical_addr = (col << 25) | (row << 20) | local_offset
```

### Grid layout (NPU2)

| Row | Tile type       |
|-----|-----------------|
| 0   | Shim tile       |
| 1   | Memory tile     |
| 2–5 | Compute cores   |

Columns: 0–7 (some kernels use cols 0–5 or 0–7 depending on model)

### Local address regions per tile type

**Compute core tiles (rows 2–5):**

| Range            | Region                            |
|------------------|-----------------------------------|
| 0x00000–0x0FFFF  | Core registers (stream switch, locks, events) |
| 0x10000–0x1CFFF  | Program memory (PM — VLIW code, if static)    |
| 0x1D000–0x1DFFF  | **Core-tile DMA buffer descriptors**          |
| 0x20000–0x2FFFF  | **Data memory (DM) — static weights/LUTs**   |
| 0x32000           | Core enable/reset register (MASKWRITE target) |

**Memory tile (row 1):**

| Range            | Region                            |
|------------------|-----------------------------------|
| 0x00000–0x7FFFF  | Data memory (512 KB SRAM)         |
| 0xA0000–0xAFFFF  | **Memory-tile DMA buffer descriptors** |

**Shim tile (row 0):**

| Range            | Region                            |
|------------------|-----------------------------------|
| 0x1D000–0x1FFFF  | Shim DMA registers                |

---

## CDO Configuration Content (observed)

For every kernel xclbin, the CDO does:

1. **Stream switch configuration** — MASKWRITE/WRITE to core register space  
   (`0x00000–0x0FFFF`) connects tiles via stream network

2. **Lock/event setup** — WRITE to core register space

3. **Core DMA buffer descriptors** — DMAWRITE to `0x1D000–0x1DFFF`  
   Sets up the core tile's local DMA channels

4. **Memory tile DMA BDs** — DMAWRITE to `0xA0000–0xAFFFF` in row 1  
   Routes data between shim, memory tile, and core tiles

5. **Static data in core DM** — DMAWRITE to `0x20000–0x2FFFF`  
   Bakes lookup tables, scale factors, or constants into tile memory.  
   ≈ 10–16 KB per tile depending on kernel type.

6. **VLIW programs**: NOT in the CDO for any observed NPU2 xclbin.  
   VLIW programs are loaded at runtime via the `instr` DPU instruction buffer.

---

## Static Data Sizes (observed cross-model analysis)

| Kernel         | CDO cmds | Tiles | Static DM total |
|----------------|----------|-------|-----------------|
| mm.xclbin      | 3807–4079 | 48   | 434–506 KB      |
| attn.xclbin    | 3812–4034 | 48   | 243–327 KB      |
| dequant.xclbin | 2980–3044 | 48   | 49–60 KB        |
| layer.xclbin   | 2763–3645 | 39–48| 164–287 KB      |
| conv.xclbin    | 3436      | 48   | 58 KB           |

### Shared kernels across models

- `dequant.xclbin`: **Identical** across all Llama and LFM2 models  
- `mm.xclbin`: Shared between Llama-3.2-1B and LFM2 family  
- `attn.xclbin`: Shared between Llama-3.2-1B and LFM2 family  
- LFM2-1.2B and LFM2-2.6B have **identical xclbins** (runtime weights differ only)
