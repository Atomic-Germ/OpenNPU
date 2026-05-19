# open-npu2 — AMD XDNA NPU2 Open Toolchain

Reverse engineering and open toolchain for the AMD XDNA NPU2 (Phoenix/Strix SoC)
found in Ryzen AI series processors. Goal: understand the xclbin binary format and
build open re-implementations of the NPU kernels.

## Status

| Component | State | Description |
|-----------|-------|-------------|
| `tools/axlf.py`      | ✅ Working | AXLF container parser — sections, UUID, kernel interface |
| `tools/cdo.py`       | ✅ Working | CDO v2 parser — tile config, DMA BDs, static weights |
| `tools/txn.py`       | ✅ Working | DPU transaction buffer disassembler (runtime instructions) |
| `tools/xclbin_inspect.py` | ✅ Working | CLI tool wrapping all parsers |
| `kernels/`           | 🔧 Planned | Open MLIR-AIE kernel implementations |

## Quick Start

```bash
# Inspect any xclbin
python3 tools/xclbin_inspect.py /path/to/kernel.xclbin

# CDO summary only (tile configuration)
python3 tools/xclbin_inspect.py /path/to/kernel.xclbin --cdo-summary

# Dump static per-tile data to a directory
python3 tools/xclbin_inspect.py /path/to/kernel.xclbin --dump-pm /tmp/out/

# Full analysis (sections + connectivity + CDO)
python3 tools/xclbin_inspect.py /path/to/kernel.xclbin --sections --connectivity --cdo-summary --tile-programs
```

### Example Output

```
File   : Llama-3.2-1B-NPU2/mm.xclbin
UUID   : 55e428ff9846b9871c0ab1be41becb9a
Kernel : MLIR_AIE
Args   : ['opcode', 'instr', 'ninstr', 'bo0', 'bo1', 'bo2', 'bo3', 'bo4']

#    Kind                          Offset        Size
──────────────────────────────────────────────────────
0    MEM_TOPOLOGY              0x000002e0           88
1    AIE_PARTITION_PDI         0x00000338      415,208
...

── AIE Tile Configuration ──────────────────────────────────
  Total CDO commands : 4079
  Tiles touched      : 48
  Shim DMA reg writes: 24
  Tiles with DMA BDs : 40

  Static weights in core-tile DM (0x20000+):
    tile(0,2)  16212 bytes
    ...
    tile(7,5)  16216 bytes

  Program memory loads (0x10000-0x1CFFF, static VLIW code):
    (none — VLIW programs loaded at runtime via DPU instr buffer)
```

## Architecture — What's in an NPU2 xclbin

### NPU tile grid (Phoenix/Strix)

```
row 5: [ C0,5 ] [ C1,5 ] [ C2,5 ] [ C3,5 ] [ C4,5 ] [ C5,5 ] [ C6,5 ] [ C7,5 ]
row 4: [ C0,4 ] [ C1,4 ] [ C2,4 ] [ C3,4 ] [ C4,4 ] [ C5,4 ] [ C6,4 ] [ C7,4 ]
row 3: [ C0,3 ] [ C1,3 ] [ C2,3 ] [ C3,3 ] [ C4,3 ] [ C5,3 ] [ C6,3 ] [ C7,3 ]
row 2: [ C0,2 ] [ C1,2 ] [ C2,2 ] [ C3,2 ] [ C4,2 ] [ C5,2 ] [ C6,2 ] [ C7,2 ]
row 1: [ M0,1 ] [ M1,1 ] [ M2,1 ] [ M3,1 ] [ M4,1 ] [ M5,1 ] [ M6,1 ] [ M7,1 ]  ← memory tiles (512KB each)
row 0: [ S0,0 ] [ S1,0 ] [ S2,0 ] [ S3,0 ] [ S4,0 ] [ S5,0 ] [ S6,0 ] [ S7,0 ]  ← shim/interface tiles
```

### What the CDO configures (static, at load time)

1. **Stream switch topology** — which tiles talk to which via the stream network
2. **DMA buffer descriptors** — memory tile BDs (`0xA0000+`) and core tile BDs (`0x1D000+`)
3. **Static data in core DM** — lookup tables, quantization scales, constants  
   (≈ 10–16 KB per compute tile; baked into the xclbin)
4. **Lock / event state initialization**

### What happens at runtime (DPU instruction buffer)

The `instr` kernel argument is a buffer of AIEBU DPU instructions that:
- Load VLIW programs into tile program memory
- Reconfigure DMA buffer descriptors for each inference pass
- Synchronise tile execution (locks)

This is the _dynamic_ part — not inside the xclbin. See `tools/txn.py` for the
DPU instruction format.

## Key Discoveries

### Kernel sharing across models

| Kernel         | Shared by                               |
|----------------|-----------------------------------------|
| `dequant.xclbin` | ALL Llama and LFM2 models (identical) |
| `mm.xclbin`    | Llama-3.2-1B and all LFM2 models       |
| `attn.xclbin`  | Llama-3.2-1B and all LFM2 models       |

LFM2-1.2B and LFM2-2.6B share **identical xclbins** — only the runtime weight
buffers (bo0–bo4) differ.

### VLIW programs are NOT in xclbins

Contrary to initial expectations, the static CDO contains **no program memory
writes** in any observed NPU2 xclbin. All VLIW tile programs are loaded at
runtime via the DPU instruction buffer. The xclbin only carries:
- Hardware plumbing (stream switches, DMA topology)
- Static constants (quantization tables, bias/scale data)

### CDO v2 command format

Source: AMD XRT `tools/xclbinutil/aie-pdi-transform/libinclude/cdo_cmd.h`

```
cmd_word bits[7:0]   = opcode (XCDO_CMD_*)
cmd_word bits[15:8]  = module_id
cmd_word bits[23:16] = payload_length
opcodes: MASK_WRITE=0x02, WRITE=0x03, DMAWRITE=0x05, NOP=0x11, END=0x01FF
```

## Reference Sources

| Resource | Location |
|----------|----------|
| XRT xclbin.h | `/opt/xilinx/xrt/include/xrt/detail/xclbin.h` |
| CDO command header | `xdna-driver/xrt/src/.../aie-pdi-transform/libinclude/cdo_cmd.h` |
| AIEBU DPU opcodes | `xdna-driver/xrt/src/.../aiebu/src/cpp/preprocessor/aie2/aie2_blob_preprocessor_input.h` |
| AIE transaction header | `xdna-driver/xrt/src/.../aie-rt/driver/src/common/xaie_txn.h` |
| CDO generate example | `xdna-driver/xrt/src/runtime_src/aie-rt/driver/examples/xaie_intr_cdo_generate.c` |

Full format documentation: [docs/xclbin_format.md](docs/xclbin_format.md)

## Tools Reference

### `axlf.py`

```python
from tools import axlf
xc = axlf.load("kernel.xclbin")
print(xc.uuid.hex())          # UUID
print(xc.kernel_name)         # "MLIR_AIE"
print(xc.kernel_args)         # list of arg dicts
sec = xc.aie_pdi              # AIE partition section (kind=0x20)
sec.data                      # raw bytes of PDI section
```

### `cdo.py`

```python
from tools import cdo
parser = cdo.parse_xclbin_cdo(aie_section_data)
print(parser.topology_summary())
weights = parser.extract_static_weights()  # (col,row) -> [(offset, bytes)]
programs = parser.extract_tile_programs()  # any static PM writes
for cmd in parser.cmds:
    print(cmd.describe())
```

### `txn.py`

```python
from tools import txn
with open("instr_buffer.bin", "rb") as f:
    buf = f.read()
ops = txn.parse_txn_buf(buf)
for op in ops:
    print(op.describe())
```
