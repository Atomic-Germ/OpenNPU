# GPT-OSS-20B-NPU2 xclbin DM Analysis Findings

## Summary

Full reverse-engineering of all 6 xclbin kernels from NPU2 DM tile-memory.
Methodology: parse CDO v2 `CdoDmaWriteCmd` writes, collect DM init data per tile,
decode config word at DM offset 0x0094.

---

## 1. Model Architecture (derived from xclbin)

| Dimension      | Value | Source                                          |
|---------------|-------|-------------------------------------------------|
| hidden_size   | 4096  | mm.xclbin loop_cnt=31 → 32 × 128 = 4096        |
| expert_ffn    | 4096  | must equal hidden for `torch.cat` in gpt_oss.py |
| num_experts   | 128   | 128 / 32 tiles = 4 passes = expert loop_cnt=4   |
| NUM_CT_per_col | 4    | expert.xclbin loop_cnt=4                        |
| col_block_size | 128  | from gpt-oss.json config                        |
| row_block_size | 32   | from gpt-oss.json config                        |
| KV dim (GQA?) | 1024  | dequant.xclbin loop_cnt=7 → 8 × 128 = 1024     |

---

## 2. All-Kernel DM Comparison (tile 0,2 @ 0x0090-0x0097)

```
kernel          DM_size   w@0x0090    w@0x0094  loop_cnt  decoded_cols
attn               7924  0xd8000104  0x00b8000e       14          1920
expert             2692  0xa0000104  0x00b80004        4           640(*)
mm                16212  0x08000104  0x00b8001f       31          4096
dequant             4148  0x78000104  0x00b80007        7          1024
short_seq_mm      15188  0x08000104  0x00b8001d       29          3840
layer               5316  0xc0000104  0x00b80009        9          1280
```
(*) expert kernel loop_cnt=4 = NUM_CT_PER_COLUMN, NOT a column count.

`w@0x0090` high byte (0xa0, 0x08, 0x78, 0xd8, 0xc0): DM-internal pointer to kernel
data-structure (e.g. 0xa0 → tile address 0x200a0, where DMA FIFO control lives).

`w@0x0094` low uint16: **primary loop-control count** (meaning varies per kernel).

---

## 3. Tile Homogeneity

**All 32 compute tiles (8 cols × 4 rows, rows 2-5) are BYTE-IDENTICAL in every xclbin.**

Implication: expert routing is **purely runtime** — no compile-time expert assignment.
The AMDXDNA/XRT driver injects per-tile buffer pointers at inference time.

---

## 4. Per-Kernel Interpretation

### expert.xclbin (DM = 2692 bytes)
- Tiny compared to mm (16 KB) — handles routing + weighted sum, NOT matmul
- loop_cnt = 4 = `NUM_CT_PER_COLUMN` (4 tiles per column process parallel experts)
- Execution model: 32 tiles × 1 expert/pass → 128 experts / 32 = 4 passes

### mm.xclbin (DM = 16 KB) / short_seq_mm.xclbin (DM = 15 KB)
- loop_cnt = 31/29 → 4096 / 3840 input columns
- Hidden-size matrix multiply (q/k/v/o projection or dense FFN)
- short_seq variant likely handles shorter sequences with tail-padding optimisation

### dequant.xclbin (DM = 4148 bytes)
- loop_cnt = 7 → 8 × 128 = **1024 input columns**
- Decompresses MXFP4 weights before matmul
- 1024 = likely KV dimension = num_kv_heads × head_dim (GQA with 8 KV heads)
- Does NOT process expert FFN weights (those have hidden=4096 columns)

### attn.xclbin (DM = 7924 bytes)
- loop_cnt = 14 → 1920 = 15 × 128
- Likely encodes context sequence granularity (possibly 1920-token attention window)
  or number of head-processing passes × 128 head_dim

### layer.xclbin (DM = 5316 bytes)
- loop_cnt = 9 → 1280 = 10 × 128
- Orchestration layer? Dim 1280 unknown — possibly attention head scheduling

---

## 5. Expert Weight Storage Format (GPT-OSS)

MXFP4 expert weights are stored as:
```
gate_exps.weight:  packed → (E, R/rbs, C, block_bytes)
                   rearranged → (E, R//4, C, 4, block_bytes)
up_exps.weight:   same shape as gate
down_exps.weight: same shape (because expert_ffn = hidden → C is equal)

After merge (post_gpt_oss_process):
  gate_up:  (E, 2*R//4, C, 4, B)
  all:      (E, 3*R//4, C, 4, B)  stored as ffn_gate_up_down_exps.weight
```

The merge works because `C = hidden/col_block = 4096/128 = 32` is **equal for all three
matrices** when `expert_ffn = hidden = 4096`.

---

## 6. Implications for Qwen3-Coder-30B-A3B

Qwen3-Coder dimensions (from GGUF):
- hidden = 2048, expert_ffn = 768, num_experts = 128, col_block_size = 256

| Tensor     | Rows  | Cols  | C = cols/256 |
|------------|-------|-------|-------------|
| gate/up    | 768   | 2048  | 8           |
| down       | 2048  | 768   | 3           |

Since `C_gate_up (8) ≠ C_down (3)`, gate/up and down **cannot be concatenated** into
a single `ffn_gate_up_down_exps.weight`. They must be stored as separate tensors.

Current `qwen3moe.py` stores them separately — **this is correct**.

The per-expert rearrangement `(E, R/rbs, C, B) → (E, R//4, C, 4, B)` with
`NUM_CT_PER_COLUMN=4` is the expected final layout once a Qwen3-Coder expert.xclbin
is available and validated.

---

## 7. Key Path References

```
expert.xclbin:   FastFlow/src/xclbins/GPT-OSS-20B-NPU2/expert.xclbin
attn.xclbin:     FastFlow/src/xclbins/GPT-OSS-20B-NPU2/attn.xclbin
converter:       FLM_Q4NX_Converter/q4nx/models/gpt_oss.py
qwen3moe:        FLM_Q4NX_Converter/q4nx/models/qwen3moe.py
analysis tool:   open-npu2/tools/xclbin_dm_analyze.py
```

## 8. Parse Commands

```python
# Standard DM collection:
import sys, struct
sys.path.insert(0, '/path/to/open-npu2/tools')
from axlf import load
from cdo  import parse_xclbin_cdo, CdoDmaWriteCmd

CORE_DM_BASE = 0x20000

def collect_dm(path, col=0, row=2):
    ax  = load(path)
    cdo = parse_xclbin_cdo(ax.aie_pdi.data)
    buf = bytearray()
    for cmd in cdo.cmds:
        if isinstance(cmd, CdoDmaWriteCmd) and cmd.is_dm_init \
                and cmd.col == col and cmd.row == row:
            off = cmd.local_off - CORE_DM_BASE
            raw = struct.pack(f'<{len(cmd.data)}I', *cmd.data)
            need = off + len(raw)
            if need > len(buf): buf.extend(b'\x00' * (need - len(buf)))
            buf[off:off+len(raw)] = raw
    return bytes(buf)
```
