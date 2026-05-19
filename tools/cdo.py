"""
cdo.py — AMD AIE2 CDO (Configuration Data Object) parser

The CDO section inside an NPU2 xclbin's AIE partition configures the AI
Engine tile array: DMA channels, locks, and — crucially — the VLIW program
memory of each core tile.

This parser understands the CDO v2.0 format used by AMD XDNA / mlir-aie.

CDO v2 layout
─────────────
  [outer wrapper, 0x21c from big-section start]
    word -2 .. -1  : sync / framing words  (0xfdfb4175 / 0x00000004)
    word  0        : "CDO\0" magic = 0x004F4443
    word  1        : version = 0x00000200
    word  2        : total_size_in_words (incl. header)
    word  3        : checksum (XOR of all following words)
    word  4..5     : reserved
    word  6+       : command stream

CDO command format (each command)
──────────────────────────────────
  word[0]: {opcode[7:0], payload_words[15:8], col[23:16], row[31:24]}
           or  {opcode[31:24], payload[23:0]}   (NPI-style)
  The format varies by opcode; we handle the subset found in NPU2 xclbins.

Opcodes observed:
  0x0111  = AIE NPI WRITE (single register write):
              [op_word][reg_addr][value]
  0x0102  = AIE NPI BLOCK WRITE:
              [op_word][reg_addr][num_words][data × num_words]
  0x0103  = AIE NPI BLOCK SET (fill):
              [op_word][reg_addr][num_words][fill_value]
  0x0111  (high variant)  = sync / end marker

AIE tile address space (NPU2, Strix / Hawk Point):
  col_base = col × 0x01000000          (4 MB per column)
  row offsets within a column:
    row 0       = shim tile    (interface tile / DMA)
    row 1       = memory tile
    rows 2-5    = compute core tiles
  Register areas within a tile:
    0x00000 - 0x0FFFF  : core registers (PC, acc, vector, etc.)
    0x10000 - 0x1FFFF  : program memory (4 KB per core, base 0x10000)
    0x20000 - 0x2FFFF  : data memory (scratchpad)
    0x1D000 - 0x1DFFF  : DMA registers (shim)
"""

from __future__ import annotations
import struct
from dataclasses import dataclass, field
from typing import Optional


# ── AIE2 address decode ───────────────────────────────────────────────────────

# AIE2 physical tile address encoding (from amdxdna / aie-rt)
# addr = (col << 23) | (row << 18) | reg_offset   (NPU2 / Strix)
AIE2_COL_SHIFT = 23
AIE2_ROW_SHIFT = 18
AIE2_COL_MASK  = 0x7F
AIE2_ROW_MASK  = 0x1F

PM_BASE  = 0x10000   # program memory base within a tile
PM_SIZE  = 0x04000   # 16 KB program memory per core tile
DM_BASE  = 0x20000   # data memory base
DM_SIZE  = 0x20000   # data memory size

SHIM_DMA_BASE = 0x1D000
SHIM_ROW      = 0
MEM_ROW       = 1
CORE_ROW_MIN  = 2
CORE_ROW_MAX  = 5


def decode_aie2_addr(addr: int) -> tuple[int, int, int]:
    """Return (col, row, reg_offset) from an AIE2 memory-mapped address."""
    col    = (addr >> AIE2_COL_SHIFT) & AIE2_COL_MASK
    row    = (addr >> AIE2_ROW_SHIFT) & AIE2_ROW_MASK
    offset = addr & ((1 << AIE2_ROW_SHIFT) - 1)
    return col, row, offset


def is_pm_write(row: int, offset: int) -> bool:
    return row >= CORE_ROW_MIN and PM_BASE <= offset < PM_BASE + PM_SIZE


def is_dm_write(row: int, offset: int) -> bool:
    return DM_BASE <= offset < DM_BASE + DM_SIZE


def is_shim_dma(row: int, offset: int) -> bool:
    return row == SHIM_ROW and SHIM_DMA_BASE <= offset < SHIM_DMA_BASE + 0x1000


# ── CDO command dataclasses ───────────────────────────────────────────────────

@dataclass
class CdoCmd:
    offset: int     # byte offset within CDO body
    raw_op: int
    words: list[int] = field(repr=False, default_factory=list)

    @property
    def opcode(self) -> int:
        return self.raw_op & 0xFFFF

    def describe(self) -> str:
        return f"[+0x{self.offset:06x}] CDO_RAW  op=0x{self.raw_op:04x}"


@dataclass
class CdoWriteCmd(CdoCmd):
    addr: int = 0
    value: int = 0
    col: int = 0
    row: int = 0
    reg: int = 0

    def describe(self) -> str:
        area = _area_name(self.row, self.reg)
        return (f"[+0x{self.offset:06x}] WRITE     tile({self.col},{self.row})  "
                f"reg=0x{self.reg:05x}({area})  val=0x{self.value:08x}")


@dataclass
class CdoBlockWriteCmd(CdoCmd):
    addr: int = 0
    col: int = 0
    row: int = 0
    reg: int = 0
    count: int = 0
    data: list[int] = field(repr=False, default_factory=list)

    @property
    def is_pm_load(self) -> bool:
        return is_pm_write(self.row, self.reg)

    @property
    def is_dm_init(self) -> bool:
        return is_dm_write(self.row, self.reg)

    def describe(self) -> str:
        area = _area_name(self.row, self.reg)
        hint = " [PROGRAM_MEMORY]" if self.is_pm_load else (" [DATA_MEMORY]" if self.is_dm_init else "")
        return (f"[+0x{self.offset:06x}] BLOCKWRITE tile({self.col},{self.row})  "
                f"reg=0x{self.reg:05x}({area}){hint}  words={self.count}")


@dataclass
class CdoBlockSetCmd(CdoCmd):
    addr: int = 0
    col: int = 0
    row: int = 0
    reg: int = 0
    count: int = 0
    fill: int = 0

    def describe(self) -> str:
        area = _area_name(self.row, self.reg)
        return (f"[+0x{self.offset:06x}] BLOCKSET  tile({self.col},{self.row})  "
                f"reg=0x{self.reg:05x}({area})  fill=0x{self.fill:08x}  words={self.count}")


def _area_name(row: int, offset: int) -> str:
    if row == SHIM_ROW:
        if is_shim_dma(row, offset): return "SHIM_DMA"
        return "SHIM"
    if row == MEM_ROW:
        if is_dm_write(row, offset): return "MEM_DM"
        return "MEM"
    if CORE_ROW_MIN <= row <= CORE_ROW_MAX:
        if is_pm_write(row, offset): return "CORE_PM"
        if is_dm_write(row, offset): return "CORE_DM"
        return "CORE_REG"
    return f"ROW{row}"


# ── Tile program extractor ────────────────────────────────────────────────────

@dataclass
class TileProgram:
    col: int
    row: int
    base_reg: int       # register offset where program was written
    data: bytes         # raw VLIW instruction bytes

    @property
    def num_instrs(self) -> int:
        """AIE2 VLIW instruction width = 128 bits = 16 bytes."""
        return len(self.data) // 16

    def describe(self) -> str:
        return (f"Tile({self.col},{self.row})  pm_offset=0x{self.base_reg:05x}  "
                f"{len(self.data)} bytes  ({self.num_instrs} VLIW instructions)")

    def save(self, path: str):
        with open(path, "wb") as f:
            f.write(self.data)


# ── CDO parser ────────────────────────────────────────────────────────────────

CDO_MAGIC    = 0x004F4443   # "CDO\0" in LE
CDO_VERSION  = 0x00000200
CDO_HEADER_OFFSET = 0x21C   # empirically determined offset of CDO header within
                              # the AIE partition section (from big-section start)
CDO_BODY_OFFSET   = 0x228   # body starts after the 6-word header


# NPI write command word patterns (observed in NPU2 xclbins):
#   0x00000111 = NPI write single (followed by addr, value)
#   0x00030102 = NPI blockwrite (col=0,row=3,opcode=2, followed by addr,count,data)
#
# The command word format in the CDO body is (little-endian):
#   byte0: opcode     (0x01=write, 0x02=blockwrite, 0x03=blockset)
#   byte1: row
#   byte2: col
#   byte3: reserved/flags
# Address in next word uses the AIE tile offset only (NOT full abs addr), because
# the CDO engine applies the col/row base automatically.

CDO_OP_WRITE      = 0x01
CDO_OP_BLOCKWRITE = 0x02
CDO_OP_BLOCKSET   = 0x03
CDO_OP_SYNC       = 0x0F
CDO_OP_END        = 0x7F
CDO_OP_NOP        = 0xFF

# Seen in hex: 0x00000111 => opcode_nibble=0x11 (17 decimal), appears to be NPI write
CDO_OP_NPI_WRITE  = 0x11   # register write with absolute address


class CdoParser:
    def __init__(self, data: bytes, start_offset: int = CDO_HEADER_OFFSET):
        self.data = data
        self.start = start_offset
        self.words = list(struct.unpack_from(
            f"<{(len(data) - start_offset) // 4}I", data, start_offset
        ))
        self.pos = 0
        self.cmds: list[CdoCmd] = []
        self._pm_accumulate: dict[tuple, list[tuple[int, list[int]]]] = {}

    def _byte_off(self) -> int:
        return self.start + self.pos * 4

    def parse(self) -> list[CdoCmd]:
        """Parse CDO header then walk the command stream."""
        self.cmds = []
        self.pos = 0

        # Validate CDO header
        # From empirical analysis: header at pos 0,1,2 within CDO section:
        # framing word(s) precede the magic; actual magic at +8 bytes from start_offset
        hdr_magic_pos = 2   # "CDO\0" is at word offset 2
        if len(self.words) < 6:
            return self.cmds

        if self.words[hdr_magic_pos] == CDO_MAGIC:
            version   = self.words[hdr_magic_pos + 1]
            total_len = self.words[hdr_magic_pos + 2]
            checksum  = self.words[hdr_magic_pos + 3]
            # Body starts 6 words into CDO section (2 framing + 4 header words)
            self.pos = 6
        else:
            # Fallback: scan for magic
            for i, w in enumerate(self.words):
                if w == CDO_MAGIC:
                    self.pos = i + 4
                    break

        self._parse_body()
        return self.cmds

    def _parse_body(self):
        while self.pos < len(self.words):
            byte_off  = self._byte_off()
            cmd_word  = self.words[self.pos]
            opcode    = cmd_word & 0xFF
            row       = (cmd_word >> 8)  & 0xFF
            col       = (cmd_word >> 16) & 0xFF

            if opcode == CDO_OP_NPI_WRITE or cmd_word == 0x00000111:
                # NPI absolute register write: word0=opcode|row|col, word1=addr, word2=val
                if self.pos + 2 >= len(self.words):
                    break
                w = self.words[self.pos: self.pos + 3]
                addr  = w[1]
                value = w[2]
                dcol, drow, dreg = decode_aie2_addr(addr)
                cmd = CdoWriteCmd(byte_off, cmd_word, w, addr, value, dcol, drow, dreg)
                self.cmds.append(cmd)
                self.pos += 3

            elif opcode == CDO_OP_WRITE:
                # Write: word0=op|row|col, word1=reg_offset, word2=value
                if self.pos + 2 >= len(self.words):
                    break
                w = self.words[self.pos: self.pos + 3]
                reg   = w[1]
                value = w[2]
                cmd = CdoWriteCmd(byte_off, cmd_word, w, reg, value, col, row, reg)
                self.cmds.append(cmd)
                self.pos += 3

            elif opcode == CDO_OP_BLOCKWRITE:
                # Blockwrite: word0=op|row|col, word1=reg_offset, word2=count, word3..N=data
                if self.pos + 2 >= len(self.words):
                    break
                reg   = self.words[self.pos + 1]
                count = self.words[self.pos + 2]
                if self.pos + 3 + count > len(self.words):
                    count = len(self.words) - self.pos - 3
                w    = self.words[self.pos: self.pos + 3 + count]
                data = w[3:]
                cmd  = CdoBlockWriteCmd(byte_off, cmd_word, w, reg, col, row, reg, count, data)
                self.cmds.append(cmd)
                # accumulate PM data
                if is_pm_write(row, reg):
                    key = (col, row)
                    self._pm_accumulate.setdefault(key, []).append((reg, data))
                self.pos += 3 + count

            elif opcode == CDO_OP_BLOCKSET:
                # Blockset: word0, word1=reg, word2=count, word3=fill_value
                if self.pos + 3 >= len(self.words):
                    break
                reg   = self.words[self.pos + 1]
                count = self.words[self.pos + 2]
                fill  = self.words[self.pos + 3]
                w     = self.words[self.pos: self.pos + 4]
                cmd   = CdoBlockSetCmd(byte_off, cmd_word, w, reg, col, row, reg, count, fill)
                self.cmds.append(cmd)
                self.pos += 4

            elif opcode in (CDO_OP_END, CDO_OP_NOP):
                self.pos += 1
                if opcode == CDO_OP_END:
                    break
            else:
                # Unknown: try to advance safely
                self.cmds.append(CdoCmd(byte_off, cmd_word, [cmd_word]))
                self.pos += 1

    def extract_tile_programs(self) -> list[TileProgram]:
        """
        Collect all block-writes targeting core tile program memory and
        reconstruct the binary VLIW program for each tile.
        """
        programs: list[TileProgram] = []
        # Group all BLOCKWRITE commands by (col, row) for PM region
        pm_data: dict[tuple, list[tuple[int, list[int]]]] = {}
        for cmd in self.cmds:
            if isinstance(cmd, CdoBlockWriteCmd) and cmd.is_pm_load:
                key = (cmd.col, cmd.row)
                pm_data.setdefault(key, []).append((cmd.reg, cmd.data))

        for (col, row), segments in sorted(pm_data.items()):
            # Sort segments by register offset, concatenate
            segments.sort(key=lambda x: x[0])
            base_reg = segments[0][0]
            all_words: list[int] = []
            for reg, data in segments:
                all_words.extend(data)
            raw = struct.pack(f"<{len(all_words)}I", *all_words)
            programs.append(TileProgram(col=col, row=row, base_reg=base_reg, data=raw))

        return programs

    def topology_summary(self) -> str:
        """Return a human-readable summary of the AIE tile configuration."""
        tiles_seen: set[tuple] = set()
        shim_dma_writes = 0
        pm_writes: dict[tuple, int] = {}

        for cmd in self.cmds:
            if isinstance(cmd, (CdoWriteCmd, CdoBlockWriteCmd, CdoBlockSetCmd)):
                key = (cmd.col, cmd.row)
                tiles_seen.add(key)
                if isinstance(cmd, CdoBlockWriteCmd) and cmd.is_pm_load:
                    pm_writes[key] = pm_writes.get(key, 0) + cmd.count * 4
                if is_shim_dma(cmd.row, cmd.reg):
                    shim_dma_writes += 1

        lines = [
            "── AIE Tile Configuration ──────────────────────────────",
            f"  Tiles touched  : {len(tiles_seen)}",
            f"  Shim DMA reg writes: {shim_dma_writes}",
            "",
            "  Program memory loads (core tiles):",
        ]
        for (col, row), nbytes in sorted(pm_writes.items()):
            lines.append(f"    tile({col},{row})  {nbytes} bytes  ({nbytes//16} VLIW instructions)")
        if not pm_writes:
            lines.append("    (none found — CDO body may use a different encoding)")
        return "\n".join(lines)


def parse_xclbin_cdo(aie_section_data: bytes) -> CdoParser:
    """
    Parse the CDO from the raw bytes of an AIE partition section
    (section kind 0x20 as returned by axlf.load().aie_pdi).
    """
    parser = CdoParser(aie_section_data, start_offset=CDO_HEADER_OFFSET)
    parser.parse()
    return parser
