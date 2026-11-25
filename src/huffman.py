"""Huffman coding for byte symbols, implemented with Counter and heapq."""
from __future__ import annotations
from collections import Counter
import heapq
import numpy as np


def build_codebook(data: np.ndarray) -> dict[int, str]:
    arr = np.asarray(data)
    if arr.dtype != np.uint8:
        raise ValueError("Huffman input must be uint8")
    counts = Counter(int(x) for x in arr.ravel())
    if not counts:
        return {}
    heap = []
    serial = 0
    for symbol, count in sorted(counts.items()):
        heapq.heappush(heap, (count, serial, symbol)); serial += 1
    if len(heap) == 1:
        return {heap[0][2]: "0"}
    while len(heap) > 1:
        f1, _, a = heapq.heappop(heap); f2, _, b = heapq.heappop(heap)
        heapq.heappush(heap, (f1 + f2, serial, (a, b))); serial += 1
    codes: dict[int, str] = {}
    def visit(node, prefix: str) -> None:
        if isinstance(node, int): codes[node] = prefix
        else:
            visit(node[0], prefix + "0"); visit(node[1], prefix + "1")
    visit(heap[0][2], "")
    return codes


def encode(data: np.ndarray) -> tuple[bytes, int, dict[int, str]]:
    arr = np.asarray(data)
    if arr.dtype != np.uint8: raise ValueError("Huffman input must be uint8")
    flat = arr.ravel()
    codes = build_codebook(flat)
    if not flat.size: return b"", 0, codes
    bits = ''.join(codes[int(x)] for x in flat)
    pad = (-len(bits)) % 8
    if pad: bits += '0' * pad
    packed = np.packbits(np.fromiter((int(b) for b in bits), dtype=np.uint8), bitorder="big").tobytes()
    return packed, len(bits) - pad, codes


def decode(payload: bytes, bit_length: int, codebook: dict[int, str], symbol_count: int) -> np.ndarray:
    if bit_length < 0 or symbol_count < 0 or bit_length > len(payload) * 8:
        raise ValueError("Invalid bit length or symbol count")
    if symbol_count == 0:
        if bit_length != 0 or codebook: raise ValueError("Invalid empty Huffman stream")
        return np.empty(0, dtype=np.uint8)
    normalized = {int(k): str(v) for k, v in codebook.items()}
    if not normalized or any(not 0 <= k <= 255 for k in normalized): raise ValueError("Invalid codebook symbols")
    if any(not c or any(b not in '01' for b in c) for c in normalized.values()): raise ValueError("Invalid Huffman code")
    rev = {v: k for k, v in normalized.items()}
    if len(rev) != len(normalized) or any(a != b and b.startswith(a) for a in rev for b in rev):
        raise ValueError("Codebook is not prefix-free")
    bits = ''.join(f'{b:08b}' for b in payload)[:bit_length]
    out, current = [], ''
    for bit in bits:
        current += bit
        if current in rev:
            out.append(rev[current]); current = ''
            if len(out) > symbol_count: raise ValueError("Decoded too many symbols")
    if current or len(out) != symbol_count: raise ValueError("Incomplete stream or incorrect decoded symbol count")
    if len(normalized) == 1 and (bit_length != symbol_count or next(iter(normalized.values())) != '0'):
        raise ValueError("Invalid single-symbol stream")
    return np.asarray(out, dtype=np.uint8)
