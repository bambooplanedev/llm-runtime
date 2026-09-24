#!/usr/bin/env python3
"""Мінімальний GGUF v3 із заголовком і tensor-info, без даних тензорів.
Використання: make_gguf.py out.gguf --arch qwen3 --layers 28 --kv-heads 8 --head-dim 128
             [--params-per-layer N] [--experts 128 --experts-used 8] [--split 1/3] [--no-tensors]
`--head-dim 0` пропускає attention.key_length → парсер мусить упасти на
embedding_length / attention.head_count.
`--split K/N` при K>1 пише лише split.* і тензори, як llama-gguf-split.
"""
import argparse, struct, sys

def s(x): b = x.encode(); return struct.pack("<Q", len(b)) + b
def kv_u32(k, v): return s(k) + struct.pack("<I", 4) + struct.pack("<I", v)
def kv_u16(k, v): return s(k) + struct.pack("<I", 2) + struct.pack("<H", v)
def kv_i32(k, v): return s(k) + struct.pack("<I", 5) + struct.pack("<i", v)
def kv_str(k, v): return s(k) + struct.pack("<I", 8) + s(v)

ap = argparse.ArgumentParser()
ap.add_argument("out"); ap.add_argument("--arch", default="qwen3")
ap.add_argument("--layers", type=int, default=28); ap.add_argument("--kv-heads", type=int, default=8)
ap.add_argument("--head-dim", type=int, default=128)
ap.add_argument("--embedding-length", type=int, default=4096)
ap.add_argument("--head-count", type=int, default=32)
ap.add_argument("--params-per-layer", type=int, default=1_000_000)
ap.add_argument("--experts", type=int); ap.add_argument("--experts-used", type=int)
ap.add_argument("--split")  # "1/3"
ap.add_argument("--no-tensors", action="store_true")  # перший шард у режимі --no-tensor-first-split
ap.add_argument("--pad-mb", type=int, default=0)
a = ap.parse_args()

split_no = None
if a.split:
    no, cnt = map(int, a.split.split("/"))
    split_no = no - 1

kvs = []
# llama-gguf-split пише метадані моделі лише в перший шард; решта — тільки split.* і тензори.
if not split_no:
    kvs += [kv_str("general.architecture", a.arch),
            kv_u32(f"{a.arch}.block_count", a.layers),
            kv_u32(f"{a.arch}.attention.head_count_kv", a.kv_heads)]
    if a.head_dim:
        kvs += [kv_u32(f"{a.arch}.attention.key_length", a.head_dim),
                kv_u32(f"{a.arch}.attention.value_length", a.head_dim)]
    else:
        kvs += [kv_u32(f"{a.arch}.embedding_length", a.embedding_length),
                kv_u32(f"{a.arch}.attention.head_count", a.head_count)]
    if a.experts:
        kvs += [kv_u32(f"{a.arch}.expert_count", a.experts), kv_u32(f"{a.arch}.expert_used_count", a.experts_used)]
if a.split:
    kvs += [kv_u16("split.no", split_no), kv_u16("split.count", cnt),
            kv_i32("split.tensors.count", a.layers * cnt)]

n_tensors = 0 if a.no_tensors else a.layers
# один тензор на шар з params_per_layer елементами (2D: 1000 x rest)
tensors = b""
for i in range(n_tensors):
    rows = 1000; cols = max(1, a.params_per_layer // rows)
    tensors += s(f"blk.{i}.attn_q.weight") + struct.pack("<I", 2) + struct.pack("<QQ", cols, rows) \
               + struct.pack("<I", 0) + struct.pack("<Q", 0)

hdr = b"GGUF" + struct.pack("<I", 3) + struct.pack("<Q", n_tensors) + struct.pack("<Q", len(kvs))
with open(a.out, "wb") as f:
    f.write(hdr + b"".join(kvs) + tensors)
    if a.pad_mb: f.write(b"\0" * (a.pad_mb * 1024 * 1024))
print(a.out, file=sys.stderr)
