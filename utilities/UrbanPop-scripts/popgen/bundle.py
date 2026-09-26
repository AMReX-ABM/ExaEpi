"""Minimal reader for the precompute bundle (UPPB) -- the format build_precompute.py writes.

Kept separate from build_precompute.py so the reference generator needs neither livelike nor
upop_to_exaepi to load a bundle. Layout, little-endian:

    header      "<4I Q": magic "UPPB", format version, section count, 0, directory offset
    sections    each deflate-compressed (codec 1) or raw (codec 0), back to back
    directory   per section: u16 name length + name, u16 dtype length + numpy dtype.str, u8 ndim,
                u64 per dimension, u32 codec, u64 offset, u64 stored bytes, u64 raw bytes
"""

import json
import struct
import zlib

import numpy as np

MAGIC = 0x42505055
FORMAT_VERSION = 2
_HEADER = struct.Struct("<4I Q")


def read(path, require_version=FORMAT_VERSION):
    """All sections as numpy arrays keyed by name, plus 'meta' decoded from JSON."""
    out = {}
    with open(path, "rb") as f:
        magic, version, n_sections, _, dir_off = _HEADER.unpack(f.read(_HEADER.size))
        if magic != MAGIC:
            raise ValueError(f"{path}: not a precompute bundle (magic {magic:#x})")
        if require_version is not None and version != require_version:
            raise ValueError(f"{path}: bundle format {version}, need {require_version}; rebuild "
                             f"it with build_precompute.py")
        f.seek(dir_off)
        entries = []
        for _ in range(n_sections):
            (nl,) = struct.unpack("<H", f.read(2))
            name = f.read(nl).decode()
            (dl,) = struct.unpack("<H", f.read(2))
            dtype = f.read(dl).decode()
            (nd,) = struct.unpack("<B", f.read(1))
            shape = struct.unpack(f"<{nd}Q", f.read(8 * nd))
            codec, off, nstored, nraw = struct.unpack("<I QQQ", f.read(28))
            entries.append((name, dtype, shape, codec, off, nstored, nraw))
        for name, dtype, shape, codec, off, nstored, nraw in entries:
            f.seek(off)
            blob = f.read(nstored)
            if codec == 1:
                blob = zlib.decompress(blob)
            if len(blob) != nraw:
                raise ValueError(f"{path}: section {name} has {len(blob)} bytes, expected {nraw}")
            out[name] = np.frombuffer(blob, dtype=np.dtype(dtype)).reshape(shape)
    out["meta"] = json.loads(out["meta"].tobytes().decode())
    return out


def strings(b, name):
    """A string list stored as name.blob (concatenated UTF-8) + name.offsets (start offsets)."""
    blob, off = b[name + ".blob"].tobytes(), b[name + ".offsets"]
    return [blob[int(a):int(e)].decode() for a, e in zip(off[:-1], off[1:])]
