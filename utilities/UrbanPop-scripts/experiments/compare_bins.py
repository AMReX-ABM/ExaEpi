"""Read two UrbanPop .bin files (format v4) and compare their population structure.

    python compare_bins.py DELIVERED.bin GENERATED.bin

Prints agents, households, workers, work-from-home, work-group and class sizes (including the size
seen by the average member), home and day neighbourhood sizes, commuting and age bands side by
side. This is how the generator's dropped letter-suffixed NAICS codes were found: workers and every
work-related row fell by the same third while everything else matched.
"""
import struct
import sys
import zlib

import numpy as np

COLS = [("id", "<i8"), ("home_geoid", "<i8"), ("work_geoid", "<i8"), ("school_class_group", "<i4"),
        ("work_group", "<i4"), ("naics", "<i2"), ("household_id", "<i2"), ("school_id", "<i2"),
        ("nborhood", "<i2"), ("work_nborhood", "<i2"), ("workgroup", "<i2"), ("hh_cluster", "<i2"),
        ("school_class", "<i2"), ("age", "i1"), ("sex", "i1"), ("race", "i1"), ("travel", "i1"),
        ("veh_occ", "i1"), ("grade", "i1")]


def read_bin(path):
    with open(path, "rb") as f:
        magic, ver, n_naics, n_geo, n_agents, rec, codec, idx_end = struct.unpack("<2I 2I Q I I Q", f.read(40))
        ist = struct.Struct(f"<QQ III {n_naics}I")
        entries = [ist.unpack(f.read(ist.size)) for _ in range(n_geo)]
        cols = {c: [] for c, _ in COLS}
        for e in entries:
            geoid, off, nbytes, hpop = e[0], e[1], e[2], e[3]
            if hpop == 0:
                continue
            f.seek(off)
            raw = f.read(nbytes)
            if codec == 1:
                raw = zlib.decompress(raw)
            pos = 0
            for c, dt in COLS:
                size = np.dtype(dt).itemsize * hpop
                cols[c].append(np.frombuffer(raw[pos:pos + size], dtype=dt))
                pos += size
    return {c: np.concatenate(v) for c, v in cols.items()}


def group_sizes(key):
    _, n = np.unique(key, return_counts=True)
    return n


def describe(D):
    out = {}
    n = len(D["id"])
    out["agents"] = n
    hh = D["home_geoid"] * 100000 + D["household_id"]
    hs = group_sizes(hh)
    out["households"] = len(hs)
    out["mean hh size"] = hs.mean()
    out["employed (naics>=0)"] = int((D["naics"] >= 0).sum())
    out["wfh workers"] = int(((D["naics"] >= 0) & (D["travel"] == 7)).sum())
    out["in workgroup"] = int((D["workgroup"] > 0).sum())
    wg = D["work_group"][D["work_group"] >= 0]
    ws = group_sizes(wg)
    out["work groups"] = len(ws)
    out["work group size mean"] = ws.mean()
    out["work group size-1 share"] = (ws == 1).mean()
    out["agents in wg of size>=10"] = ws[ws >= 10].sum() / max(ws.sum(), 1)
    out["mean wg size seen by worker"] = (ws ** 2).sum() / max(ws.sum(), 1)
    out["at school (school_id>0)"] = int((D["school_id"] > 0).sum())
    sc = D["school_class_group"][D["school_class_group"] >= 0]
    cs = group_sizes(sc)
    out["class groups"] = len(cs)
    out["mean class size seen"] = (cs ** 2).sum() / max(cs.sum(), 1)
    nb = group_sizes(D["home_geoid"] * 1000 + D["nborhood"])
    out["home nborhoods"] = len(nb)
    out["mean home nborhood seen"] = (nb ** 2).sum() / nb.sum()
    day = np.where(D["workgroup"] > 0, D["work_geoid"],
                   np.where(D["school_id"] > 0, D["work_geoid"], D["home_geoid"]))
    dn = group_sizes(day * 1000 + D["work_nborhood"])
    out["day nborhoods"] = len(dn)
    out["mean day nborhood seen"] = (dn ** 2).sum() / dn.sum()
    cl = group_sizes(D["home_geoid"] * 10000 + D["hh_cluster"])
    out["mean hh cluster seen"] = (cl ** 2).sum() / cl.sum()
    out["works outside home bg"] = int(((D["naics"] >= 0) & (D["work_geoid"] != D["home_geoid"])).sum())
    out["mean age"] = D["age"].mean()
    for lo, hi, lab in ((0, 4, "age0-4"), (5, 17, "age5-17"), (65, 200, "age65+")):
        out[lab] = int(((D["age"] >= lo) & (D["age"] <= hi)).sum())
    return out


a, b = read_bin(sys.argv[1]), read_bin(sys.argv[2])
da, db = describe(a), describe(b)
print(f"{'':32s}{'delivered':>14s}{'generated':>14s}")
for k in da:
    x, y = da[k], db[k]
    fmt = (lambda v: f"{v:14.3f}") if isinstance(x, float) else (lambda v: f"{v:14d}")
    print(f"{k:32s}{fmt(x)}{fmt(y)}")
