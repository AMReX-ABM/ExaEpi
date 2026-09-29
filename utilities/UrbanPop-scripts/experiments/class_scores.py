"""Class, childcare and college structure of .bin populations, side by side.

    python class_scores.py LABEL=FILE.bin [LABEL=FILE.bin ...]

What the vs-epicast school fixes dfddb1c (teachers shared across grades by enrollment) and 5cce68a
(NCES center-based care rates; college classes of ~30) change:
  * student-weighted mean / median class size per level, and the share of students in classes
    over 35 (school_class_group, students only);
  * students in (school, grade) groups with no teacher;
  * under-5s in care (childcare or preschool) by age, and childcare-center size (agents per
    childcare school, staff included).
"""
import os
import struct
import sys
import zlib

import numpy as np

# .bin reader, as in compare_bins.py (which can't be imported: it runs its comparison at import)
COLS = [("id", "<i8"), ("home_geoid", "<i8"), ("work_geoid", "<i8"), ("school_class_group", "<i4"),
        ("work_group", "<i4"), ("naics", "<i2"), ("household_id", "<i2"), ("school_id", "<i2"),
        ("nborhood", "<i2"), ("work_nborhood", "<i2"), ("workgroup", "<i2"), ("hh_cluster", "<i2"),
        ("school_class", "<i2"), ("age", "i1"), ("sex", "i1"), ("race", "i1"), ("travel", "i1"),
        ("veh_occ", "i1"), ("grade", "i1")]


def read_bin(path):
    with open(path, "rb") as f:
        _, _, n_naics, n_geo, _, _, codec, _ = struct.unpack("<2I 2I Q I I Q", f.read(40))
        ist = struct.Struct(f"<QQ III {n_naics}I")
        entries = [ist.unpack(f.read(ist.size)) for _ in range(n_geo)]
        cols = {c: [] for c, _ in COLS}
        for e in entries:
            off, nbytes, hpop = e[1], e[2], e[3]
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

LEVELS = [("preschool", 4, 4), ("elementary", 5, 10), ("middle", 11, 13), ("high", 14, 17),
          ("college", 18, 19)]


def weighted(sizes):
    """Student-weighted mean and median of class sizes (each student sees their class's size)."""
    per = np.repeat(sizes, sizes)
    return per.mean(), np.median(per)


def score(D):
    out = {}
    student = D["naics"] < 0
    at = D["school_id"] > 0
    grade = D["grade"].astype(np.int64)
    for name, lo, hi in LEVELS:
        m = student & at & (grade >= lo) & (grade <= hi) & (D["school_class"] >= 0)
        if not m.any():
            continue
        _, n = np.unique(D["school_class_group"][m], return_counts=True)
        mean, med = weighted(n)
        out[f"{name} class mean/median"] = f"{mean:.1f} / {med:.1f}"
        out[f"{name} students in classes > 35"] = f"{100 * n[n > 35].sum() / n.sum():.1f}%"
    # (school, grade) raw groups: work geoid, school id, grade
    k12 = at & (grade >= 4) & (grade <= 17)
    key = (D["work_geoid"][k12] * 1000 + D["school_id"][k12]) * 100 + grade[k12]
    uk, inv = np.unique(key, return_inverse=True)
    n_st = np.bincount(inv, weights=student[k12], minlength=len(uk))
    n_te = np.bincount(inv, weights=~student[k12], minlength=len(uk))
    out["K-12 students with no teacher in their grade"] = int(n_st[n_te == 0].sum())
    out["K-12 teachers in grades with no students"] = int(n_te[n_st == 0].sum())
    # care: under-5s in childcare (grade 3) or preschool (grade 4) at a school
    for a in range(5):
        m = (D["age"] == a) & (grade < 5)
        care = m & at & ((grade == 3) | (grade == 4))
        out[f"age {a} in care"] = f"{100 * care.sum() / max(m.sum(), 1):.1f}%"
    cc = at & (grade == 3)
    ck = D["work_geoid"][cc] * 1000 + D["school_id"][cc]
    _, csz = np.unique(ck, return_counts=True)
    out["childcare agents"] = int(cc.sum())
    out["childcare centers, mean / median agents"] = f"{csz.mean():.0f} / {np.median(csz):.0f}"
    return out


def main():
    cols = []
    for arg in sys.argv[1:]:
        label, path = arg.split("=", 1)
        print(f"reading {label}: {os.path.basename(path)}", file=sys.stderr)
        cols.append((label, score(read_bin(path))))
    keys = list(dict.fromkeys(k for _, s in cols for k in s))
    w = max(len(k) for k in keys)
    print(" " * w + "  " + "".join(f"{lb:>16s}" for lb, _ in cols))
    for k in keys:
        print(f"{k:{w}s}  " + "".join(f"{str(s.get(k, '-')):>16s}" for _, s in cols))


if __name__ == "__main__":
    main()
