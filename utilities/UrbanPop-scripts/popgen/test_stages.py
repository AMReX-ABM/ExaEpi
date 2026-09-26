"""Order independence of every stage, placement through S10, by per-stage digests.

Run from utilities/UrbanPop-scripts:  python -m popgen.test_stages BUNDLE [PUMA ...]

With no PUMAs the whole bundle is used. Placements are generated once, then the stages are re-run:

  * with every loop over independent units (popgen/units.py) reversed, and shuffled;
  * on the expanded persons with their rows shuffled;

and each stage's digest must equal the forward run's. The same placements run with a different
seed must change every stage (each draws, or depends on a stage that does). Placement itself is
re-run reversed too (block groups within each PUMA).
"""

import sys

import numpy as np

from generate_exaepi import assign, generate, placement_digest
from popgen import bundle, placement, units

SEED, OTHER_SEED = 5, 6


def run(b, pers, seed, mode):
    units.MODE = mode
    try:
        return assign(b, pers, seed, 0, verbose=False)[1]
    finally:
        units.MODE = "forward"


def main():
    b = bundle.read(sys.argv[1])
    pumas = sys.argv[2:] or None
    placements, _ = generate(b, SEED, 0, pumas, verbose=False)
    ref_pl = placement_digest(placements)
    units.MODE = "reverse"
    try:
        rev_pl = placement_digest(generate(b, SEED, 0, pumas, verbose=False)[0])
    finally:
        units.MODE = "forward"
    assert rev_pl == ref_pl, "placement depends on block-group order"

    pers = placement.expand(b, placements)
    print(f"{len(pers['bg'])} persons; placement digest {ref_pl}")
    ref = run(b, pers, SEED, "forward")
    trials = {"reverse": run(b, pers, SEED, "reverse"), "shuffle": run(b, pers, SEED, "shuffle")}
    perm = np.random.default_rng(1).permutation(len(pers["bg"]))
    shuffled = {k: v[perm] for k, v in pers.items() if k != "n_households"}
    trials["shuffled rows"] = run(b, shuffled, SEED, "forward")
    other = run(b, pers, OTHER_SEED, "forward")

    ok = True
    print(f"{'stage':26s}{'digest':>18s}  " + "  ".join(f"{t:>13s}" for t in trials) + "  new seed")
    for stage, d in ref.items():
        same = [trials[t][stage] == d for t in trials]
        changed = other[stage] != d
        ok &= all(same) and changed
        print(f"{stage:26s}{d:>18s}  " + "  ".join(f"{'same' if x else 'DIFFERS':>13s}"
                                                   for x in same)
              + f"  {'changes' if changed else 'SAME'}")
    if not ok:
        print("FAILED: a stage depends on iteration or row order, or ignores the seed")
        return 1
    print("popgen stages: every stage independent of iteration and row order; the seed changes "
          "every stage")
    return 0


if __name__ == "__main__":
    sys.exit(main())
