# UrbanPop population-variation experiments

Exploratory scripts behind the in-process population generation work: whether ExaEpi can draw a
fresh synthetic population each run instead of reading one fixed UrbanPop realization. They are
measurements, not production code; `build_precompute.py` and `generate_population.py` in the
parent directory are the parts meant to be kept.

## Running them

Their inputs and outputs (livelike ACS caches, bundles, logs, ensemble runs) live in
`data/UrbanPop/experiments/`, which is gitignored and on the host bind mount. Every default path is
relative to that directory, so run from there with the livelike environment:

```
cd data/UrbanPop/experiments
JAX_PLATFORMS=cpu $LIVELIKE_VENV/bin/python ../../../utilities/UrbanPop-scripts/experiments/theme_curve.py --variant minimal
```

Anything that calls `acs.puma` or `extract_pums_descriptors` needs `CENSUS_API_KEY`. The caches
make repeat runs fast, but livelike's cache key ignores `constraints_selection`, so a cache folder
must never be shared between constraint sets.

## What each one measured

All results are New Mexico, 2019 ACS 5-year, unless stated.

| script | question | result |
|---|---|---|
| `bench_pmedm.py` | cost of one P-MEDM solve per PUMA (3500804, up_expanded) | first ACS download 196 s, build 4.0 s, solve 1.9 s, TRS 5.2 s, peak RSS 8.1 GB |
| `bench_nreps.py` | do Laplace replicates reach ACS-scale spread? | yes on up_expanded (CV 0.1425, ratio 0.889); cost is the Hessian, flat in n_reps. **Retracted**: see `laplace_expanded.py` |
| `bench_repweights.py` | PUMS replicate weights via `make_replicate_pumas` | fails inside livelike (`IndexError` in `pums.housing_units`); the API does serve `WGTP1`–`80` |
| `compare_fit.py` | does the 123-constraint set fit the fields ExaEpi reads worse? | no: in_MOE 1.0000 vs 0.9996, median RAE 0.1421 vs 0.1466 |
| `compress_probe.py`, `quant_validate.py` | can allocation matrices be stored as uint16? | yes: perturbs the population 0.556x as much as a TRS re-draw |
| `rank_probe.py` | can replicates be stored as a low-rank basis? | no: spectrum nearly flat, k = 88 of 99 for 95% |
| `flow_baselines.py` | is LODES correlation a fair score for worker allocation? | no: per-worker LODES sampling scores 0.927 but gives workplaces of median size 2 |
| `fill_ipf.py`, `fill_ipf2.py` | can IPF replace `alloc_workers`' greedy fill loop? | yes; column-wise TRS keeps workplace sizes |
| `fill_order.py` | is the IPF fill order-independent? | bitwise identical under permutation once RNG is keyed on global ids |
| `alloc_ipf.py` | greedy vs IPF fill on identical CBP-derived demand | IPF: corr 0.857 vs 0.837, 296 vs 12,471 one-worker workplaces, 0.13% vs 3.10% fallback |
| `margin_rake.py` | rake onto perturbed ACS margins instead of Laplace? | hard raking diverges on inconsistent perturbed margins |
| `reps_by_constraints.py` | spread of minimal vs up_expanded replicates | 0.0494 vs 0.1425 (ratio 0.308 vs 0.889). **Retracted**: the 0.889 was one degenerate replicate; see `laplace_expanded.py` |
| `theme_curve.py` | which theme carries the spread? | none singly; flat at 0.31–0.36 up to 222 constraints, 0.899 only at 298. **The 298 jump is almost certainly a degenerate replicate** (run before the 1% concentration guard) |
| `reps_credible.py` | is up_expanded's spread a numerical artifact? | said no (totals exact, replicate in_MOE 0.9754), but none of those checks sees a degenerate replicate. **Retracted** |
| `rep_degeneracy.py` | why some replicates collapse into one cell | not the SE fill, not the seed; 9 of 18 PUMAs affected |
| `ridge_reps.py` | does a Hessian ridge fix degenerate replicates? | yes, per PUMA, with rejection for 3500300 |
| `validate_reps.py` | do replicates vary the population more than TRS? | 2.59x once targeting keeps each replicate's deviation |
| `hessian_spectrum.py` | Hessian eigenvalues (slow: smallest-eigenvalue Lanczos) | superseded by a rank test: design matrix rank 117 of 123 |
| `solve_profile.py` | where the solve time goes; dense-GEMM formulation | solve ~2 s, Laplace step ~37 s; dense objective 17x faster, identical to 1e-14; L-BFGS stops at its 500-iteration cap |
| `resolve_bundle_size.py` | size of a re-solve bundle | NM 1.3 MB core; US ~400–450 MB total |
| `gpu_solve.py` | dense solve on GPU vs CPU, 64- vs 32-bit | one PUMA: GPU f64 0.31 s vs CPU 0.84 s at 500 iterations; f32 never reaches tolerance (stuck at gradient 5e-3); padded vmapped batch of 18 PUMAs slower than sequential |
| `gpu_batch.py` | why batching didn't help | vmapped L-BFGS gains only ~1.5x even with identical shapes; padding (2.8x cells) and a summed objective both make it worse |
| `resolve_spread.py` | perturb-and-re-solve (also `--selection expanded`, composition via `composition.py`) (targets, prior via PUMS replicate weights or Bayesian bootstrap, both) vs iteration caps; PUMAs 3500804, 3500300, 3500100 | nested targets + replicate-weight prior: ratio 0.30–0.33, ≤0.2% of block groups outside MOE, 53–59% household turnover; prior alone moves households (44%) but not block-group totals (ratio 0.03–0.04); cap 1,000 within 0.3–1.5% of each draw's own converged answer with every measure unchanged; warm start gains nothing; cap 250–500 inflates spread and worsens fit |
| `laplace_ref.py` | the bundle's Laplace replicates on resolve_spread's measures | ratio 0.34/0.47/0.66 on 3500804/3500300/3500100, but 2.7%/11.7%/27.6% of block groups outside MOE; turnover 8–17% |
| `composition.py` | shared measures: constrained block-group shares vs ACS, household-level joint features no constraint sees, in_moe | — |
| `laplace_expanded.py` | pymedm's up_expanded Laplace replicates (3500804) with the concentration check | 1 of 20 replicates holds 55% of mass in one cell; the other 19 give ratio 0.41, not 0.909 |
| `fp32_capped.py` | is float32 good enough at a 1,000–2,000 iteration cap? (f64, f32, mixed, hybrid, inc32, inc64) | jaxopt in f32 stalls at ~4.5% from converged (line search can't see sub-rounding decreases) and is slower than f64; `lbfgs_incremental.py` in f32 matches f64 accuracy at 2.5–3.3x the speed (1,000 iterations: 0.24/0.36/0.41 s vs 0.61/1.15/1.35 s on 3500804/3500300/3500100) |
| `lbfgs_incremental.py` | the float32 solver: exponent matrix held in f64 and updated along the search direction, line search on the decrease via log1p/expm1, compact L-BFGS, fixed iterations | — (module) |
| `batch_draws.py` | batch draws of one PUMA into one vmapped `lbfgs_incremental` solve | at most 1.1–1.4x; GPU saturates (30 → ~88 W) by 8 draws; worse past 16. Batched and solo solves of the same draw differ by 0.3–1.8% at 1,000 iterations (0.07% / 1.2% on 3500804 / 3500300 at 2,000): capped results depend on rounding order |
| `solver_options.py` | line search (all 12 candidates vs full step first) and a per-PUMA stopping rule vs fixed 2,000, all 18 NM PUMAs against float64 converged | full step accepted 90–95% of iterations; trying it first gives identical results 8–20% faster. Stop when moved <0.1% per 250 iterations: 1,250–4,500 iterations, worst PUMA 0.38% → 0.05% from converged, statewide 0.070% → 0.038%, NM realization 8.7 → 9.2 s |
| `compare_bins.py` | structural comparison of two `.bin` populations (delivered vs generated) | tool; NM seed 1 after livelike-style placement: workers 837,006 vs 836,778, one-person work groups 6.7% vs 16.3%, 65+ 368,375 vs 365,883 |
| `concurrent_pumas.py` | different PUMAs solved concurrently in separate processes, each alone at its own shape | bit-identical to solo solves (8 of 8), but slower without MPS: 0.80–0.88x (the driver time-slices contexts). MPS would not start in this container (needs `--ipc=host`) |
| `gpu_sequential.py` | all 18 PUMAs to convergence, one at a time | GPU 75 s, CPU 174 s (2.3x); 3,048–12,143 iterations; pymedm's 500-iteration cap misplaces 1.19% of NM households |

`arm_e/run_E.sh` is arm E: 10 populations generated by `generate_exaepi.py` from the v2 bundle
(seeds 1–10), 3 disease seeds each, 160 days, plus arm A re-run with the same binary (runs in
`data/UrbanPop/experiments/ens/runs160E`). The disease model changed between the arm A–D runs and
this one (vs-epicast 730e3c3, signed pre-symptomatic period), so E is compared only with the re-run
A. Results, `compare_realizations.py --spatial cases00160`:

| measure | A (delivered, 10 seeds) | E (10 generated x 3 seeds) |
|---|---|---|
| attack rate | 0.5746 | 0.5755, between-population sd 0.0010, ICC 0.81 |
| cumulative infected | 1,200,543 | 1,202,240, ICC 0.85 |
| peak infectious | 228,219 | 230,306 (+0.9%, ~2.4 between-population sd) |
| peak day | 70.3 | 70.3, no population effect |
| per-block-group attack, mean pairwise r | 0.943 | 0.875 |

Means sit within the population-to-population spread except the peak, ~1% higher, consistent
with the generated populations' larger work groups (IPF fill: 6.7% vs 16.3% one-person groups).
For comparison on the older model, arm C (agent counts perturbed at full ACS scale) spread the
aggregates more (attack sd_between 0.0022) but moved the spatial pattern less (r 0.918 vs A 0.938).

`arm_f/run_F.sh` is arm F: the same 10 population seeds generated by the C++ port inside ExaEpi
(`agent.population_source = bundle`, CUDA build, written out with `agent.write_population`), 3
disease seeds each, with the epidemics on the arm A/E binary (checked first by reproducing
`A_base_s0` exactly), so the generator is the only difference from E. Same runs directory:

| measure | E (Python generator) | F (C++ generator) |
|---|---|---|
| attack rate | 0.5755, sd_between 0.0010, ICC 0.81 | 0.5756, sd_between 0.0014, ICC 0.87 |
| cumulative infected | 1,202,240 | 1,202,514 |
| peak infectious | 230,306, sd_between 869 | 230,991, sd_between 1,581 |
| peak day | 70.3 | 70.3 |
| per-block-group attack, mean pairwise r | 0.875 | 0.875 |

Paired by population seed, the two generators give populations within ~150 agents of each other
(2.09 M) and per-population mean attack rates correlated at r 0.87; every F-E mean difference is
within 1.6 standard errors. F's wider peak spread is a variance ratio of 2.65 on 9/9 degrees of
freedom, short of the one-sided 5% point (3.18).

**California.** `ca_v2.upb` (54.4 MB; its format-3 upgrade is committed via git-lfs as
`data/UrbanPop/ca_popgen.upb`, and `nm_v2.upb`'s plainly as `data/UrbanPop/nm_popgen.upb` -- see
Commutes below) is built like `nm_v2.upb` with
`--pumas $(python ../../utilities/UrbanPop-scripts/state_pumas.py 'base/06_CA/*.feather')` (265
PUMAs) and the CA feathers and LODES file; 3.2 h, almost all of it Census downloads. The donor recode
validates at 1.0000 on every field against the feathers (1.88 M overlapping persons). With the Python
oracle's seed-1 allocations injected, every C++ stage digest matches it and the population written
from inside ExaEpi is byte-identical to the oracle's `.bin` (39.25 M agents). Real solves on the
laptop GPU generate CA in 211 s (solve 106 s at `agent.popgen.gpu_streams = 4`, 119 s at 1; stages
96 s), 10.2 GB host memory; the Python oracle takes 14 min and 20 GB. Against the delivered
population, `compare_bins.py` agrees within ~0.5% on every measure except the intended drop in
one-person work groups (5.1% vs 14.1%). One 120-day run of `examples/inputs.ca` on each (same
disease seed): attack rate 0.9449 delivered vs 0.9454 generated, peak 17.14 M vs 17.19 M, peak day
32 both, deaths +0.6%.

**Commutes (bundle format 3).** LODES links jobs to residences from administrative records, not
daily commutes, and its long-distance tail is ~10x CTPP's: 11.7% of NM LODES jobs are > 100 km
from home against 0.8% of CTPP commuters (CA 11.4% vs 1.2%), 99.5% of them in 1-2-job block pairs.
The IPF fill of the version-2 bundle sent 20.4% of NM commuters > 100 km, and a worker's distance
ignored their reported travel time (0-5-minute commuters: median 13 km). What it took, measured one
change at a time on NM seed 1 with a scratch copy of `workers.allocate` (bit-identical to it with
every change off):

- A distance reliability r(d) on the LODES prior alone, iterated against CTPP, stalls at ~11%
  > 100 km: the CBP-sized slot demand is a hard column total, and IPF scales tiny far priors back up
  to fill slots nearby workers cannot reach. Soft column totals (unbalanced Sinkhorn: the column
  scale is sqrt(target / prior sum) rather than a damped running factor, whose fixed point is the
  hard total again) release that demand: 27% of it moves.
- Row repair added workers with weight 2 cnt + 1, near-uniform over a row's LODES cells. Harmless
  with LODES-shaped rows, but with rows split by commute-time band (mostly one worker each) it sent
  6% of 0-6-minute commuters > 100 km. Adding in proportion to the IPF values fixed it.
- A background over the 300 nearest destinations within 50 km (weight jobs x exp(-d / 10 km))
  covers destinations LODES never recorded: ~1.5% of commuters had no destination demanding their
  industry within 100 km in their home's LODES row.
- Workers who work from home (5.4% of NM's) are left out of the fill; ExaEpi keeps them home.
- CTPP county-to-county flows as further soft targets fought the distance fit and were dropped.

`calibrate_commute.py` fits the result per bundle (NM: speed kernel v50 30 km/h, sigma 0.9; r(d)
converged in 14 iterations to a profile TVD of 0.003). Scored on NM, seed 1; seed 7 rep 2, which
the calibration never saw, matches it to within 0.006 on every share and correlation below:

| measure | version 2 | version 3 | CTPP |
|---|---|---|---|
| commuters > 100 km | 20.4% | 1.10% | 0.89% |
| median / p90 distance | 17.2 / 235 km | 8.8 / 35 km | 8.8 / 34 km |
| commuters leaving their county | 37.9% | 13.7% | 12.4% |
| county-to-county TVD vs CTPP (LODES: 0.186) | 0.262 | 0.106 | — |
| tract-to-tract TVD vs CTPP (LODES: 0.468) | 0.532 | 0.423 | — |
| LODES block-group-pair correlation | 0.861 | 0.874 | — |
| implied straight-line speed > 60 km/h | 51% | 25% | — |
| time-distance rank correlation | 0.007 | 0.40 | — |
| walkers' median distance | 19.2 km | 3.2 km | — |
| workers in (dest, NAICS) cells of 20+ | 0.903 | 0.884 | — |
| work-group members in teams of <= 2 | 5.5% | 5.2% | — |

On the same seed-1 allocations, a 120-day NM epidemic (`nm-july4.cases`, disease seeds 3-5, CUDA
build): attack rate 0.5666 -> 0.5553 (seed-to-seed range 0.0012 -> 0.0028), peak 261 k -> 240 k,
peak day 71 -> 71-74, deaths -4.6%: with fewer long links, the epidemic spreads more slowly.

California (kernel v50 30 km/h, sigma 0.8; r(d) converged in 4 iterations to TVD 0.002), scored
from the `.bin` files as ExaEpi moves agents (educators at their schools), seed 1:

| measure | delivered | version 2 | version 3 | CTPP |
|---|---|---|---|---|
| commuters > 100 km | 11.3% | 15.0% | 1.35% | 1.31% |
| median / p90 distance | 19.0 / 111 km | 21.0 / 146 km | 11.6 / 43 km | — |
| commuters leaving their county | 36.6% | 40.1% | 21.3% | 18.0% |
| county-to-county TVD vs CTPP | 0.188 | 0.225 | 0.050 | — |
| county-to-county log-share correlation | 0.848 | 0.823 | 0.938 | — |
| LODES block-group-pair correlation | 0.736 | 0.760 | 0.864 | — |
| work-group members in teams of <= 2 | 5.0% | 3.5% | 3.2% | — |

The time kernel holds up (time-distance rank correlation 0.63, walkers' median 1.8 km). Cost: S3 is
~1.5x slower than version 2 (C++, CA, 20 threads: 25.8 vs 19.0 s; one thread 96 vs 58 s) with
peak memory 13.1 vs 10.3 GB -- each commuter's fill row now carries its home's LODES and
background destinations for its own time band. The Python oracle's S3 takes ~11 min for CA.

`arm_d/` holds the population-perturbation experiment (arm D of `utilities/compare_realizations.py`)
as patches against `upop_to_exaepi.py`, which apply cleanly to the commit they were written on,
plus `perturb_population.py`. The ensemble runs for arms A–D are in
`data/UrbanPop/experiments/ens/`.

## Data in `data/UrbanPop/experiments/`

| path | what |
|---|---|
| `nm_reps20_fixed.upb` | NM bundle, 20 replicates, minimal constraints, ridge-regularised (117.8 MB) |
| `nm_full.upb` | NM bundle, point estimate only (6.9 MB) |
| `llcache_minimal/`, `llcache_expanded/`, `theme_cache/`, `cache_*/`, `livelike_acs_cache/` | livelike ACS/PUMS caches, one per constraint selection |
| `theme_curve.jsonl`, `theme_cum.jsonl` | per-theme and cumulative spread measurements |
| `*.log` | output of the benchmark runs |
| `ens/` | arms A–D populations and 100 ExaEpi runs |
| `timing/` | ExaEpi NM timing run |
