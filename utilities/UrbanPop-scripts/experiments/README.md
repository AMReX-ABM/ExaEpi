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
