The scripts in this directory are for analyzing, comparing and plotting ExaEpi runs.

Scripts that *generate* ExaEpi's UrbanPop input `.bin` files live in `UrbanPop-scripts/` instead,
and are documented by that directory's own README.

Most of the plotting scripts share `plos_compbio_style.py`, so figures come out with one consistent
set of fonts, line widths and page widths rather than each script's own.

## Plotting a run

### `plot_timeseries.py`

Plots the number of infected, hospitalized and dead over time for a run of ExaEpi. Takes the output
file from an ExaEpi run as input -- the one named by the `diag.output_filename` option.

### `plot_geo.py`

Plots a map of infections, coloring each census tract (or county, with `--county_level`) according
to the number of infections there.

Its input is the per-day aggregated diagnostic files an ExaEpi run writes when
`agent.aggregated_diag_int` is set (e.g. `cases00050`) -- not plot files -- passed with
`--exaepi_files`, along with the day(s) to plot. It also needs shape files for the census tract (or
county) geometry and for state boundaries. To plot day 50 of a New Mexico run (census state code
35):

```
plot_geo.py -g cases00050 -d 50 -s tl_2010_35_tract10.shp -e gz_2010_us_040_00_500k.shp -o geo-nm.png
```

Passing several days to `-d` gives one column per day. Adding `--events_file` overlays an Epicast
run as a second row, with each column labeled by that day's log-scale Pearson r and RMSLE.

### `plot_geo_daynight.py`

Choropleth of daytime-minus-nighttime population per census tract or county -- where the commute
moves people to, rather than where infections are. Can overlay Epicast's LODES-derived commute
flows for comparison.

### `plot_geo_compare.py`

Compares an ExaEpi run against an Epicast run over a sequence of days at the level of individual
communities, as a time series of agreement metrics rather than as maps.

### `plot_gini_timeseries.py`

Plots how geographically widespread infections are over time, as either a Gini coefficient or
Moran's I.

### `plot_group_size_histogram.py`

Histogram of ExaEpi community or neighborhood sizes, or of neighborhoods per community, read from
the static aggregated diagnostics a run writes when `agent.aggregated_diag_int` is set
(`<prefix>_day_night_population.csv`, `<prefix>_nborhood_sizes.txt`,
`<prefix>_nborhoods_per_community.txt`) -- passed with `--prefix`, not a plot file.

### `plot_community_size_vs_density.py`

Plots community size (population per community) against population density.

## Comparing ensembles of runs

### `compare_realizations.py`

Decomposes the variance across a set of ExaEpi runs into the part coming from the disease model's
own randomness and the part coming from which synthetic population was used. A single run tells you
nothing about either, and a set of runs that vary only `agent.seed` measures only the first, so
without this it is easy to mistake one population's idiosyncrasies for a result.

Runs are grouped into arms by a prefix on their directory name:

| arm | varies | measures |
| --- | --- | --- |
| `A_*` | `agent.seed` only, one population | the noise floor |
| `B_*` | population *and* seed | adds whatever the populations differ by |
| `C_*` | population *and* seed | a second population-varying arm, e.g. a different generator |

```
compare_realizations.py --runs <dir> [--curve output_nm.dat] [--spatial cases00160]
```

Each arm's runs live in `<dir>/<arm>_<population>_s<seed>/`, and the population id is everything
before the `_s<seed>` suffix, so `B_r3_s1` is population `B_r3` at seed 1. Two families of metric
are reported, because they can disagree and the disagreement is informative: scalars reduced from
the per-day diagnostic table (peak infectious, peak day, attack rate, deaths), and the per-block-group
attack rate from the aggregated diagnostic, compared by correlation rather than variance since it is
a field rather than a scalar. A state-wide curve can absorb a lot of spatial rearrangement, so an
arm that looks identical on the aggregate may still have moved the epidemic around geographically.

For each scalar it reports each arm's spread, an F test of whether an arm is genuinely wider than
arm A (the sd ratio alone is not evidence -- with ten runs in A its own sd is uncertain to roughly a
quarter of its value), and, for arms built as several populations times several seeds, a one-way
decomposition into between-population and within-population components with an ICC. The ICC is the
number to read: it is the share of variance attributable to which population was used.

Runs still being written are detected by curve length and excluded by name rather than averaged in.
A truncated diagnostic table reduces to a plausible-looking but wrong result -- a mid-epidemic
snapshot reads as a much lower attack rate and puts the "peak" at whatever day the file stops -- and
nothing about the numbers themselves flags it.

## Comparing against Epicast

### `compare_to_epicast.py`

The main ExaEpi-vs-Epicast comparison: groups ExaEpi's per-phase `context_diag` columns into the
same transmission-context buckets Epicast reports, and plots the two side by side.

### `compare_group_sizes_to_epicast.py`

Compares the two models' distributions of workgroup size, school-class size and school size. With
`--cbp_state` it also overlays the real CBP establishment-size distribution as a reference.

### `compare_initial_cases.py`

Compares a case-seeding file against Epicast's seeded infections, per county.

### `calc_epicast_trans_probs.py`

Fits gamma distributions against Epicast's per-day transmission probabilities, for tuning ExaEpi's
own disease parameters.

### `extract_epicast_data.py`

Pre-extracts the small per-day summary `compare_to_epicast.py` actually plots from one or more
Epicast `run.events.bin` files, which are otherwise tens to hundreds of millions of rows each.

### `read_epicast_events.py`

Library (not a command) for reading an Epicast `run.events.bin` into a pandas DataFrame. Used by the
comparison scripts above.

## Shared modules

These are imported by the scripts above rather than run on their own.

* `plos_compbio_style.py` -- shared matplotlib styling (fonts, line widths, page widths) so every
  paper figure matches. See its docstring for why figure width is calibrated against the paper's
  LaTeX `\linewidth` rather than the journal's raw submission spec.
* `geo_agg_utils.py` -- geographic aggregation helper for the `plot_geo*` scripts, deliberately
  standard-library-only so reusing it does not pull in `yt` or the Epicast readers.

## Development tooling

### `custom-clang-format.py`

Wraps `clang-format` to fix its handling of AMReX's `AMREX_*` specifiers, which it otherwise puts on
the same line as the function name.

## Older standalone scripts

These predate the current workflow and nothing else references them; they are kept for reference
rather than being part of any current pipeline.

* `seirhd_metapop.py` -- a standalone metapopulation (multi-patch) SEIRHD model.
* `parseSafeGraph.py`, `travel_model_scripts.py`, `hdf5_process.py` -- earlier mobility-data
  exploration, from before the LODES-based travel model.

## Subdirectories

* `UrbanPop-scripts/` -- generates ExaEpi's UrbanPop input `.bin` files, including every agent's
  group structure (neighborhoods, work-groups, school classes); has its own README.
* `plotMovie/` -- renders a run as an animation, frame per day (`generate_frames.py`,
  `plotCases.py`, `createMov.sh`).
* `cformat/` -- the pinned `clang-format` binary, style file and Docker image used to format the
  C++ sources.
* `hardware_usage/` -- profiling helpers for AMD CPU and NVIDIA GPU runs.
* `mps/` -- launcher for running with NVIDIA's Multi-Process Service.
* `tests/` -- helper scripts for the regression tests.
