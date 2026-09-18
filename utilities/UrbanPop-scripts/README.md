The scripts in this directory generate ExaEpi's UrbanPop input `.bin` files: the main converter,
the scripts producing the tables and files it requires, and a check on its output.

Scripts for analyzing, comparing and plotting ExaEpi *runs* live one level up in `utilities/`, and
are documented by that directory's README.

## Scripts

### `upop_to_exaepi.py`

The main script is `upop_to_exaepi.py` (polars-based; it replaced an earlier pandas
implementation of the same name, since its output format is the one `src/UrbanPopData.cpp`
actually reads). Run with `-h` to see the options:

```
usage: upop_to_exaepi.py [-h] [-c FILE] [--output OUTPUT] --upop_files
                         UPOP_FILES [UPOP_FILES ...] --lodes_files LODES_FILES
                         [LODES_FILES ...] --schools_file SCHOOLS_FILE
                         [--rseed RSEED]
                         [--county_adjacency_file COUNTY_ADJACENCY_FILE]
                         [--workgroup_sizes_file WORKGROUP_SIZES_FILE]
                         [--establishment_sizes_file ESTABLISHMENT_SIZES_FILE]
                         [--nborhood_size NBORHOOD_SIZE]
                         [--workgroup_size WORKGROUP_SIZE]
                         [--school_class_size SCHOOL_CLASS_SIZE]
                         [--school_class_size_min SCHOOL_CLASS_SIZE_MIN]
                         [--school_class_size_max SCHOOL_CLASS_SIZE_MAX]
                         [--college_instructional_fraction COLLEGE_INSTRUCTIONAL_FRACTION]

options:
  -h, --help            show this help message and exit
  -c FILE, --config FILE
                        Config file
  --output OUTPUT, -o OUTPUT
                        Output file
  --upop_files UPOP_FILES [UPOP_FILES ...], -f UPOP_FILES [UPOP_FILES ...]
                        UrbanPop feather files
  --lodes_files LODES_FILES [LODES_FILES ...], -l LODES_FILES [LODES_FILES ...]
                        LODES7 origin-destination (OD) files in CSV format
  --schools_file SCHOOLS_FILE, -s SCHOOLS_FILE
                        File containing schools data in CSV format
  --rseed RSEED, -r RSEED
                        Random seed
  --county_adjacency_file COUNTY_ADJACENCY_FILE
                        CSV of geoid,neighbor_geoid county-adjacency pairs
                        (see compute_county_adjacency.py). Used to bound
                        university students to their home county plus its
                        neighbors when their home county has no university
  --workgroup_sizes_file WORKGROUP_SIZES_FILE
                        Per-(state, NAICS) target workgroup size table (see
                        compute_workgroup_sizes.py). REQUIRED -- it caps how
                        strongly alloc_workers' NAICS-concentration bonus can
                        pull worker destinations away from real LODES flow
                        proportions, and falling back to one flat target for
                        every industry would do that silently.
  --establishment_sizes_file ESTABLISHMENT_SIZES_FILE
                        Per-(state, NAICS) establishment-size distribution
                        (see compute_workgroup_sizes.py). REQUIRED --
                        alloc_workers draws each workplace's size from this,
                        and sizing them all at the industry average instead
                        makes destination populations pile up at multiples of
                        that average, so a missing file is a hard error rather
                        than a silent fallback.
  --nborhood_size NBORHOOD_SIZE
                        Target residents per neighborhood. Sets how many home
                        neighborhoods a block group is split into, and how
                        many work neighborhoods a work block group is split
                        into
  --workgroup_size WORKGROUP_SIZE
                        Fallback target work-group size, used for any (state,
                        NAICS) pair missing from --workgroup_sizes_file, and
                        as the size of school admin groups
  --school_class_size SCHOOL_CLASS_SIZE
                        Fallback target students per class, used only for a
                        (school, grade) group that has students but no
                        identified teachers
  --school_class_size_min SCHOOL_CLASS_SIZE_MIN
                        Floor on a school group's average class size (bounds
                        its class count from above)
  --school_class_size_max SCHOOL_CLASS_SIZE_MAX
                        Cap on a school group's average class size (bounds its
                        class count from below)
  --college_instructional_fraction COLLEGE_INSTRUCTIONAL_FRACTION
                        Fraction of a college's employment treated as
                        instructional staff. College teacher counts come from
                        total college employment, not a faculty-specific
                        count, so they are scaled by this before being used as
                        a homeroom-instructor headcount
```

When using the config file option, specify configurations as section `main`, as shown in this
example file for New Mexico:

```
[main]
upop_files=base/35_NM/*.feather
lodes_files=../LODES7/nm_od_main_JT00_2019.csv.gz
schools_file=../EducationData/schools_with_geoids.csv
county_adjacency_file=county_adjacency.csv
output=urbanpop_nm
rseed=29
```

This is `data/UrbanPop/nm.cfg`, which ships with the repo alongside `ca.cfg`; both are meant to be
run from `data/UrbanPop/`, since their paths are relative to it. Neither sets the two size-table
options described below, even though both are required -- set those only to point at a pair other
than the repo's own. Neither sets the group-structure options either, so both get the defaults.

Any options specified on the command line after the config file will override settings in the config
file.

Worker, student, and teacher flows are generated from the LODES flows input and the schools input
(the LODES and schools files are required).

It also requires the two per-(state, NAICS) size tables written by `compute_workgroup_sizes.py`
(see below), which decide how many workplaces each destination has and how big each one is. Both
default to the copies in `data/UrbanPop/`, resolved from the script's own location rather than the
working directory, so they are found wherever the run starts and normally neither needs to be
given. A missing or mismatched pair is a hard error rather than a silent fallback.

`upop_to_exaepi.py` will generate a single binary output file, `<output>.bin`, containing both the
per-agent data and the block-group index (used for reading in parallel) in one combined format --
this is the only file `src/UrbanPopData.cpp` opens at runtime.

In addition, `upop_to_exaepi.py` will produces a file, `UrbanPopAgentStruct.H`, which is the C++
header file containing data structures needed in ExaEpi. This should be placed in the `src`, but it only
needs to be updated if the data fields generated by `upop_to_exaepi.py` have been modified.

That header also carries the `.bin` layout version (`BIN_FORMAT_VERSION` in `upop_to_exaepi.py`),
which ExaEpi checks against the file it opens and refuses if it does not match, along with a check
on the per-agent record size that catches a field being added or widened without the version being
bumped. So a `.bin` and the `src/UrbanPopAgentStruct.H` it was generated with have to be updated
together: an old `.bin` against new code aborts on startup rather than reading agents at the wrong
offsets.

### `group_assignment.py`

Not a command -- a module `upop_to_exaepi.py` imports. It assigns every agent the structural groups
it belongs to, which are then written into the `.bin`:

| attribute | what it is |
| --- | --- |
| `nborhood` | home neighborhood, assigned per household so a household is never split |
| `hh_cluster` | household cluster within the block group, for close-contact mixing |
| `work_nborhood` | work neighborhood -- the "workplace" tier, shared by a whole establishment |
| `workgroup` | the team within an establishment, at the industry's target size |
| `school_class` | class within a (block group, school, grade) group, or a negative sentinel for a teacher in a non-classroom admin pool |
| `school_class_group` | globally dense id for the actual class-sized mixing bucket |

ExaEpi used to draw all of these itself at init. Doing it here instead means a `.bin` fully
determines its own population structure, which buys three things:

* **Rank invariance.** The C++ drew from per-rank RNG streams, so the same population came out
  differently on a different number of MPI ranks -- measured on New Mexico, going from 1 to 4 ranks
  reassigned 65% of agents' home neighborhoods and 22% of their work-groups. Reading the values
  from the file cannot do that. Structure is also independent of `agent.seed` now.

* **Coherent work tiers.** `workgroup` and `work_nborhood` were independent uniform draws, so a
  worker's team and the workplace containing it were unrelated. They now come from one pass:
  establishments are drawn from the real CBP establishment-size distribution
  (`establishment_sizes_us.txt`), each lands in one work neighborhood, and each is split into teams
  at the target size from `workgroup_sizes_us.txt`.

* **Households intact by construction.** The C++ drew a neighborhood per *agent* and then ran a
  second GPU kernel that forced each family onto its last member's draw by scanning forward --
  correct only while agents stayed contiguous and ordered by (home cell, family), an invariant
  nothing checked.

Educators and agents who declared work-from-home are deliberately left out of the work tiers
(`workgroup` 0, work neighborhood = home neighborhood), as they were in the C++: educators already
mix through `school_class_group`, and a work-from-home agent never visits the workplace they were
assigned to.

### `get_schools.py`

Generates the schools file that `upop_to_exaepi.py` takes as `--schools_file`, from HIFLD files
containing data on private schools, public schools, childcare and colleges/universities, plus
block-group shapefiles to attach a GEOID to each school. An example config file is:

```
[main]
private_school_files=hifld-data-2024/Private_Schools_-7285710811296673603.csv
public_school_files=hifld-data-2024/Public_Schools_-7669544197405643438.csv
college_files=hifld-data-2024/Colleges_and_Universities_Campuses_-4919296714247390032.csv
childcare_files=hifld-data-2024/Child_Care_Centers.csv
census_bg_files=../US_Census_BlockGroups/*.shp
```

This will generate schools for all of the US, but the file can be used for any single state without
any issues.

Public schools can also be taken from NCES rather than HIFLD, via `--public_nces_school_file`, which
is the recommended source of the two.

### `compute_county_adjacency.py`

Writes the `geoid,neighbor_geoid` county-adjacency CSV that `upop_to_exaepi.py` takes as
`--county_adjacency_file`, from a county boundary shapefile (see Census data below). It is used to
bound university students to their home county plus its neighbors when their own county has no
university, rather than letting them be placed anywhere in the state.

Both of its paths default to repo-root-relative locations, so run it from the top of the repo:

```
python utilities/UrbanPop-scripts/compute_county_adjacency.py
```

That reads `data/US_2000_Counties/tl_2000_us_county.shp` (a national county shapefile; not one of
the per-state Census files under `data/US_2010_Census_County/`) and writes
`data/UrbanPop/county_adjacency.csv`. Like the size tables the result is national, so it only needs
generating once -- and the generated CSV is already in the repo, so normally this does not need
running at all.

### `compute_workgroup_sizes.py`

Derives two per-(state, NAICS-code) tables from the 2019 Census County Business Patterns (CBP)
survey, both written into `data/UrbanPop/`:

| File | What it holds | Read by |
| --- | --- | --- |
| `workgroup_sizes_us.txt` | one *target work-group size* per (state, NAICS): CBP's `employment / establishments`, capped at 86 | `upop_to_exaepi.py` |
| `establishment_sizes_us.txt` | the establishment-size *distribution* behind that average: CBP's nine employment-size bands, each as (establishments, employees) | `upop_to_exaepi.py` |

The difference between the two matters, because an average is a poor summary of a heavily
right-skewed quantity. California hospitals (NAICS 622) average 1110 employees, but that average
covers 27 establishments with under 5 people alongside 186 with over 1000. The two are used for
different things:

* The **distribution** sizes the individual workplaces -- both when deciding which destination a
  worker commutes to, and when `group_assignment.py` splits a destination's workers into actual
  establishments. Sizing every workplace at the average instead made each destination's demand for
  an industry an exact integer multiple of that average, so the resulting per-(block group, NAICS)
  populations piled up at 1x, 2x, 3x ... it.
* The **average** is the target team size: how many work-groups an establishment is split into.

Neither is read by ExaEpi any more. Both are consumed entirely in preprocessing, and what the C++
sees is the resulting per-agent `workgroup` id.

Usage:

```
# one-time (or when refreshing from the Census): download CBP and rebuild the derived cache
python compute_workgroup_sizes.py --refresh-cbp-cache

# normal use: read the cache already checked into the repo and rewrite both tables
python compute_workgroup_sizes.py
```

Both tables cover every state CBP publishes, so one pair is correct for any UrbanPop `.bin`,
including a multi-state or national build -- there is no need to regenerate them per state. They
change only when the CBP release, the NAICS code list in `src/UrbanPopAgentStruct.H`, or the
size/cap parameters change, so in practice this is run rarely. The script is deliberately
stdlib-only and takes seconds, unlike the main conversion pipeline.

Both tables are **required** by `upop_to_exaepi.py`, which fails rather than falling back to a flat
default -- a silent fallback produces a plausible-looking `.bin` that is only discovered to be wrong
when someone plots the group sizes. They also carry a matching `# generation-stamp:` header, hashed
from the inputs that produced them, and `upop_to_exaepi.py` refuses to run on a mismatched pair (a
stale one alongside a freshly generated one). Regenerating from unchanged inputs reproduces
byte-identical files, so these checked-in tables stay out of the diff unless something really
changed.

The cap of 86 on the target size follows the Epicast 2.0 paper, which limits work-group size "based
on studies of workplace contact patterns"; see `related/epicast.pdf`.

### `check_nt_dt.py`

Also provided is a script `check_nt_dt.py`, which compares the results generated from the LODES flows
and schools files with the UrbanPop day/night flows. It computes correlations between the various
outputs. Assuming that we have used `upop_to_exaepi.py` to generate an output `upop_nm_gen.csv` for
generated day/night data, and an output `upop_nm_up.csv` for data from separated UrbanPop day/night
files, an example config file for New Mexico is:

(Note: `upop_to_exaepi.py`'s current, polars-based implementation no longer has the
`--up_nt_dt_files` option that produces the `upop_nm_up.csv` side of this comparison -- that
existed only in the earlier pandas implementation it replaced. Generating that file currently
requires porting `process_upop_nt_dt` back in, or running this comparison against an older
checkout.)

```
[main]
generated_nt_dt_file=nm_gen_nt_dt.csv
urbanpop_nt_dt_file=nm_up_nt_dt.csv
schools_file=../../EducationData/schools_with_geoids.csv
lodes_files=../LODES7/nm_od_main_2019.csv
```

This will produce results like the following:

```
Options:
  config               None
  gen_file             upop_nm_gen.csv
  up_file              upop_nm_up.csv
  schools_file         ../EducationData/schools_with_geoids.csv
  lodes_files          ['../LODES7/nm_od_main_2019.csv']
  upop_files           []
Found 1873670 common ids between datasets
Correlations for night/day flows (generated vs UrbanPop):
  worker: correlation 0.862
    Total flows - Generated: 756314, UrbanPop: 755420
    Plot saved to flow_comparison_worker.png
  student: correlation 0.598
    Total flows - Generated: 458970, UrbanPop: 459339
    Plot saved to flow_comparison_student.png
  all: correlation 0.965
    Total flows - Generated: 1873670, UrbanPop: 1873670
    Plot saved to flow_comparison_all.png
Correlation of school populations per GEOID (generated vs UrbanPop): 0.827 458970 458970
Correlations with school populations for GEOIDS using "../EducationData/schools_with_geoids.csv":
  generated: 0.979 458970 481381
  Plot saved to school_population_comparison_generated.png
  UrbanPop: 0.837 458970 480650
  Plot saved to school_population_comparison_urbanpop.png
Loading LODES files
Loading ../LODES7/nm_od_main_2019.csv
Elapsed time for get_lodes_groups: 4.34 seconds, memory 0.09 G
Checking flows for generated workers
  Correlation: 0.881 (flows: 756314, LODES: 2712058)
  Plot saved to worker_flows_comparison_generated_vs_lodes.png
Checking flows for UrbanPop workers
  Correlation: 0.787 (flows: 755420, LODES: 2712058)
```

As can be seen, it has computed correlations between the separate UrbanPop day/night files
and the generated data for workers (0.862), students (0.598), and school populations per block
group (GEOID) (0.827). It also computes the correlations between the schools data and the UrbanPop
day/night data and the schools data and the generated data (0.837, 0.979). Finally, it computes the
correlations between the LODES data and the UrbanPop day/night data and the LODES data and the
generated data (0.881, 0.787). It also produces plots of all these results, with names as indicated
in the output.

Two ploting scripts are also provided.

## Data sources

The required sources of data are for schools, worker flows, and business establishment sizes.

### Education data

The HIFLD education data can be obtained from:

`https://hifld-geoplatform.hub.arcgis.com/search?groupIds=f16c582f00184cb094affff556fe57ee`

### LODES data

The LODES data can be obtained from:

`https://lehd.ces.census.gov/data/lodes/LODES7`

For compatibility, use the 2019 files (UrbanPop uses the 2010 Census data). For each state there
are several files of the form (e.g. for New Mexico):

`nm_od_main_JT0?_2019.csv.gz`

These are the main flows within the state. Then there are also files of the form:

`nm_od_aux_JT0?_2019.csv.gz`

These are for flows to/from the state to other states.

### County Business Patterns (CBP) data

Establishment counts and sizes per (state, NAICS code), used by `compute_workgroup_sizes.py` to
build the two size tables above. The 2019 state-level file is at:

`https://www2.census.gov/programs-surveys/cbp/datasets/2019/cbp19st.zip`

No API key is needed, and `compute_workgroup_sizes.py --refresh-cbp-cache` downloads and filters it
for you. The filtered result (`data/UrbanPop/cbp19st_derived.csv`) is checked into the repo, so the
download is only needed when refreshing from a newer Census release.

### Census data

The 2010 Census data should be used for compatibility with UrbanPop. The files can be found at:

```
https://www2.census.gov/geo/tiger/TIGER2010/STATE/2010/
```



