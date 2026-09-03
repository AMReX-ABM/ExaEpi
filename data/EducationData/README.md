## Data sources

School data comes from two sources, both processed by `utilities/UrbanPop-scripts/get_schools.py`
into `schools_with_geoids.csv`:

* **HIFLD** (`hifld-data-2024/`) -- public schools, private schools, colleges/universities, and
  childcare centers. This is the only source for private schools, colleges, and childcare -- there
  is no alternative for those. It is 2024 data, i.e. current, but that's not actually an advantage
  here (see below).
* **NCES** (`nces-data/Public_School_Characteristics_2018-19.csv`) -- an alternative source for
  public schools only, from the 2018-19 school year.

`schools_with_geoids.csv` is generated using NCES for public schools (the recommended default --
see `nm-schools.cfg`), with HIFLD supplying everything else. `schools_with_geoids.hifld.csv` and
`schools_with_geoids.nces.csv` are kept alongside it as the two variants, for comparison.

### Why NCES over HIFLD for public schools

* **Vintage**: the rest of this pipeline already deliberately uses ~2019-vintage data paired with
  the fixed 2010 census geography -- see the LODES note in `utilities/UrbanPop-scripts/README.md`
  ("use the 2019 files ... for compatibility"). NCES 2018-19 matches that convention; HIFLD 2024 is
  a 5-6 year mismatch against everything else in the pipeline (LODES flows, UrbanPop demographics).
* **Virtual/online school detection**: NCES has an authoritative `VIRTUAL` field, so
  `get_nces_public_schools()` can precisely drop statewide virtual/cyber schools (see below).
  HIFLD has no equivalent field, so its filtering falls back to an incomplete name-based match that
  misses the largest offenders (e.g. a 20,355-student HIFLD-sourced "school" that is actually a
  statewide cyber charter is *not* caught by the name filter, but *is* caught by NCES's `VIRTUAL`
  flag on the equivalent record).

Comparing the two sources directly on the schools they share (matched by NCES school ID):
enrollment correlates strongly (Pearson r = 0.96), geoid assignment agrees 97.8% of the time, and
teacher counts correlate at r = 0.94 once known bad records are excluded (see below). The
remaining differences are mostly attributable to the 6-year vintage gap, not a processing bug.

To regenerate using HIFLD for public schools instead, edit `nm-schools.cfg` (comment out
`public_nces_school_file`, uncomment `public_school_files`) or pass `--public_school_files`
directly on the command line -- see `get_schools.py -h`.

## Known data-quality issues, and how `get_schools.py` handles them

Both HIFLD and NCES contain real data-quality problems that `get_schools.py` corrects. This
section exists so future readers understand why that logic is there, rather than mistaking it for
noise.

### Missing childcare sizes (HIFLD)

HIFLD's `Child_Care_Centers.csv` uses `-999` as a sentinel for "population unknown" in 20.5% of
records nationally -- and for 11 states (including NM), *100%* of childcare records are missing.
`get_hifld_childcare()` used to backfill every missing value with a single flat national average
(84), which meant every childcare center in an affected state ended up with the exact same,
fabricated size. It now samples each missing value individually from the empirical distribution of
known sizes nationwide, preserving the same average while giving each center a realistic,
individual size.

### Implausible student/teacher ratios (HIFLD and NCES)

Both sources have a small number of records with a nonsensical students-per-teacher ratio -- e.g.
a HIFLD school reporting "1,961 students, 1 full time teacher", or an NCES record whose
`STUTERATIO` implies 51,080 teachers for 5,108 students. These are reporting errors, not real
schools. `invalidate_bad_teacher_ratios()` (for HIFLD public/private schools) and an equivalent
check on NCES's `STUTERATIO` field treat a ratio outside `[1, 100]` the same as a missing teacher
count, so `get_complete()`'s existing average-ratio backfill supplies a plausible estimate instead
of the garbage value.

This check is **not** applied to colleges. A university's `TOT_EMP` legitimately includes
non-teaching staff -- e.g. Ohio State's ~35,000 `TOT_EMP` includes its medical center's hospital
staff, and University of the People's ~1,682:1 ratio reflects its real, almost entirely
volunteer-faculty online model. Applying the same-shaped filter there would corrupt real data for
large research universities and non-traditional institutions, rather than fix an error.

### Virtual/online schools (no single physical location)

Some "schools" have enrollment in the thousands or tens of thousands but no single physical
building -- statewide virtual charter schools (e.g. Ohio Virtual Academy, Epic Charter School) and
fully-online universities (e.g. Western Governors University, University of Phoenix). Their
enrollment figures are real, but treating them as one physical school for ExaEpi's contact-mixing
model would manufacture a huge fake in-person contact hub that doesn't exist.

* For NCES public schools, `get_nces_public_schools()` drops records where the authoritative
  `VIRTUAL` field is `Full Virtual` or `Virtual with face to face options`.
* For HIFLD (public schools, private schools, and colleges), there is no such field, so
  `drop_virtual_schools()` matches school/institution names against a pattern of known keywords
  (`VIRTUAL`, `CYBER`, `CONNECTIONS ACADEMY`, `ONLINE`, etc). This is necessarily incomplete: some
  large virtual schools have names that don't contain any of these keywords (e.g. "Commonwealth
  Charter Academy" is a statewide PA cyber charter despite the name), and non-classroom-based
  independent-study charters (a handful of CA schools) don't match any pattern and aren't flagged
  as virtual by NCES either. This is the second reason NCES is preferred for public schools -- its
  `VIRTUAL` field catches cases the HIFLD name match cannot.

## Processing pipeline

The data files have lng/lat for all but the colleges. We use that data with the 2010 census
shapefiles to convert the lng/lat into a census block group. For colleges, we only have the
address, so we look that up using the census geocode service with vintage 2010. This is slow, so
we reuse the results of previous lookups -- the `batch.*.csv` files.
