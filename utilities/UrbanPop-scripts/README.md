The scripts in this directory are for processing UrbanPop files into a format that is usable by ExaEpi.

## Scripts

The script `gen_nt_dt.py` must always be run first, and it will generate a nightime/daytime feather file that will be
used by `convert_urbanpop_to_exaepi.py`. This latter file takes as input a set of UrbanPop feather files, census block
group files, and daytime/nightime files. It merges these all together to produce three files: `urbanpop_<output>.csv`,
which contains the list of agent data, `urbanpop_<output>.idx`, which contains the indexes into the first file, for
reading in parallel, and `UrbanPopAgentStruct.H`, which contains the C++ header file containing data structures needed in
ExaEpi. This should be placed in the `src`, but it only needs to be updated if the data fields have changed.

A config file or options can be passed to `convert_urbanpop_to_exaepi.py`. This is an example of a config file for New
Mexico:

```
[main]
output=urbanpop_nm_new
files=data/35_NM/syp*.feather
shape_files=../../NM_2010_Census_BlockGroups/*.shp
day_night_files=nm_nt_dt.feather
```

Currently, UrbanPop nightime/daytime files are not available for all locations. If they are available, they can be used
in the `gen_nt_dt.py` script. If not, `gen_nt_dt.py` requires as input the UrbanPop feather files, LODES files containing
origin/destination flows, and a schools file, containing information on school sizes and locations. Here's an example
config file for New Mexico is:

```
[main]
urbanpop_files=data/35_NM/syp*.feather
lodes_files=../../LODES7/nm_od_main_2019.csv
schools_file=../../EducationData/schools_with_geoids.csv
output_file=nm
rseed=29
```

To generate the schools data, use the script `get_schools.py`. This takes as input HIFLD files containing data on
private schools, public schools, childcare and colleges/universities. An example config file for New Mexico is:

```
[main]
private_school_files=hifld-data-2024/Private_Schools_-7285710811296673603.csv
public_school_files=hifld-data-2024/Public_Schools_-7669544197405643438.csv
college_files=hifld-data-2024/Colleges_and_Universities_Campuses_-4919296714247390032.csv
childcare_files=hifld-data-2024/Child_Care_Centers.csv
census_bg_files=../US_Census_BlockGroups/*.shp
```

Also provided is a script `check_nt_dt.py`, which compares the results generated from the LODES flows and schools files with the
UrbanPop nighttime/daytime flows. It computes correlations. An example config file for New Mexico is:

```
[main]
generated_nt_dt_file=nm_gen_nt_dt.csv
urbanpop_nt_dt_file=nm_up_nt_dt.csv
schools_file=../../EducationData/schools_with_geoids.csv
```


## Data sources

### Education data

The HIFLD education data can be obtained from:

`https://hifld-geoplatform.hub.arcgis.com/search?groupIds=f16c582f00184cb094affff556fe57ee`

### LODES data

The LODES data can be obtained from:

`https://lehd.ces.census.gov/data/lodes/LODES7`

For compatibility, use the 2011 files (UrbanPop uses the 2010 Census data). For each state, there are several files,
of the form (e.g. for New Mexico):

`nm_od_main_JT0?_2011.csv.gz`

These are the main flows within the state. Then there are also files of the form:

`nm_od_aux_JT0?_2011.csv.gz`

These are for flows to/from the state to other states.

### Census data

The 2010 Census data should be used for compatibility with UrbanPop. The files can be found at:

```
https://www2.census.gov/geo/tiger/TIGER2010/STATE/2010/
```



