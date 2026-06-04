Input Data
##########

ExaEpi reads its input data from the ``data/`` directory in the repository. This
page describes the datasets and how they are organized; the hospital-capacity
data is documented in full, and the other datasets are summarized (their detailed
formats can be filled in over time).

Directory layout
================

- ``data/CensusData/`` -- demographic input for census initialization
  (``<region>.dat`` files such as ``CA.dat``, ``MA.dat``, ``US.dat``), plus
  airport (``*_airports.dat``) and air-traffic data. Each demographic file lists,
  per unit (a census tract), the population, number of day-workers, county FIPS,
  census tract number, and age/household breakdowns. Communities of ~2000 people
  are created within each unit.
- ``data/CaseData/`` -- initial-case seed files (``*.cases``), keyed by county
  FIPS, used to seed infections.
- ``data/EducationData/`` -- school assignment data (e.g.
  ``schools_with_geoids.csv.gz``).
- ``data/UrbanPop/`` -- UrbanPop synthetic-population binaries for UrbanPop
  initialization (which carries NAICS industry codes natively).
- ``data/HospitalData/`` -- acute-care hospital bed supply per county/tract for
  the medical-workers / hospital model (documented below).
- ``data/CA_2020_Counties/``, ``data/CA_2020_Census_Tracts/``,
  ``data/San_Francisco_Bay_Region_2020_Census_Tracts/`` -- Census TIGER/Line
  shapefiles used by the plotting utilities and by the hospital-data preprocessing
  (to assign hospitals to census tracts).
- ``data/<region>_CY<NN>AirTraffic.dat`` -- air-traffic matrices for the travel
  model.

Air-traffic, census, case, and education data formats are not yet documented in
detail here.

Hospital-capacity data
======================

The medical-workers / hospital model can set each community's staffed-bed supply
from real hospital data instead of a uniform per-capita density. The data lives in
``data/HospitalData/`` as small whitespace-delimited ``.dat`` files, one per region
and year, and is built from public sources by
``utilities/build_hospital_data.py``. See ``data/HospitalData/README.md`` for a
quickstart.

Sources
-------

- **HHS "COVID-19 Reported Patient Impact and Hospital Capacity by Facility"**
  (HealthData.gov dataset ``anag-cw7u``): facility-level **staffed** and **ICU**
  beds reported weekly (2019-2024), so the bed supply can be tied to a specific
  year. This is the default source (``--source hhs``).
- **HIFLD "Hospitals"** (Homeland Infrastructure Foundation-Level Data): hospital
  point locations, licensed bed counts, facility type, and county FIPS. Required
  for tract-level placement (``--source hifld --level tract``), where hospital
  points are spatially joined to census tracts.

Only acute-care facilities that admit infectious-disease inpatients are kept
(``STATUS = OPEN`` and ``TYPE`` in ``GENERAL ACUTE CARE`` / ``CRITICAL ACCESS``);
psychiatric, rehabilitation, children's-only and military facilities are excluded.

File format
-----------

Each file has ``#``-comment provenance lines, a record count, then one row per
geography::

    # provenance ...
    <N>
    FIPS  beds  icu_beds  n_hospitals                               (county level)
    FIPS  TRACT  beds  icu_beds  n_hospitals  hosp_FIPS  hosp_TRACT (tract level)

- ``FIPS`` is the integer county FIPS (e.g. ``6037`` for Los Angeles County),
  matching the FIPS in the demographic ``.dat`` files. ``TRACT`` is
  ``int(TRACTCE)`` (e.g. ``400`` for tract ``000400``).
- ``beds`` is the staffed acute-care bed supply at full workforce strength;
  ``icu_beds`` is staffed adult ICU beds.
- ``hosp_FIPS`` / ``hosp_TRACT`` (tract level) is the hospital tract that this
  tract's patients are routed to -- itself if it has a hospital, otherwise the
  nearest one.

Generating the data
-------------------

Install the dependencies and run the script (from ``data/HospitalData/``)::

    pip install -r ../../utilities/requirements-hospital-data.txt

    # County-level staffed beds from HHS, tied to a reporting week
    ../../utilities/build_hospital_data.py --state CA --year 2021 --source hhs \
        --download --hhs-week 2021-12-26 --out CA_hospitals_2021.dat

``--state US`` builds the whole country; ``--counties <FIPS ...>`` restricts to a
metro area (e.g. the San Francisco Bay Area). Tract-level files additionally need a
TIGER tract shapefile and ``geopandas``/``scipy``. Re-run with a new ``--year`` and
matching week to update periodically.

Using the data in ExaEpi
-------------------------

Enable the medical-workers model and point it at a file (census initialization
only)::

    agent.model_medical_workers = true
    hospital_model.use_HIFLD_HHS_data = true
    hospital_model.hospital_data_file = data/HospitalData/CA_hospitals_2021.dat

The model uses the data according to its granularity:

- **County-level** data sets each community's bed supply to its county's real bed
  supply, apportioned by population. This captures real between-county bed-density
  variation; patients are treated in their home community.
- **Tract-level** data places beds at the tracts that actually have hospitals and
  routes each community's patients to its nearest hospital tract, reusing the same
  agent-movement machinery as the home/work commute. Hospitals therefore cluster
  where they exist, and capacity strain is shared across each hospital's catchment.

See :doc:`how_to_run` for the full list of ``hospital_model.*`` input parameters.
