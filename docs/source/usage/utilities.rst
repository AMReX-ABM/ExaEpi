Utilities
#########

Output Format
=============
ExaEpi writes data on two levels: individual data and community data.

Individual data fields
----------------------
Individual datasets have more fields written on day 0 that are constant across the simulation.
Data is either written as an ``int`` or a ``real``.

- Day 0 only (constant)

  - Int data:

    - ``age_group``: Ranges from 0 to 4, indicating age group of agent. Age ranges are 0-5, 5-17, 18-29, 30-64, and 65+, respectively.

    - ``family``: ID of family of agent.

    - ``home_i``: x-coordinate of home location on AMReX grid. Can be used to match particle to home community.

    - ``home_j``: y-coordinate of home location on AMReX grid. Can be used to match particle to home community.

    - ``work_i``: x-coordinate of work location on AMReX grid. Can be used to match particle to work community.

    - ``work_j``: y-coordinate of work location on AMReX grid. Can be used to match particle to work community.

    - ``nborhood``: Home neighborhood.

    - ``school``: Ranges from -1 to 5, indicating type of school. -1 indicates not of school age, 0 indicates of school age but not in school, 1 indicates high school, 2 indicates middle school, and 3-5 indicate different elementary schools.

    - ``work_nborhood``: Work neighborhood.

    - ``strain``: Marks which strain of disease (used for multiple diseases).

  - Real data:

    - ``latent_period``: Latent period length, i.e. time between exposure and infectiousness.

    - ``infectious_period``: Infections period length, i.e. time during which agent can spread disease

    - ``incubation_period``: Symptom development period length (known as incubation period in some contexts), i.e. time between exposure and symptoms appearing.

- Every day (time-varying)

  - Int data

    - ``withdrawn``: Marks whether the agent is withdrawn. 0 for not withdrawn, 1 for withdrawn.

    - ``status``: Ranges from 0 to 4, indicating disease status. Corresponds to never infected, infected, immune, susceptible, and dead, respectively.

    - ``symptomatic``: Ranges from 0 to 2, indicating symptomaticity. Corresponds to presymptomatic (not yet symptomatic but will be), symptomatic, and a symptomatic.

  - Real data

    - ``treatment_timer``: Time that agent has been in hospital.

    - ``disease_counter``: Time that agent has been infected.

    - ``infection_prob``: Probability that agent is infected at end of day.

Community data fields
---------------------
All data is written as a real, although all data should be integers. If any community does not exist, its ``unit``, ``FIPS``, ``Tract``, and ``comm`` values will be -1, while all agent counts will be 0.
Note: this data is specific to the ``agent.ic_type="census"``.

- ``total``: Total population in community.

- ``never_infected``: Number of never-infected agents in community.

- ``infected``: Number of infected agents in community.

- ``immune``: Number of immune (recently recovered from disease) agents in community.

- ``susceptible``: Number of susceptible (recovered from disease long enough ago to be able to be infected again) agents in community.

- ``unit``: Unit at cell.

- ``FIPS``: FIPS code (county code) of community.

- ``Tract``: Census tract of community.

- ``comm``: Community ID (unique).

Reading output
==============
Output is read in different ways, depending on I/O format selected.
If ``-DAMReX_HDF5`` flag is set to ``TRUE`` when compiling, HDF5 will be used.
Otherwise, the default AMReX plotfile will be written.

AMReX plotfile
--------------
Any library that has functionality for AMReX plotfiles can be used:
see `AMReX documentation <https://amrex-codes.github.io/amrex/docs_html/Visualization_Chapter.html#chap-visualization>`__ for more.

We suggest using yt with Python:

.. code-block:: python

  import yt
  from yt.frontends.boxlib.data_structures import AMReXDataset

  ds = AMReXDataset("plt00000")
  ad = ds.all_data()

  community_totals = ad["total"]
  statuses = ad["particle_status"]
  infection_probs = ad["particle_infection_prob"]

Note that individual-level fields are prefixed by ``particle_``, while community-level fields are as-is.

HDF5
----
The individual-level data is contained in files at ``plt*****/agents/agents.h5``, while the community-level data is conatined in files ``plt*****.h5``. Note that string attributes are written as byte strings, i.e. ``b'FIPS'`` instead of ``"FIPS"``.

Individual-level data
^^^^^^^^^^^^^^^^^^^^^
The data itself is located in the ``level_0`` group, separated by datatype. ``int`` data is at ``level_0/data:datatype=0`` and the ``real`` data is at ``level_0/data:datatype=1``.

Data is written agent-by-agent, so all fields for a given agent will be contiguous. The first two ``int`` components for any agent are its CPU and its particle ID on the CPU, while the first two ``real`` components for any agent are its x- and y-positions on the AMReX grid.

The names of other ``int`` components can be accessed as an attribute of the file named ``"int_component_x"``, where x ranges from 2 to the maximum number of components per agent. Not counting the first two non-labelled components, the number of ``int`` components is an attribute of the file named ``"num_component_int"``, so the total number of components is ``num_component_int + 2``. Thus, the size of the ``level_0/data:datatype=0`` dataset is N * (num_component_int + 2), where N is the number of agents.

The same scheme applies to ``real`` data, with every instance of ``int`` replaced with ``real``.

Community-level data
^^^^^^^^^^^^^^^^^^^^
Each field is written as its own dataset in the ``level_0`` group. The number of datasets can be accessed as an attribute of the file named ``"num_components"``, and the name of each dataset can be accessed as the attribute named ``"component_x"`` where x ranges from 0 to ``num_components - 1``, and the corresponding dataset is found at ``level_0/data:datatype=x``. The order of the data is the same within a plotfile, so one can match the community IDs to the FIPS codes by index, for example.

Python example
^^^^^^^^^^^^^^
We provide an example of parsing files using h5py and python:

.. code-block:: python

  import h5py

  with h5py.File("plt00000.h5", "r") as comm_file:
      # Get the datatype number for each field
      community_indicies = {}
      for i in range(comm_file.attrs["num_components"][0]):
          community_indices[comm_file.attrs["component_" + str(i)]] = i
      # community_indices is:
      # {b'total': 0, b'never_infected': 1, b'infected': 2, b'immune': 3,
      #  b'susceptible': 4, b'unit': 5, b'FIPS': 6, b'Tract': 7, b'comm': 8}

      # For example, get the totals
      community_totals = comm_file["level_0"]["data:datatype=" + str(community_indicies[b'total'])]

  with h5py.File("plt00000/agents/agents.h5", "r") as agent_file:
      # Get the index for each int field
      int_indicies = {}
      num_int_comps = agent_file.attrs["num_component_int"][0] + 2
      for i in range(2, num_int_comps):
          int_indicies[agent_file.attrs["int_component_" + str(i)]] = i

      int_data = agent_file["level_0"]["data:datatype=0"][()].reshape(-1, num_int_comps)
      statuses = int_data[:, int_indicies[b'status']]

      # Get the index for each real field
      real_indicies = {}
      num_real_comps = agent_file.attrs["num_component_real"][0] + 2
      for i in range(2, num_real_comps):
          real_indicies[agent_file.attrs["real_component_" + str(i)]] = i

      real_data = agent_file["level_0"]["data:datatype=1"][()].reshape(-1, num_real_comps)
      statuses = real_data[:, real_indicies[b'infection_prob']]

Processing scripts
==================
We provide a few processing scripts for visualization and data aggregation.

Visualization
-------------
There are some visualization scripts located in ``utilities/plotMovie``.
Primarily, ``generate_frames.py`` allows one to generate still frames from
either native AMReX output or HDF5 output, plotting infections (or an arbitrary
function on community-level data) spatially across a given map (as a .shx file).
Some of these shapefiles (for USA, California, and the Bay Area) are located in
the ``data/`` directory. See the script file and its comments for further details.

Data Processing
---------------
To aggregate data written to HDF5 output, whether by county or census tract,
we have some functions designed to aid processing in ``utilities/hfd5_process.py``.
These functions allow grouping of data and counting various conditions of individuals,
such as counting total number of infections in each census tract in each age group
over all days in a simulation. See the script file and its comments for further details.
