.. _usage_run:

Run ExaEpi
==========

In order to run a new simulation:

#. create a **new directory**, where the simulation will be run
#. make sure the ExaEpi **executable** is either copied into this directory or in your ``PATH`` `environment variable <https://en.wikipedia.org/wiki/PATH_(variable)>`__
#. add an **inputs file** and on :ref:`HPC systems <install-hpc>` a **submission script** to the directory
#. run

.. code-block:: bash

   cd <run_directory>

   # run with an inputs file:
   mpirun -np <n_ranks> ./agent <input_file>

On an :ref:`HPC system <install-hpc>`, you would instead submit the :ref:`job script <install-hpc>` at this point, e.g. ``sbatch <submission_script>`` (SLURM on Cori/NERSC) or ``bsub <submission_script>`` (LSF on Summit/OLCF).

Running ``./agent`` with no arguments (or ``./agent --help``) prints the full list of recognized
parameters and their current defaults, read live from the code -- this is the authoritative
reference and can't go stale. Below, we document those same parameters with more context on
what each one means.

Inputs Parameters
=================

Runtime parameters are specified in an `inputs` file, which is required to run ExaEpi.
Example `inputs` files can be found at `ExaEpi/examples/`. The file `inputs.defaults` lists
every recognized setting, set to its default value where one exists. Below, we document the
runtime parameters that can be set in the inputs file.

The following are inputs for the overall simulation:

* ``agent.number_of_diseases`` (`integer`, default ``1``)
    The number of diseases to track.
* ``agent.disease_names`` (`list of strings`, default ``default00``)
    Names of the diseases; the size of the vector must be the same as ``agent.number_of_diseases``.
    If unspecified, the disease names are set as ``default00``, ``default01``, ``...``.
* ``agent.nborhood_size`` (`int`, default ``500``)
    Target size of a neighborhood, for home and work communities.
* ``agent.workgroup_size`` (`int`, default ``20``)
    Target size of a workgroup, for work communities. Used as the fallback for any
    (state, NAICS) combination not covered by ``agent.workgroup_size_filename``.
* ``agent.urbanpop_filename`` (`string`)
    The path to the ``*.csv`` and ``*.idx`` files containing the UrbanPop data used to set initial conditions. For each input
    there should be two files, one with a ``.csv`` extension, and one with a ``.idx`` extension, both with the same name.
    Do not specify the extension in this parameter.
    Must be provided. Examples of these data files are provided in ``ExaEpi/data/UrbanPop``.
* ``agent.workgroup_size_filename`` (`string`, default ``""``)
    Optional path to a per-(state, NAICS-code) work-group target size table (see
    ``utilities/UrbanPop-scripts/compute_workgroup_sizes.py``). Any (state, NAICS) combination
    not listed in the file falls back to the flat ``agent.workgroup_size``. Leaving this empty
    (the default) makes every combination use that flat value.
* ``agent.size_scale_enabled`` (`bool`, default ``true``)
    Enables a population-size-based correction that keeps community/neighborhood transmission
    frequency-dependent (depending on local prevalence) rather than density-dependent
    (depending on the raw number of infectious agents present, which would otherwise scale with
    how populous a community happens to be).
* ``agent.airports_filename`` (`string`)
    The path to the ``*.dat`` file containing available airports and the counties they serve.
    Must be provided if ``agent.air_travel_int > 0``.
* ``agent.air_traffic_filename`` (`string`)
    The path to the ``*.dat`` file containing passenger flows among airports.
    Must be provided if ``agent.air_travel_int > 0``.
* ``agent.weather_int`` (`integer`, default ``-1``)
    The number of time steps between updates of weather-driven effects. Set to -1 to disable.
* ``agent.weather_filename`` (`string`)
    The path to the weather data file. Must be provided if ``agent.weather_int > 0``.
* ``agent.startdate`` (`string`, default ``""``)
    The simulation's start date, as ``YYYY-MM-DD``. Only meaningful if ``agent.weather_int > 0``.
* ``agent.nsteps`` (`integer`, default ``1``)
    The number of days to simulate.
* ``agent.plot_int`` (`integer`, default ``-1``)
    The number of time steps between successive plot file writes. Set to -1 to disable writing.
* ``agent.check_int`` (`integer`, default ``-1``)
    The number of time steps between successive checkfile writes. Set to -1 to disable writing.
* ``agent.random_travel_int`` (`integer`, default ``-1``)
    The number of time steps between random long distance travel events. Set to -1 to disable all random travel.
* ``agent.random_travel_prob`` (`float`, default ``0.0001``)
    Probability of an agent engaging in random travel in each event.
* ``agent.air_travel_int`` (`integer`, default ``-1``)
    The number of time steps between air travel events. Set to -1 to disable all air travel events.
* ``agent.aggregated_diag_int`` (`integer`, default ``-1``)
    The number of time steps between writing aggregated data, for example wastewater data. Set to -1 to disable writing.
* ``agent.aggregated_diag_prefix`` (`string`, default ``cases``)
    Prefix to use when writing aggregated data. For example, if this is set to `cases`, the
    aggregated data files will be named `cases000010`, etc.
* ``agent.restart`` (`string`, default ``""``)
    Name of the checkpoint file to restart from. If not present, the simulation will run from the beginning.
* ``agent.seed`` (`long integer`)
    Use this to specify the random seed to use for the run. If not set, AMReX's own default
    seeding is used.
* ``agent.fast`` (`bool`, default ``false``)
    Use faster, non-bitwise-reproducible implementations (e.g. binning) at the cost of exact
    reproducibility for a given random seed.
* ``agent.context_diag`` (`bool`, default ``false``)
    If true, attribute infections to interaction contexts (work, school, household, ...) and
    write per-context columns to the output file.
* ``agent.shelter_start`` (`integer`, default ``-1``)
    Day on which to start shelter-in-place. Disabled when set to -1.
* ``agent.shelter_length`` (`integer`, default ``0``)
    Number of days shelter-in-place is in effect.
* ``agent.shelter_compliance`` (`float`, default ``0.95``)
    Fraction of agents that comply with shelter-in-place order.
* ``agent.symptomatic_withdraw_compliance_day_0`` (`list of float`, default: ``0.3 0.3 0.3 0.3 0.3 0.3``)
    Compliance rate for agents withdrawing on day 0 when they have symptoms, per age group
    (u5, 5-17, 18-29, 30-49, 50-64, 65+). Should be 0.0 to 1.0. Set to 0 to disable withdrawal.
* ``agent.symptomatic_withdraw_compliance_day_1`` (`list of float`, default: ``0.8 0.6 0.5 0.5 0.5 0.5``)
    Compliance rate for agents withdrawing on day 1 when they have symptoms, per age group.
* ``agent.symptomatic_withdraw_compliance_day_2`` (`list of float`, default: ``0.9 0.8 0.7 0.7 0.7 0.7``)
    Compliance rate for agents withdrawing on day 2 (or later) when they have symptoms, per age group.
* ``agent.school_class_size`` (`integer`, default ``15``)
    Fallback target students-per-class, for a (community, school, grade) group with students
    but no identified teachers in the underlying data.
* ``agent.school_class_size_min`` (`integer`, default ``5``)
    Floor on average class size (bounds the derived class count from above).
* ``agent.school_class_size_max`` (`integer`, default ``50``)
    Cap on average class size (bounds the derived class count from below).
* ``agent.college_instructional_fraction`` (`float`, default ``0.1``)
    Correction factor applied to a college-level group's reported teacher/staff headcount
    (sourced from total employment, not a faculty-specific count) before it drives class count.
* ``agent.max_box_size`` (`integer`, default ``16``)
    This option sets the maximum box size used for MPI domain decomposition.
* ``diag.output_filename`` (`string`, default ``output.dat`` for a single disease,
    ``output_[disease name].dat`` for multiple diseases)
    Filename for the output data; the number of list elements must be the same as ``agent.number_of_diseases``.
    The default is ``output.dat`` for ``agent.number_of_diseases = 1`` and ``output_[disease name].dat``
    for ``agent.number_of_diseases > 1``, where ``[disease name]`` is from the list of names specified
    in ``agent.disease_names`` (or the default values).


The following inputs specify the disease parameters:


* ``disease.initial_case_type`` (`string`, default ``random``)
    The value can be ``random`` or ``file``.
    If ``random``, then ``disease.num_initial_cases`` must be set. If ``file``, then ``disease.case_filename`` must be set.
* ``disease.case_filename`` (`string`)
    The path to the ``*.cases`` file containing the initial case data for a single disease.
    Must be provided if ``initial_case_type`` is ``"file"``.
    Examples of these data files are provided in ``ExaEpi/data/CaseData``.
* ``disease.num_initial_cases`` (`int`, default ``0``)
    The number of initial cases to seed for a single disease. Must be provided if
    ``initial_case_type`` is ``"random"``. It can be set to 0 for no cases.
* ``disease.p_trans`` (`float`, default ``0.2``)
    Probability of transmission given contact.
* ``disease.p_asymp`` (`float`, default ``0.3``)
    The fraction of cases that are asymptomatic.
* ``disease.asymp_relative_inf`` (`float`, default ``0.7``)
    The relative infectiousness of asymptomatic individuals, from 0 to 1.
* ``disease.vac_eff`` (`float`, default ``0``)
    The vaccine efficacy - the probability of transmission will be multiplied by one minus this factor.
    `Vaccination is not yet implemented, so this factor must be left at 0`.
* ``disease.child_compliance`` (`float`, default ``0.5``)
    Compliance rate for children when schools are closed. This reduces the probability of
    transmission in the neighborhood.
    `Note`: currently has no effect regardless of what is set here -- ``DiseaseParm::initialize()``
    unconditionally overwrites this to ``0.5``.
* ``disease.child_hh_closure`` (`float`, default ``2.0``)
    Factor for increasing within-household transmission by children when schools are closed.
    `Note`: currently has no effect regardless of what is set here -- ``DiseaseParm::initialize()``
    unconditionally overwrites this to ``2.0``.
* ``disease.compare_to_epicast`` (`bool`, default ``false``)
    Sample latent/incubation/infectious periods from Epicast's fixed CDFs instead of the Gamma
    distributions below.
* ``disease.immune_length_alpha`` / ``disease.immune_length_beta`` / ``disease.immune_length_loc``
    (`float`, defaults ``1.0`` / ``1.0`` / ``10000.0``)
    Gamma-distribution alpha, beta, and location (shift) for the immunity length: the length of
    time in days that agents are immune to the disease after recovering from it. For a Gamma
    distribution, the mean is alpha*beta (plus loc) and the variance is alpha*beta^2.
* ``disease.latent_length_alpha`` / ``disease.latent_length_beta`` (`float`, defaults ``6.0`` / ``0.73``)
    Gamma-distribution alpha and beta for the latent length: the length of time in days until
    agents become infectious after exposure.
* ``disease.infectious_length_alpha`` / ``disease.infectious_length_beta`` / ``disease.infectious_length_loc``
    (`float`, defaults ``3.54`` / ``1.22`` / ``2.5``)
    Gamma-distribution alpha, beta, and location for the infectious length: the length of time
    in days that agents are infectious. This counter starts once the latent phase is over.
* ``disease.incubation_length_alpha`` / ``disease.incubation_length_beta`` / ``disease.incubation_length_loc``
    (`float`, defaults ``6.0`` / ``0.73`` / ``1.0``)
    Gamma-distribution alpha, beta, and location for the incubation length: the length of time
    in days after exposure until agents develop symptoms.
* ``disease.hospital_delay_length_alpha`` / ``disease.hospital_delay_length_beta`` / ``disease.hospital_delay_length_loc``
    (`float`, defaults ``0.1`` / ``0.1`` / ``1.0``)
    Gamma-distribution alpha, beta, and location for the hospital-admission delay: the length
    of time in days after developing symptoms that agents seek treatment.
* ``disease.hospital_stay_type`` (`string`, either ``constant`` or ``random``, default ``constant``)
    If ``constant``, all the agents in an age group will be in the hospital for a fixed number of days.
    This number is set by the ``disease.hospitalization_days`` parameter.
    If ``random``, the agents will draw a number of days from a Gamma distribution, with the parameters
    of that distribution depending on age. These parameters are set by ``disease.hospitalization_days_alpha``
    and ``disease.hospitalization_days_beta``.
* ``disease.hospitalization_days`` (`list of float`, default ``3 3 3 3 8 7``)
    Number of hospitalization days, by age group (u5, 5-17, 18-29, 30-49, 50-64, 65+).
    This parameter is only used if ``disease.hospital_stay_type`` is ``constant``.
* ``disease.hospitalization_days_alpha`` (`list of float`, default ``3 3 3 3 8 7``)
    Alpha parameter for the Gamma distribution for hospital stay length, by age group. For a
    Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
    This parameter is only used if ``disease.hospital_stay_type`` is ``random``.
* ``disease.hospitalization_days_beta`` (`list of float`, default ``1 1 1 1 1 1``)
    Beta parameter for the Gamma distribution for hospital stay length, by age group.
    This parameter is only used if ``disease.hospital_stay_type`` is ``random``.
* ``disease.xmit_work`` (`float`, default ``0.0575``)
    Transmission probability within a workgroup.
* ``disease.xmit_comm`` (`list of float`, default ``0.000021 0.000062 0.000165 0.000165 0.000165 0.000247``)
    Transmission probabilities at the community level, for both work and home locations,
    given the age group of the susceptible agent (u5, 5-17, 18-29, 30-49, 50-64, 65+).
* ``disease.xmit_comm_scale`` (`float`, default ``1``)
    Overall magnitude of community transmission, applied on top of the population-size scaling
    from ``agent.size_scale_enabled``. Does not affect ``disease.xmit_hood``.
* ``disease.xmit_hood`` (`list of float`, default ``0.000082 0.000247 0.00066 0.00066 0.00066 0.001``)
    Transmission probabilities at the neighborhood level, for both work and home locations,
    given the age group of the susceptible agent.
* ``disease.xmit_hh_adult`` (`list of float`, default ``0.3 0.3 0.4 0.4 0.4 0.4``)
    Transmission probabilities at the household level, where the infectious agent is an adult,
    given the age group of the susceptible agent.
* ``disease.xmit_hh_child`` (`list of float`, default ``0.6 0.6 0.3 0.3 0.3 0.3``)
    Transmission probabilities at the household level, where the infectious agent is a child,
    given the age group of the susceptible agent.
* ``disease.xmit_nc_adult`` (`list of float`, default ``0.132 0.132 0.165 0.165 0.165 0.165``)
    Transmission probabilities at the neighborhood cluster level in the home location, where the
    infectious agent is an adult, given the age group of the susceptible agent.
* ``disease.xmit_nc_child`` (`list of float`, default ``0.25 0.25 0.132 0.132 0.132 0.132``)
    Transmission probabilities at the neighborhood cluster level in the home location, where the
    infectious agent is a child, given the age group of the susceptible agent.
* ``disease.xmit_school`` (`list of float`, default ``0 0.0002 0.0024 0.0027 0.0088 0.0125``)
    Transmission probabilities within schools, where both the infectious and susceptible agents
    are children, given the school level (none, college, high, middle, elementary, daycare).
    The first entry is ignored and should always be set to 0.
* ``disease.xmit_school_a2c`` (`list of float`, default ``0 0.001 0.01 0.013 0.043 0.025``)
    Transmission probabilities within schools, where the infectious agent is an adult and the
    susceptible agent is a child, given the school level.
    The first entry is ignored and should always be set to 0.
* ``disease.xmit_school_c2a`` (`list of float`, default ``0 0.001 0.01 0.013 0.043 0.025``)
    Transmission probabilities within schools, where the infectious agent is a child and the
    susceptible agent is an adult, given the school level.
    The first entry is ignored and should always be set to 0.
* ``disease.xmit_school_scale`` (`float`, default ``1``)
    Overall scale factor applied to all ``disease.xmit_school*`` values above.
* ``disease.CHR`` (`list of float`, default ``0.0104 0.0104 0.070 0.28 0.28 1.0``)
    Probability of hospitalization when disease symptoms first appear, by age group.
* ``disease.CIC`` (`list of float`, default ``0.24 0.24 0.24 0.36 0.36 0.35``)
    Probability of moving from hospitalization to ICU, by age group.
* ``disease.CVE`` (`list of float`, default ``0.12 0.12 0.12 0.22 0.22 0.22``)
    Probability of being placed on a ventilator when already in ICU, by age group.
* ``disease.hospCVF`` (`list of float`, default ``0 0 0 0 0 0``)
    Probability of death when in hospital (non-ICU), by age group.
* ``disease.icuCVF`` (`list of float`, default ``0 0 0 0 0 0.26``)
    Probability of death when in the ICU (non-ventilated), by age group.
* ``disease.ventCVF`` (`list of float`, default ``0.20 0.20 0.20 0.45 0.45 1.0``)
    Probability of death when on a ventilator, by age group.

The following inputs specify the disease-coupling parameters. They are valid only when simulating more than one disease
(i.e., ``agent.number_of_diseases > 1``.

* ``disease_coupling.coimmunity_matrix`` (matrix of `float`, default identity matrix)
    Co-immunity matrix: co-immunity is the immunity that an agent has against a disease due to past infection with other
    disease(s). The number of rows and columns of this matrix must be the same as the number of diseases
    (``agent.number_of_diseases``).
* ``disease_coupling.cosusceptibility_matrix`` (matrix of `float`, default full matrix of ``1.0``)
    Co-susceptibility matrix: co-susceptibility is the factor why which an agent is more susceptible to a disease due to
    current infection with other disease(s). The number of rows and columns of this matrix must be the same as the number
    of diseases (``agent.number_of_diseases``).

`Note`: for ``agent.number_of_diseases > 1``, the disease parameters that are common
to all the diseases can be specified as above. Any parameter that is `different for a specific disease`
can be specified as follows:

* ``disease_[disease name].[key] = [value]``

where ``[disease name]`` is any of the names specified in ``agent.disease_names`` (or the
default value), and ``[key]`` is any of the parameters listed above.

ExaEpi also adjusts a small number of AMReX/particle defaults on startup (see ``main.cpp``,
``overrideAmrexDefaults()``); these can be overridden the same way as any other parameter:

* ``amrex.the_arena_is_managed`` (`bool`, default ``false``)
    Whether AMReX's default Arena uses managed memory.
* ``amrex.use_comms_arena`` (`bool`, default ``true``)
    Whether AMReX uses a dedicated comms arena.
* ``particles.do_tiling`` (`bool`, default ``true`` on CPU builds, ``false`` on GPU builds)
    Whether to tile particle iteration.

In addition to the ExaEpi inputs, there are also a number of other runtime options that can be configured for AMReX itself.
Please see `<https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__ for more information on these options.
