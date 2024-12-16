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

Inputs Parameters
=================

Runtime parameters are specified in an `inputs` file, which is required to run ExaEpi.
Example `inputs` files can be bound at `ExaEpi/examples/`. The file `inputs.default` lists all of the settings,
set to the default values where appropriate. Below, we document the runtime parameters than can be set in the inputs file.

The following are inputs for the overall simulation:

* ``agent.number_of_diseases`` (`integer`, default ``1``)
    The number of diseases to track.
* ``agent.disease_names`` (`list of strings`, default ``default00``)
    Names of the diseases; the size of the vector must be the same as ``agent.number_of_diseases``.
    If unspecified, the disease names are set as ``default00``, ``default01``, ``...``.
* ``agent.ic_type`` (`string`, default ``"census"``)
    Can be either ``census`` or ``urbanpop``.
    If ``census``, initial conditions will be read from the provided census data file.
    If ``urbanpop``, initial conditions will be read from the provided UrbanPop data files.
* ``agent.census_filename`` (`string`)
    The path to the ``*.dat`` file containing the census data used to set initial conditions.
    Must be provided if ``ic_type = census``. Examples of these data files are provided
    in ``ExaEpi/data/CensusData``.
* ``agent.worker_filename`` (`string`)
    The path to the ``*.bin`` file containing worker flow information.
    Must be provided if ``ic_type = census``. Examples of these data files are provided
    in ``ExaEpi/data/CensusData``.
* ``agent.nborhood_size`` (`int`, default ``500``)
    Size of neighborhood for home and work communities.
* ``agent.workgroup_size`` (`int`, default ``20``)
    Size of workgroups for work communities.
* ``agent.urbanpop_filename`` (`string`)
    The path to the ``*.csv`` and ``*.idx`` files containing the UrbanPop data used to set initial conditions. For each input
    there should be two files, one with a ``.csv`` extension, and one with a ``.idx`` extension, both with the same name.
    Do not specify the extension in this parameter.
    Must be provided if ``ic_type = urbanpop``. Examples of these data files are provided in ``ExaEpi/data/UrbanPop``.
* ``agent.airports_filename`` (`string`)
    The path to the ``*.dat`` file containing available airports and the counties they serve. Currently this is implemented
    only for ``ic_type = census``.
* ``agent.air_traffic_filename`` (`string`)
    The path to the ``*.dat`` file containing passenger flows among airports. Currently this is implemented
    only for ``ic_type = census``.
* ``agent.initial_case_type`` (`list of strings`, default ``random``)
    The size of the list must be the same as ``agent.number_of_diseases``. The value can be ``random`` or ``file``.
    If ``random``, then ``agent.num_initial_cases`` must be set. If ``file``, then ``agent.case_filename`` must be set.
* ``agent.case_filename`` (`string`)
    The path to the ``*.cases`` file containing the initial case
    data for a single disease. Must be provided if ``initial_case_type`` is ``"file"``. Examples of these data files
    are provided in ``ExaEpi/data/CaseData``.
* ``agent.case_filename_[disease name]`` (`string`)
    The path to the ``*.cases`` file containing the initial case data for ``[disease name]``,
    where ``[disease name]`` is from the list of names specified in ``agent.disease_names`` (or
    the default value), for multiple diseases.
    Must be provided for each disease ``[disease name]`` where ``initial_case_type`` is ``"file"``.
    Examples of these data files are provided in ``ExaEpi/data/CaseData``.
* ``agent.num_initial_cases`` (`int`, default ``0``)
    The number of initial cases to seed for a single disease. Must be provided if
    ``initial_case_type`` is ``"random"``. It can be set to 0 for no cases.
* ``agent.num_initial_cases_[disease name]`` (`int``)
    The number of initial cases to seed for ``[disease name]``,
    where ``[disease name]`` is any of the names specified in ``agent.disease_names`` (or the
    the default value), for multiple diseases.
    Must be provided if ``initial_case_type`` is ``"random"`` for ``[disease name]``.
* ``agent.nsteps`` (`integer`, default ``1``)
    The number of days to simulate.
* ``agent.plot_int`` (`integer`, default ``-1``)
    The number of time steps between successive plot file writes. Set to -1 to disable writing.
* ``agent.random_travel_int`` (`integer`, default ``-1``)
    The number of time steps between random long distance travel events. Set to -1 to disable all random travel.
* ``agent.random_travel_prob`` (`float`, default ``0.0001``)
    Probability of an agent engaging in random travel in each event.
* ``agent.air_travel_int`` (`integer`, default ``-1``)
    The number of time steps between air travel events. Set to -1 to disable all air travel events. Currently this is implemented
    only for ``ic_type = census``.
* ``agent.aggregated_diag_int`` (`integer`, default ``-1``)
    The number of time steps between writing aggregated data, for example wastewater data. Set to -1 to disable writing.
* ``agent.aggregated_diag_prefix`` (`string`, default ``cases``)
    Prefix to use when writing aggregated data. For example, if this is set to `cases`, the
    aggregated data files will be named `cases000010`, etc.
* ``agent.seed`` (`long integer`, default ``0``)
    Use this to specify the random seed to use for the run.
* ``agent.shelter_start`` (`integer`, default ``-1``)
    Day on which to start shelter-in-place. Disabled when set to -1.
* ``agent.shelter_length`` (`integer`, default ``0``)
    Number of days shelter-in-place is in effect.
* ``agent.shelter_compliance`` (`float`, default ``0.95``)
    Fraction of agents that comply with shelter-in-place order.
* ``agent.symptomatic_withdraw_compliance`` (`float`, default: ``0.95``)
    Compliance rate for agents withdrawing when they have symptoms. Should be 0.0 to 1.0. Set it to 0 if not using withdrawal.
* ``agent.child_compliance`` (`float`, default ``0.95``)
    Compliance rate for children when schools are closed. This reduces the probability of transmission within
    neighborhood clusters, neighborhoods and communities.
* ``agent.child_hh_closure`` (`float`, default ``2``)
    Factor for increasing transmission by children witihn households when schools are closed.
* ``agent.student_teacher_ratio`` (`list of int`, default: ``0 15 15 15 15 15``)
    This option sets the desired student-teacher ratio for school levels (none, college, high, middle, elementary, daycare).
    The first entry is ignored and should always be set to 0. This option is only used with ``ic_type = census``.
* ``agent.max_box_size`` (`integer`, default ``16`` or ``500`` or ``100``)
    This option sets the maximum box size used for MPI domain decomposition. If set to
    ``16``, for example, for ``ic_type = census``, the domain will be broken up into boxes of `16^2` communities, and
    these boxes will be assigned to different MPI ranks / GPUs.
    The default for ``ictype = census`` is 16, and for ``ic_type = urbanpop`` it is 500 when using GPUs, and 100 otherwise.
* ``diag.output_filename`` (`list of strings`, default ``output.dat`` for a single disease,
    ``output_[disease name].dat`` for multiple diseases)
    Filename for the output data; the number of list elements must be the same as ``agent.number_of_diseases``.
    The default is ``output.dat`` for ``agent.number_of_diseases = 1`` and ``output_[disease name].dat``
    for ``agent.number_of_diseases > 1``, where ``[disease name]`` is from the list of names specified
    in ``agent.disease_names`` (or the default values).


The following inputs specify the disease parameters:

* ``disease.nstrain`` (`integer`, default ``1``)
    The number of disease strains being modeled. `In the current implementation, only the first strain is used`.
* ``disease.p_trans`` (`list of float`, default ``0.2``)
    Probability of transmission given contact. There must be one entry for each disease strain.
* ``disease.p_asymp`` (`list of float`, default ``0.4``)
    The fraction of cases that are asymptomatic. There must be one entry for each disease strain.
* ``disease.asymp_relative_inf`` (`list of float`, default ``0.75``)
    The relative infectiousness of asymptomatic individuals, from 0 to 1. There must be one entry for each disease strain.
    `This is not yet implemented`.
* ``disease.vac_eff`` (`float`, default ``0``)
    The vaccine efficacy - the probability of transmission will be multiplied by one minus this factor.
    `Vaccination is not yet implemented, so this factor must be left at 0`.
* ``disease.immune_length_alpha`` (`float`, default ``9.0``)
    Alpha parameter for the immunity length Gamma distribution. The immunity length is the length of time in days that agents
    are immune to the disease after recovering from it. For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.immune_length_beta`` (`float`, default ``20.0``)
    Beta parameter for the immunity length Gamma distribution. The immunity length is the length of time in days that agents
    are immune to the disease after recovering from it. For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.latent_length_alpha`` (`float`, default ``9.0``)
    Alpha parameter for the latent length Gamma distribution. The latent length is the length of time in days until agents become infectious after exposure.
    For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.latent_length_beta`` (`float`, default ``0.33``)
    Beta parameter for the latent length Gamma distribution. The latent length is the length of time in days until agents become infectious after exposure.
    For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.infectious_length_alpha`` (`float`, default ``36.0``)
    Alpha parameter for the infectious length Gamma distribution. The infectious length is the length of time in days that agents are infectious.
    This counter starts once the latent phase is over.
    For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.infectious_length_beta`` (`float`, default ``0.17``)
    Beta parameter for the infectious length Gamma distribution. The infectious length is the length of time in days that agents are infectious.
    This counter starts once the latent phase is over.
    For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.incubation_length_alpha`` (`float`, default ``25.0``)
    Alpha parameter for the incubation length Gamma distribution. The incubation length is the length of time in days after exposure until agents develop symptoms.
    For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.incubation_length_beta`` (`float`, default ``0.2``)
    Beta parameter for the incubation length Gamma distribution. The incubation length is the length of time in days after exposure until agents develop symptoms.
    For a Gamma distribution, the mean is alpha*beta and the variance is alpha*beta^2.
* ``disease.hospitalization_days`` (`list of float`, default ``3.0 8.0 7.0``)
    Number of hospitalization days for age groups: under 50, 50-64, 65 and over.
* ``disease.xmit_comm`` (`list of float`, default ``0.000018125 0.000054375 0.000145 0.000145 0.000145 0.0002175``)
    Transmission probabilities at the community level, for both work and home locations,
    given the age group of the susceptible agent (0-4, 5-17, 18-29, 30-49, 50-64).
* ``disease.xmit_hood`` (`list of float`, default ``0.0000725 0.0002175 0.00058 0.00058 0.00058 0.00087``)
    Transmission probabilities at the neighborhood level, for both work and home locations,
    given the age group of the susceptible agent (0-4, 5-17, 18-29, 30-49, 50-64)
* ``disease.xmit_hh_adult`` (`list of float`, default ``0.3 0.3 0.4 0.4 0.4 0.4``)
    Transmission probabilities at the household level, where the infectious agent is an adult,
    given the age group of the susceptible agent (0-4, 5-17, 18-29, 30-49, 50-64).
* ``disease.xmit_hh_child`` (`list of float`, default ``0.6 0.6 0.3 0.3 0.3 0.3``)
    Transmission probabilities at the household level, where the infectious agent is a child,
    given the age group of the susceptible agent (0-4, 5-17, 18-29, 30-49, 50-64).
* ``disease.xmit_nc_adult`` (`list of float`, default ``0.04 0.04 0.05 0.05 0.05 0.05``)
    Transmission probabilities at the neighborhood cluster level in the home location, where the infectious agent is an adult,
    given the age group of the susceptible agent (0-4, 5-17, 18-29, 30-49, 50-64).
* ``disease.xmit_nc_child`` (`list of float`, default ``0.075 0.075 0.04 0.04 0.04 0.04``)
    Transmission probabilities at the neighborhood cluster level in the home location, where the infectious agent is a child,
    given the age group of the susceptible agent (0-4, 5-17, 18-29, 30-49, 50-64).
* ``disease.xmit_school`` (`list of float`, default ``0 0.0315 0.0315 0.0375 0.0435 0.15``)
    Transmission probabilities within schools, where both the infectious and susceptible agents are children, given the
    school level (none, college, high, middle, elementary, daycare). The first entry is ignored and should always be set to 0.
* ``disease.xmit_school_a2c`` (`list of float`, default ``0 0.0315 0.0315 0.0375 0.0435 0.15``)
    Transmission probabilities within schools, where the infectious agent is an adult and the susceptible agent
    is a child, given the chool level (none, college, high, middle, elementary, daycare).
    The first entry is ignored and should always be set to 0.
* ``disease.xmit_school_c2a`` (`list of float`, default ``0 0.0315 0.0315 0.0375 0.0435 0.15``)
    Transmission probabilities within schools, where the infectious agent is a child and the susceptible agent
    is an adult, given the chool level (none, college, high, middle, elementary, daycare).
    The first entry is ignored and should always be set to 0.
* ``disease.CHR`` (`list of float`, default ``0.0104 0.0104 0.070 0.28 0.28 1.0``)
    Probability of hospitalization when disease symptoms first appear,
    for age groups: 0-4, 5-17, 18-29, 30-49, 50-64, 65 and over.
* ``disease.CIC`` (`list of float`, default ``0.24 0.24 0.24 0.36 0.36 0.35``)
    Probability of moving from hospitalization to ICU when symptoms first appear,
    for age groups: 0-4, 5-17, 18-29, 30-49, 50-64, 65 and over.
* ``disease.CVE`` (`list of float`, default ``0.12 0.12 0.12 0.22 0.22 0.22``)
    Probability of being placed on a ventilator when already in ICU, when symptoms first appear,
    for age groups: 0-4, 5-17, 18-29, 30-49, 50-64, 65 and over.
* ``disease.hospCVF`` (`list of float`, default ``0 0 0 0 0 0``)
    Probability of death when in hospital, for age groups: 0-4, 5-17, 18-29, 30-49, 50-64, 65 and over.
* ``disease.icuCVF`` (`list of float`, default ``0 0 0 0 0 0.26``)
    Probability of death when in hospital, in the ICU, for age groups: 0-4, 5-17, 18-29, 30-49, 50-64, 65 and over.
* ``disease.ventCVF`` (`list of float`, default ``0.20 0.20 0.20 0.45 0.45 1.0``)
    Probability of death when in hospital, on ventilator, for age groups: 0-4, 5-17, 18-29, 30-49, 50-64, 65 and over.

`Note`: for ``agent.number_of_diseases > 1``, the disease parameters that are common
to all the diseases can be specified as above. Any parameter that is `different for a specific disease`
can be specified as follows:

* ``disease_[disease name].[key] = [value]``

where ``[disease name]`` is any of the names specified in ``agent.disease_names`` (or the
default value), and ``[key]`` is any of the parameters listed above.

In addition to the ExaEpi inputs, there are also a number of runtime options that can be configured for AMReX itself.
Please see <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__ for more information on these options.



