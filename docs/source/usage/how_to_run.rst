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
An example `inputs` file can be bound at `ExaEpi/examples/inputs`. Below, we document
the runtime parameters than can be set in the inputs file.

The following are inputs for the overall simulation:

* ``agent.number_of_diseases`` (`integer`)
    The number of diseases to track. (Default is ``1``).
* ``agent.disease_names`` (vector of `strings`)
    Names of the diseases; the size of the vector must be the same as ``agent.number_of_diseases``.
    If unspecified, the disease names are set as ``default00``, ``default01``, ``...``.
* ``agent.ic_type`` (`string`: either ``"census"``)
    If ``"census"``, initial conditions will be read from the provided census data file.
* ``agent.census_filename`` (`string`)
    The path to the ``*.dat`` file containing the census data used to set initial conditions.
    Must be provided if ``ic_type`` is ``"census"``. Examples of these data files are provided
    in ``ExaEpi/data/CensusData``.
* ``agent.worker_filename`` (`string`)
    The path to the ``*.bin`` file containing worker flow information.
    Must be provided if ``ic_type`` is ``"census"``. Examples of these data files are provided
    in ``ExaEpi/data/CensusData``.
* ``agent.airports_filename`` (`string`)
    The path to the ``*.dat`` file containing available airports and the counties they serve.
* ``agent.air_traffic_filename`` (`string`)
    The path to the ``*.dat`` file containing passenger flows among airports.
* ``agent.initial_case_type`` (vector of `strings`: each of which is either ``"random"`` or ``"file"``)
    The size of the vector must be the same as ``agent.number_of_diseases``.
    If ``random``, ``agent.num_initial_cases`` must be set.
    If ``file``, ``agent.case_filename`` must be set. Must be provided if ``ic_type`` is ``"census"``.
* ``agent.case_filename`` (`string`)
    When ``agent.number_of_diseases = 1``: The path to the ``*.cases`` file containing the initial case
    data to use. Must be provided if ``initial_case_type`` is ``"file"``. Examples of these data files
    are provided in ``ExaEpi/data/CaseData``.
* ``agent.case_filename_[disease name]`` (`string`)
    When ``agent.number_of_diseases > 1``:
    The path to the ``*.cases`` file containing the initial case data for ``[disease name]`` to use,
    where ``[disease name]`` is from the list of names specified in ``agent.disease_names`` (or the
    the default value).
    Must be provided if ``initial_case_type`` for ``[disease name]`` is ``"file"``;
    Examples of these data files are provided in ``ExaEpi/data/CaseData``.
* ``agent.num_initial_cases`` (int)
    When ``agent.number_of_diseases = 1``:  The number of initial cases to seed. Must be provided if
    ``initial_case_type`` is ``"random"``.
* ``agent.num_initial_cases_[disease name]`` (int)
    When ``agent.number_of_diseases > 1``:  The number of initial cases for to seed for ``[disease name]``,
    where ``[disease name]`` is any of the names specified in ``agent.disease_names`` (or the
    the default value).
    Must be provided if ``initial_case_type`` is ``"random"`` for ``[disease name]``.
* ``agent.nsteps`` (`integer`)
    The number of days to simulate.
* ``agent.plot_int`` (`integer`)
    The number of time steps between successive plot file writes.
* ``agent.random_travel_int`` (`integer`)
    The number of time steps between long distance travel events - note that this is
    currently only meaningful if `ic_type` = ``"census"``.
* ``agent.air_travel_int`` (`integer`)
    The number of time steps between air travel events - similar to random travel, this option is
    currently only meaningful if `ic_type` = ``"census"``.
* ``agent.aggregated_diag_int``
    The number of time steps between writing aggregated data, for example wastewater data.
* ``agent.aggregated_diag_prefix`` (`string`)
    Prefix to use when writing aggregated data. For example, if this is set to `cases`, the
    aggregated data files will be named `cases000010`, etc.
* ``agent.seed`` (`long integer`)
    Use this to specify the random seed to use for the run.
* ``agent.shelter_start`` (`integer`)
    Day on which to start shelter-in-place.
* ``agent.shelter_length`` (`integer`)
    Number of days shelter in-place-is in effect.
* ``agent.shelter_compliance`` (`float`)
    Fraction of agents that comply with shelter-in-place order.
* ``agent.symptomatic_withdraw`` (`integer`, default: 1)
    Whether or not to have symptomatic agents withdraw.
* ``agent.symptomatic_withdraw_compliance`` (`float`, default: 0.95)
    Compliance rate for agents withdrawing when they have symptoms. Should be 0.0 to 1.0.
* ``agent.student_teacher_ratios`` (`list of integers`, default: ``20 20 20 20 20 1000000000``)
    This option sets the desired student-teacher ratio for High School, Middle School, Elementary School in Neighborhood 1, Elementary School in Neighborhood 2, and Day Care, respectively. A large value of this ratio indicates that there should be 0 teachers in the associated school type (e.g., by default, there are no teachers assigned to Day Care).
* ``agents.size`` (`tuple of 2 integers`: e.g. ``(1, 1)``, default: ``(1, 1)``)
    This option is deprecated and will removed in a future version of ExaEpi. It controls
    the number of cells in the domain when running in `demo` mode. During actual usage,
    this number will be overridden and is irrelevant.
* ``agent.max_grid_size`` (`integer`, default: ``16``)
    This option sets the maximum grid size used for MPI domain decomposition. If set to
    ``16``, for example, the domain will be broken up into grids of `16^2` communities, and
    these grids will be assigned to different MPI ranks / GPUs.
* ``diag.output_filename`` (vector of `strings`, default: ``output.dat``, ``output_[disease name].dat``)
    Filename for the output data; the size of the vector must be the same as ``agent.number_of_diseases``.
    The default is ``output.dat`` for ``agent.number_of_diseases = 1`` and ``output_[disease name].dat``
    for ``agent.number_of_diseases > 1``, where ``[disease name]`` is from the list of names specified
    in ``agent.disease_names`` (or the default values).


The following inputs specify the transmission parameters:

* ``contact.pSC`` (`float`, default: 0.2)
    This is contact matrix scaling factor for schools.
* ``contact.pCO`` (`float`, default: 1.45)
    This is contact matrix scaling factor for communities.
* ``contact.pNH`` (`float`, default: 1.45)
    This is contact matrix scaling factor for neighborhoods.
* ``contact.pWO`` (`float`, default: 0.5)
    This is contact matrix scaling factor for workplaces.
* ``contact.pFA`` (`float`, default: 1.0)
    This is contact matrix scaling factor for families.
* ``contact.pBAR`` (`float`, default: -1.0)
    This is contact matrix scaling factor for bars or other large social gatherings.
    Setting this to < 0 turns this transmission off.

The following inputs specify the disease parameters:

* ``disease.nstrain`` (`integer`)
    The number of disease strains we're modeling.
* ``disease.p_trans`` (`list of float`, example: ``0.2 0.3``)
    These numbers are the probability of transmission given contact. There must be
    one entry for each disease strain.
* ``disease.p_asymp`` (`list of float`, example: ``0.4 0.4``)
    The fraction of cases that are asymptomatic. There must be
    one entry for each disease strain.
* ``disease.reduced_inf`` (`list of float`, example: ``0.75 0.75``)
    The relative infectiousness of asymptomatic individuals. There must be
    one entry for each disease strain.
* ``disease.vac_eff`` (`float`, example: ``0.4``)
    The vaccine efficacy - the probability of transmission will be multiplied by one minus this factor
* ``disease.immune_length_mean`` (`float`, default: 180)
    The mean amount of time *in days* agents are immune post-infection
* ``disease.immune_length_std`` (`float`, default: 60)
    The standard deviation associated with the above mean, i.e. the length is drawn from a normal distribution
* ``disease.latent_length_mean`` (`float`, default: ``3.0``)
    Mean length of time from exposure until agent becomes infectious in days.
* ``disease.infectious_length_mean`` (`float`, default: ``6.0``)
    Mean length of the infectious period in days. This counter starts once the latent phase is over.
* ``disease.incubation_length_mean`` (`float`, default: ``5.0``)
    Mean length of the time from exposure until symptoms develop in days.
* ``disease.latent_length_std`` (`float`, default: ``1.0``)
    Standard deviation of the latent period in days.
* ``disease.infectious_length_std`` (`float`, default: ``1.0``)
    Standard deviation of the infectious period in days.
* ``disease.incubation_length_std`` (`float`, default: ``1.0``)
    Standard deviation of the time until symptom development in days.
* ``disease.hospitalization_days`` (`list of float`, default: ``3.0 8.0 7.0``)
    Number of hospitalization days for age groups: under 50, 50-64, 65 and over.
* ``disease.CHR`` (`list of float`, default: ``.0104, .0104, .070, .28, 1.0``)
    Probability of hospitalization for age groups: under 5, 5-17, 18-29, 30-64, 65+
* ``disease.CIC`` (`list of float`, default: ``.24, .24, .24, .36, .35``)
    Probability of ICU for age groups: under 5, 5-17, 18-29, 30-64, 65+
* ``disease.CVE`` (`list of float`, default: ``.12, .12, .12, .22, .22``)
    Probability of ventilator for age groups: under 5, 5-17, 18-29, 30-64, 65+
* ``disease.CVF`` (`list of float`, default: ``.20, .20, .20, 0.45, 1.26``)
    Probability of death for age groups: under 5, 5-17, 18-29, 30-64, 65+

`Note`: for ``agent.number_of_diseases > 1``, the disease parameters that are common
to all the diseases can be specified as above. Any parameter that is `different for a specific disease`
can be specified as follows:

* ``disease_[disease name].[key] = [value]``

where ``[disease name]`` is any of the names specified in ``agent.disease_names`` (or the
default value), and ``[key]`` is any of the parameters listed above.

In addition to the ExaEpi inputs, there are also a number of runtime options that can be configured for AMReX itself. Please see <https://amrex-codes.github.io/amrex/docs_html/GPU.html#inputs-parameters>`__ for more information on these options.



