This is a demo for an Agent-based epidemiology code built using the AMReX framework.

For more information about AMReX:
    website: https://amrex-codes.github.io/
    documentation: https://amrex-codes.github.io/amrex/docs_html/
    source code: https://github.com/AMReX-Codes/amrex

## Building the code

This demo uses CMake version 3.14 or higher. To build it:
     mkdir build
     cd build
     cmake ..
     make -j8

To build with GPU support, use the CMake option `-DAMReX_GPU_BACKEND=CUDA` on NVIDIA GPU, 
`-DAMReX_GPU_BACKEND=SYCL` on Intel GPU. 
On AMD GPU, both GPU backend and GPU architecture should be provided, e.g. `-DAMReX_GPU_BACKEND=HIP -DAMReX_AMD_ARCH=gfx90a`.

To write output as (compressed) HDF5, use the `-DAMReX_HDF5=TRUE` CMake option.
Parallel HDF5 installation is required. On perlmutter, a conda environment is
available at:

    conda activate /global/common/software/m3623/exaepi

Compression level can be altered in src/IO.cpp file. The environment variable
HDF5_CHUNK_SIZE controls the chunk size if compression is used. A value of 
around 100,000 is recommended to start.

For convenience, a script for setting up the module environment without HDF5
for Perlmutter is provided in etc/perlmutter_environment.sh. To use it, do:

    source etc/perlmutter_environment.sh

A similar script for setting up the environment for Frontier can be found in etc/frontier_environment.sh.

## Running the code

Navigate to build/bin and run the executable using one of the "inputs.*" files in "examples".
Run `./agent` with no arguments (or `./agent --help`) to print the full list of recognized
parameters and their current defaults; `examples/inputs.defaults` lists the same information
as a ready-to-edit inputs file.

Each example points at an UrbanPop data file under `data/UrbanPop/`; the large ones are
committed gzip'd (`*.bin.gz`) to keep the repository small, so gunzip the one you need first,
e.g. `gunzip -k data/UrbanPop/urbanpop_nm.bin.gz`.

For example:
    cd build/bin
    gunzip -k ../../data/UrbanPop/urbanpop_nm.bin.gz
    ./agent ../../examples/inputs.nm

On Aurora at ALCF, if compiled with SYCL be sure to add "agent.fast=1" at the end of the command line to speed up the case initialization

For example: mpirun -np 1 ./agent ../../examples/inputs.ca agent.fast=1

If reproducibility is a must, we do a quick run for 0 day to checkpoint the initial condition.

mpirun -np 1 ./agent ../../examples/inputs.ca agent.fast=1 agent.nsteps=0 agent.check_int=1 

Subsequent runs will start from the checkpoint by adding "agent.restart=chk00000" to the end of the command line.

mpirun -np 1 ./agent ../../examples/inputs.ca agent.restart=chk00000


## Looking at the output

Running the code succesfully will create a number of "plt?????" files. You can visualize
these using the script at etc/plot.py. This will require the "yt" package to be installed:

    https://yt-project.org/

## Copyright Notice

ExaEpi Copyright (c) 2022, The Regents of the University of California,
through Lawrence Berkeley National Laboratory (subject to receipt of
any required approvals from the U.S. Dept. of Energy). All rights reserved.

If you have questions about your rights to use or distribute this software,
please contact Berkeley Lab's Intellectual Property Office at
IPO@lbl.gov.

NOTICE.  This Software was developed under funding from the U.S. Department
of Energy and the U.S. Government consequently retains certain rights.  As
such, the U.S. Government has been granted for itself and others acting on
its behalf a paid-up, nonexclusive, irrevocable, worldwide license in the
Software to reproduce, distribute copies to the public, prepare derivative
works, and perform publicly and display publicly, and to permit others to do so.
