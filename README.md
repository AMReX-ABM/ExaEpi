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
`-DAMReX_GPU_BACKEND=SYCL` on Intel GPU, and `-DAMReX_GPU_BACKEND=HIP` on AMD GPU.

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

## Running the code

Navigate to build/bin and run the executable using one of the "inputs" files in "examples".

For example:
    cd build/bin
    ./agent ../../examples/inputs

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
