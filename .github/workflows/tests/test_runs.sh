#!/bin/bash

nproc=$1 # number of MPI ranks
nomp=$2  # number of OpenMP threads

root_dir=$PWD
tests_dir=$root_dir/tests
data_dir=$root_dir/data
bin_dir=$root_dir/build/bin
tmp_dir=$tests_dir/tmp

exaepi_exec=$(ls $bin_dir/agent)
if [ -z "$exaepi_exec" ]; then
    echo "Error: ExaEpi executable not found in $bin_dir"
    exit 1
fi

# create tmp dir
echo "Creating $tmp_dir"
if [ -d "tmp_dir" ]; then
    rm -rf $tmp_dir
fi
mkdir $tmp_dir

set -e

cd $tests_dir
# create run directories and run tests
for i in inputs*; do
    dirname=run.$i
    echo "Creating run directory $dirname in $tmp_dir"
    mkdir $tmp_dir/$dirname
    echo "  Entering $dirname"
    pushd $tmp_dir/$dirname
#   copy input file
    cp $tests_dir/$i .
#   make symlinks to data files
    ln -sf $data_dir .
#   run ExaEpi
    echo "  Running $exaepi_exec with inputs $i..."
    export OMP_NUM_THREADS=$nomp
    mpiexec -n $nproc $exaepi_exec $i
#   done
    echo "  Done."
    popd
    echo  ""
done

