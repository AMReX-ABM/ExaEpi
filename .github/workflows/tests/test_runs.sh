#!/bin/bash

root_dir=$PWD
tests_dir=$root_dir/tests
data_dir=$root_dir/data
bin_dir=$root_dir/build/bin
tmp_dir=$root_dir/../tmp

exaepi_exec=$(ls $bin_dir/agent)
if [ -z "$exaepi_exec" ]; then
    echo "Error: ExaEpi executable not found in $bin_dir"
    exit 1
fi

# create tmp dir
if [ -d "tmp_dir" ]; then
    rm -rf $tmp_dir
fi
mkdir $tmp_dir
cd $tmp_dir

set -e

# create run directories and run tests
for i in $tests_dir/inputs*; do
    dirname=run.$i
    echo "Creating run directory $dirname"
    mkdir $dirname
    echo "  Entering $dirname"
    cd $dirname
#   copy input file
    cp $tests_dir/$i .
#   make symlinks to data files
    ln -sf $data_dir .
#   run ExaEpi
    echo "  Running $exaepi_exec with inputs $i..."
    mpiexec -n 4 $exaepi_exec $i
#   done
    echo "  Done."
    cd ..
    echo  ""
done

