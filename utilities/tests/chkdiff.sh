#!/bin/bash

# Bash script to compare ExaEpi simulation results in current directory with those
# in a reference directory.
#
# Usage: use when output is *not* in HDF5 format
#
#   /path/to/chkdiff.sh -r /reference/directory [-a /path/to/amrex_fcompare]
#
# Note: By default, it looks for amrex_fcompare in the default path. It will also
# look for environment variable AMREX_FCOMPARE that contains path to amrex_fcompare.
# Specify path to amrex_fcompare using the "-a" flag.
#
# + Compares output*.dat using the diff command
# + Compares cases* using the diff command
# + Compares plt* files

compare_files() {

    refdir=$1
    fname=$2
    diff_fname=$3
    diff_cmd=$4

    rm -rf $diff_fname
    result=$($diff_cmd $fname $refdir/$fname 2>&1 >> $diff_fname)
    if [ -z "$result" ]; then
        if [ -s "$diff_fname" ]; then
            echo "    **FAILED** (files differ)"
        else
            echo "    passed"
        fi
    else
        echo "    **FAILED** (diff command failed)"
        echo $result
    fi
}

compare_amrex_files() {

    refdir=$1
    fname=$2
    diff_fname=$3
    diff_cmd=$4

    declare -a var_list=( "total"
                          "never_infected"
                          "infected"
                          "immune"
                          "susceptible"
                          "unit"
                          "FIPS"
                          "Tract"
                          "comm")

    rm -rf $diff_fname
    result=$($diff_cmd $fname $refdir/$fname 2>&1 >> $diff_fname)
    n_fail=0
    chk_abort=$(echo $result |grep "amrex::Abort")
    if [ ! -z "$chk_abort" ]; then
        ((n_fail+=1))
        echo $result
    fi
    chk_notfound=$(echo $result |grep -i "not found")
    if [ ! -z "$chk_notfound" ]; then
        ((n_fail+=1))
        echo $result
    fi

    # Check if plotfiles agree - if so, it's a pass
    chk_agree=$(grep "PLOTFILE AGREE" $diff_fname 2>/dev/null)
    if [ ! -z "$chk_agree" ]; then
        echo "    passed"
        return
    fi

    # Extract only the comparison section (before TinyProfiler output)
    # This avoids false positives from profiling statistics
    comparison_output=$(sed -n '1,/^TinyProfiler\|^Pinned Memory\|^Cpu Memory/p' $diff_fname | head -n -1)

    for var in ${var_list[@]}; do
        diff_var=$(echo "$comparison_output" | grep "\b$var\b")
        for i in {1..9}; do
            tmp=$(echo $diff_var |grep "$i")
            if [ ! -z "$tmp" ]; then
                ((n_fail+=1))
                echo "DIFFERENCE: $tmp"
                break
            fi
        done
    done
    if [[ $n_fail -ne 0 ]]; then
        echo "    **FAILED**"
    else
        echo "    passed"
    fi
}

amrex_diff="amrex_fcompare"
if [ ! -z "$AMREX_FCOMPARE" ]; then
    amrex_diff=$AMREX_FCOMPARE
fi
while getopts r:a: flag
do
    case "${flag}" in
        r) refdir=${OPTARG};;
        a) amrex_diff=${OPTARG};;
    esac
done
if [ -z "$refdir" ]; then
    echo "Error: no reference directory specified (-r /path/to/reference/directory)"
    exit
fi

echo "amrex_fcompare is $amrex_diff"
rm -rf diff*.*

diff_cmd="diff"
for f in output*.dat
do
    echo "Comparing $f...  "
    diff_file="diff_$f"
    compare_files $refdir $f $diff_file $diff_cmd
done

diff_cmd="diff"
for f in cases*
do
    echo "Comparing $f...  "
    diff_file="diff_$f"
    compare_files $refdir $f $diff_file $diff_cmd
done

diff_cmd=$amrex_diff
for f in plt*
do
    echo "Comparing $f... "
    diff_file="diff_$f"
    compare_amrex_files $refdir $f $diff_file $diff_cmd
done
