#!/bin/bash

root_dir=$PWD
src_dir=$PWD/src

format_script="$root_dir/utilities/custom-clang-format.py"

n_fail=0

echo -n "Format checking using "

echo "Checking header files."
for f in $src_dir/*.H; do
    echo "    $f"
    python3 $format_script < $f > $f.tmp
    diff_output=$(diff $f $f.tmp)
    if [[ ! -z "$diff_output" ]]; then
        ((n_fail+=1))
        echo "$f does not follow the ExaEpi style guide."
        echo "--"
        echo "Output of diff $f $f.tmp:"
        diff $f $f.tmp
        echo "--"
        echo ""
    fi
    rm $f.tmp
done

echo "Checking source files."
for f in $src_dir/*.cpp; do
    echo "    $f"
    python3 $format_script < $f > $f.tmp
    diff_output=$(diff $f $f.tmp)
    if [[ ! -z "$diff_output" ]]; then
        ((n_fail+=1))
        echo "$f idoes not follow the ExaEpi style guide"
        echo "--"
        echo "Output of diff $f $f.tmp:"
        diff $f $f.tmp
        echo "--"
        echo ""
    fi
    rm $f.tmp
done

if [[ $n_fail -gt 0 ]]; then
  echo "$n_fail files failed style check."
  exit 1
fi
