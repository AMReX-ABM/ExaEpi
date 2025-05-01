#!/usr/bin/env python

import sys
import os
import subprocess as proc

# This script fixes an issue where clang-format puts the AMREX_ specifiers on the same line as the function name.
# There is no option to not do this, so this script first turns those into comments, runs clang-format, and then
# turns them back into specifiers.
# A similar approach could be used to fix other clang-format limitations

# first run to ensure image is downloaded
os.system("docker run xianpengshen/clang-tools:19 clang-format")

content = sys.stdin.read()
content = content.replace("AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE\n", "/*AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE*/\n")
content = content.replace("AMREX_GPU_DEVICE AMREX_FORCE_INLINE\n", "/*AMREX_GPU_DEVICE AMREX_FORCE_INLINE*/\n")
# cmd = ["clang-format", "-i", "--style=file:.clang-format"]
cmd = ["docker", "run", "-i", "-v", ".:/host", "xianpengshen/clang-tools:19", "clang-format", "--style=file:/host/.clang-format"]
proc_obj = proc.Popen(cmd, stdin=proc.PIPE, stdout=proc.PIPE, stderr=proc.PIPE, text=True)
proc_obj.stdin.write(content)
content, error = proc_obj.communicate()
content = content.replace("/*AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE*/\n", "AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE\n")
content = content.replace("/*AMREX_GPU_DEVICE AMREX_FORCE_INLINE*/\n", "AMREX_GPU_DEVICE AMREX_FORCE_INLINE\n")
content = content.replace(" AMREX_GPU_DEVICE(", " AMREX_GPU_DEVICE (")

if proc_obj.returncode != 0:
    print("Process running clang-format returned", proc_obj.returncode, file=sys.stderr)

if len(error) > 0:
    print("ERROR running clang-format", file=sys.stderr)
    print(error, file=sys.stderr)
else:
    sys.stdout.write(content)
