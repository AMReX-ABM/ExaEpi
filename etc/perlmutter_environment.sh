# please set your project account
export proj=m3623_g

# required dependencies
module load cmake
module load PrgEnv-gnu
module load cudatoolkit

# an alias to request an interactive batch node for one hour
#   for parallel execution, start on the batch node: srun <command>
alias getNode="salloc -N 1 --ntasks-per-node=1 -t 1:00:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A $proj"
# an alias to run a command on a batch node for up to 30min
#   usage: runNode <command>
alias runNode="srun -N 1 --ntasks-per-node=1 -t 0:30:00 -q interactive -C gpu --gpu-bind=single:1 -c 32 -G 1 -A $proj"

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

# necessary to use CUDA-Aware MPI and run a job
export CRAY_ACCEL_TARGET=nvidia80

# optimize CUDA compilation for A100
export AMREX_CUDA_ARCH=8.0

# compiler environment hints
export CC=cc
export CXX=CC
export FC=ftn
export CUDACXX=$(which nvcc)
export CUDAHOSTCXX=CC
