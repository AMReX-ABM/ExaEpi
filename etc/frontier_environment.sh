# please set your project account
export proj=MED137

# required dependencies
module load craype-accel-amd-gfx90a
module load rocm

# an alias to request an interactive batch node for one hour
alias getNode="salloc -N 1 -t 1:00:00 -A $proj"

# GPU-aware MPI
export MPICH_GPU_SUPPORT_ENABLED=1

#Enable automatic page migration
export HSA_XNACK=1
