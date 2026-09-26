#!/usr/bin/env bash
# Arm F: populations generated inside ExaEpi (agent.population_source = bundle, the C++ port) from
# the v2 bundle, seeds 1-10 as in arm E, 3 disease seeds each, same inputs as arm E. Each
# population is generated once by a CUDA build of this branch and written out with
# agent.write_population (byte-identical to the in-process population); the epidemics then run on
# the same binary as arms A and E, so the generator is the only difference from arm E.
set -u

ENS=/workspaces/ExaEpi/data/UrbanPop/experiments/ens
UP=/workspaces/ExaEpi/data/UrbanPop
GEN_AGENT=/workspaces/ExaEpi/.claude/worktrees/popgen-port/.build/CUDA/Release/bin/agent
AGENT=/workspaces/ExaEpi/.build/CPU/Release/bin/agent
DATA=/workspaces/ExaEpi/data
RUNS=$ENS/runs160E

for r in 1 2 3 4 5 6 7 8 9 10; do
    if [ ! -f "$ENS/pops/F_r$r.bin" ]; then
        g=$ENS/pops/F_gen_r$r
        mkdir -p "$g"
        cat > "$g/inputs" <<EOF
agent.nsteps = 1
agent.plot_int = -1
agent.aggregated_diag_int = -1
agent.weather_int = 1
agent.weather_filename = "$DATA/weatherData_NM_2015_2020.csv"
agent.startdate="2018-1-1"
agent.seed = 0
disease.initial_case_type = "file"
disease.case_filename = "$DATA/CaseData/nm-july4.cases"
agent.population_source = bundle
agent.population_bundle = "$UP/experiments/nm_v2.upb"
agent.population_seed = $r
agent.write_population = "$ENS/pops/F_r$r.bin"
amrex.the_arena_init_size = 1500000000
EOF
        (cd "$g" && "$GEN_AGENT" inputs > run.log 2>&1) && grep -h "Generated population" "$g/run.log" \
            || { echo "FAIL generating F_r$r"; exit 1; }
    fi
done

make_input() {
    local dir=$1 bin=$2 seed=$3
    mkdir -p "$dir"
    cat > "$dir/inputs" <<EOF
agent.urbanpop_filename = "$bin"
agent.nsteps = 160
agent.plot_int = -1
agent.weather_int = 1
agent.weather_filename = "$DATA/weatherData_NM_2015_2020.csv"
agent.startdate="2018-1-1"
agent.aggregated_diag_int = 160
agent.aggregated_diag_prefix = cases
agent.seed = $seed
diag.output_filename = output_nm.dat
disease.initial_case_type = "file"
disease.case_filename = "$DATA/CaseData/nm-july4.cases"
EOF
}

# The epidemic binary must still be the one arms A and E ran on.
chk=$ENS/F_binary_check
make_input "$chk" "$ENS/pops/A_base.bin" 0
(cd "$chk" && OMP_NUM_THREADS=5 "$AGENT" inputs > run.log 2>&1)
cmp -s "$chk/output_nm.dat" "$RUNS/A_base_s0/output_nm.dat" \
    || { echo "FAIL: $AGENT no longer reproduces $RUNS/A_base_s0"; exit 1; }
echo "binary check passed"

JOBS=$ENS/jobs_F.txt
: > "$JOBS"
for r in 1 2 3 4 5 6 7 8 9 10; do
    for seed in 0 1 2; do
        d=$RUNS/F_r${r}_s${seed}
        make_input "$d" "$ENS/pops/F_r${r}.bin" "$seed"
        echo "$d" >> "$JOBS"
    done
done
echo "queued $(wc -l < "$JOBS") runs"

export OMP_NUM_THREADS=5
xargs -a "$JOBS" -P 4 -I{} bash -c 'cd "$1" && '"$AGENT"' inputs > run.log 2>&1 && echo "ok $1" || echo "FAIL $1"' _ {}
echo "done"
