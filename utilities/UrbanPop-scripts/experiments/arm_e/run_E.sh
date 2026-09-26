#!/usr/bin/env bash
# Arm E: populations generated from the v2 bundle by generate_exaepi.py (seeds 1-10), 3 disease
# seeds each, same inputs as arms A-D in runs160. Also re-runs arm A with the current binary:
# output identical to runs160/A_base_* means the old arms are still a valid reference.
set -u

ENS=/workspaces/ExaEpi/data/UrbanPop/experiments/ens
UP=/workspaces/ExaEpi/data/UrbanPop
GEN=/workspaces/ExaEpi/.claude/worktrees/popgen-port/utilities/UrbanPop-scripts/generate_exaepi.py
AGENT=/workspaces/ExaEpi/.build/CPU/Release/bin/agent
DATA=/workspaces/ExaEpi/data
RUNS=$ENS/runs160E
mkdir -p "$RUNS"

for r in 1 2 3 4 5 6 7 8 9 10; do
    if [ ! -f "$ENS/pops/E_r$r.bin" ]; then
        (cd "$UP" && JAX_PLATFORMS=cuda,cpu XLA_PYTHON_CLIENT_PREALLOCATE=false \
            /opt/livelike-venv/bin/python -u "$GEN" --bundle experiments/nm_v2.upb --seed "$r" \
            --bin "$ENS/pops/E_r$r" > "$ENS/pops/E_r$r.log" 2>&1) \
            && echo "generated E_r$r" || { echo "FAIL generating E_r$r"; exit 1; }
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

JOBS=$ENS/jobs_E.txt
: > "$JOBS"
for seed in 0 1 2 3 4 5 6 7 8 9; do
    d=$RUNS/A_base_s$seed
    make_input "$d" "$ENS/pops/A_base.bin" "$seed"
    echo "$d" >> "$JOBS"
done
for r in 1 2 3 4 5 6 7 8 9 10; do
    for seed in 0 1 2; do
        d=$RUNS/E_r${r}_s${seed}
        make_input "$d" "$ENS/pops/E_r${r}.bin" "$seed"
        echo "$d" >> "$JOBS"
    done
done
echo "queued $(wc -l < "$JOBS") runs"

export OMP_NUM_THREADS=5
xargs -a "$JOBS" -P 4 -I{} bash -c 'cd "$1" && '"$AGENT"' inputs > run.log 2>&1 && echo "ok $1" || echo "FAIL $1"' _ {}
echo "done"
