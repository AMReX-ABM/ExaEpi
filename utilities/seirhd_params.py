#!/usr/bin/env python3
"""Derive SEIRHD compartment rates from an ExaEpi input file.

compare_to_epicast.py calls params_from_ini() below for its --seir_from_ini flag,
so plotting a derived SEIRHD curve needs nothing from this file's command line;
run it directly when you want to read the derivation rather than plot it.

That flag overlays a six-compartment SEIRHD model (run_seirhd_erlang there)
on the ExaEpi and Epicast curves.  The model is parameterised by rates:

    sigma    E -> I       1/sigma        = mean latent period
    gamma    I -> R       1/(gamma + h)  = mean infectious period
    h        I -> H       h/(gamma + h)  = P(hospitalised | infected)
    gamma_h  H -> R       1/(gamma_h+mu) = mean hospital stay
    mu       H -> D       mu/(gamma_h+mu)= P(dead | hospitalised)

ExaEpi has no such parameters: it draws per-agent latent/pre-symptomatic/infectious
periods from gamma distributions (incubation = latent + pre-symptomatic) and resolves hospitalisation and death through
per-age-group conditional probabilities (CHR/CIC/CVE/hospCVF/icuCVF/ventCVF).
This script maps the latter onto the former by replaying ExaEpi's *agent
lifecycle* -- the draws in setInfected() plus the day-by-day transitions in
DiseaseStatus.H and HospitalModel.H -- over the age composition of the
population, and reporting the mean and coefficient of variation of the time
each agent spends in each compartment.  Nothing here reads a simulation
result; the only inputs are the .ini and the UrbanPop population file.

It also reports the Erlang shape parameters kE and kI, since the model takes
those too: a compartment whose dwell time has coefficient of variation CV is
matched by an Erlang chain of k = 1/CV^2 sub-stages.  H is left as a single
exponential compartment (kD=1) by choice -- see to_rates() for why its shape
does not earn its compartments.

beta is deliberately *not* derived.  It depends on the contact network -- group
sizes, who shares a household, school and workgroup -- and not on any disease
parameter, so it stays a fitted quantity.  --r0 converts a fitted R0 into the
matching beta, which is the useful direction: R0 = beta/(gamma + h), so beta
has to be refitted whenever the derived rates change.

Usage:
    seirhd_params.py cfgs/epicast.ca.ini
    seirhd_params.py cfgs/epicast.ca.ini --r0 1.57 --show-inputs
"""

import argparse
import json
import os
import struct
import sys
import zlib
from typing import Any, NamedTuple

import numpy as np

# Age groups, in the order ExaEpi's AgeGroups enum (AgentDefinitions.H) and every
# per-age-group input array use.  The upper bounds match UrbanPopData.cpp.
AGE_GROUP_NAMES = ["0-4", "5-17", "18-29", "30-49", "50-64", "65+"]
AGE_GROUP_UPPER = [5, 18, 30, 50, 65, 128]
N_AGE = len(AGE_GROUP_NAMES)

# Defaults for every parameter read below, copied from DiseaseParm.H.  An .ini only
# has to set what it changes -- epicast.ca.ini, for one, leaves hospitalization_days
# at its default -- so deriving from the .ini alone would silently use the wrong
# value for anything left out.
DEFAULTS: dict[str, Any] = {
    "p_asymp": 0.3,
    "compare_to_epicast": False,
    "latent_length_alpha": 2.77,
    "latent_length_beta": 1.5,
    "presymptomatic_length_alpha": 0.0,
    "presymptomatic_length_beta": 0.0,
    "presymptomatic_length_loc": 1.0,
    "infectious_length_alpha": 3.54,
    "infectious_length_beta": 1.22,
    "infectious_length_loc": 2.75,
    "hospital_delay_length_alpha": 0.0,
    "hospital_delay_length_beta": 0.0,
    "hospital_delay_length_loc": 2.0,
    "hospital_stay_type": "constant",
    "hospitalization_days": [3.0, 3.0, 3.0, 3.0, 8.0, 7.0],
    "hospitalization_days_alpha": [3.0, 3.0, 3.0, 3.0, 8.0, 7.0],
    "hospitalization_days_beta": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
    "CHR": [0.0104, 0.0104, 0.070, 0.28, 0.28, 1.0],
    "CIC": [0.24, 0.24, 0.24, 0.36, 0.36, 0.35],
    "CVE": [0.12, 0.12, 0.12, 0.22, 0.22, 0.22],
    "hospCVF": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "icuCVF": [0.0, 0.0, 0.0, 0.0, 0.0, 0.26],
    "ventCVF": [0.20, 0.20, 0.20, 0.45, 0.45, 1.0],
}

# The agent record layout from UrbanPopAgentStruct.H, in declaration order, as
# (field, width in bytes).  A frame stores its agents column by column, so the byte
# offset of the age column inside a frame of n agents is the sum of the widths ahead
# of it times n.  The total is checked against the agent_record_size in the file
# header, which means a field added, widened or reordered without this table being
# updated is caught rather than silently shifting the histogram.
AGENT_FIELD_WIDTHS = [
    ("id", 8), ("home_geoid", 8), ("work_geoid", 8),
    ("school_class_group", 4), ("work_group", 4),
    ("naics", 2), ("household_id", 2), ("school_id", 2), ("nborhood", 2),
    ("work_nborhood", 2), ("workgroup", 2), ("hh_cluster", 2), ("school_class", 2),
    ("age", 1), ("sex", 1), ("race", 1), ("travel", 1), ("veh_occ", 1), ("grade", 1),
]

# Fixed cumulative distributions used instead of the gamma draws when
# compare_to_epicast is set; these mirror the arrays in DiseaseParm.H.
LATENT_PERIOD_CDF = [0.0, 0.1, 0.25, 0.55, 0.66, 0.78, 0.85, 1.0]
INFECTIOUS_PERIOD_CDF = [0.0, 0.0, 0.0, 0.0, 0.1, 0.3, 0.5, 0.7, 0.85, 0.95, 1.0]


# ---------------------------------------------------------------------------
# .ini parsing


def parse_ini(path: str, overrides: list[str]) -> dict[str, list[str]]:
    """Parse an ExaEpi (AMReX ParmParse) input file into {key: [tokens]}."""
    params: dict[str, list[str]] = {}
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            params[key.strip()] = value.replace('"', " ").split()
    for override in overrides:
        if "=" not in override:
            sys.exit(f"error: override '{override}' is not key=value")
        key, _, value = override.partition("=")
        params[key.strip()] = value.replace('"', " ").split()
    return params


class Params:
    """Parameter lookup that mirrors ParmParse's query()/queryarr() semantics.

    Keys absent from the .ini fall back to the DiseaseParm.H default rather than
    to zero, and every lookup is recorded so --show-inputs can report which
    values came from the file and which are defaults.
    """

    def __init__(self, raw: dict[str, list[str]], prefix: str):
        self.raw = raw
        self.prefix = prefix
        self.used: dict[str, tuple[Any, str]] = {}
        # DiseaseParm::readInputs() aborts on these, so refuse them here too.
        for old in ("incubation_length_alpha", "incubation_length_beta",
                    "incubation_length_loc"):
            if f"{prefix}.{old}" in raw:
                sys.exit(
                    f"error: {prefix}.{old} is no longer supported: the incubation period "
                    f"is now always latent + pre-symptomatic period. Use "
                    f"{prefix}.presymptomatic_length_alpha/beta/loc instead.")

    def _tokens(self, name: str) -> list[str] | None:
        return self.raw.get(f"{self.prefix}.{name}")

    def scalar(self, name: str) -> float:
        tokens = self._tokens(name)
        value = float(DEFAULTS[name] if tokens is None else tokens[0])
        self.used[name] = (value, "default" if tokens is None else "ini")
        return value

    def boolean(self, name: str) -> bool:
        tokens = self._tokens(name)
        value = (bool(DEFAULTS[name]) if tokens is None
                 else tokens[0].lower() in ("1", "true", "t", "yes"))
        self.used[name] = (value, "default" if tokens is None else "ini")
        return value

    def string(self, name: str) -> str:
        tokens = self._tokens(name)
        value = str(DEFAULTS[name] if tokens is None else tokens[0])
        self.used[name] = (value, "default" if tokens is None else "ini")
        return value

    def array(self, name: str) -> np.ndarray:
        """Read an N_AGE array.  ParmParse leaves trailing entries at their
        default when the .ini supplies fewer than N_AGE values, so do the same."""
        value = list(DEFAULTS[name])
        tokens = self._tokens(name)
        source = "default"
        if tokens is not None:
            source = "ini"
            for i, token in enumerate(tokens[:N_AGE]):
                value[i] = float(token)
        value = np.array(value, dtype=float)
        self.used[name] = (value, source)
        return value


# ---------------------------------------------------------------------------
# Population age composition


def read_urbanpop_age_fractions(path):
    """Return (age-group fractions, total population) from a UrbanPop .bin.

    Layout is the one upop_to_exaepi.py writes and UrbanPopData.cpp reads: a
    40-byte header, an index with one fixed-size entry per GEOID, then one
    independently deflated frame per home GEOID.  Within a frame the agent
    records are stored column by column, so all the ages sit in a single
    contiguous run and there is no need to unpack whole records.
    """
    header_struct = struct.Struct("<2I 2I Q I I Q")
    if not os.path.isfile(path):
        sys.exit(f"error: no such UrbanPop file: {path}")
    with open(path, "rb") as f:
        header = f.read(header_struct.size)
        if len(header) < header_struct.size:
            sys.exit(f"error: {path} is too short to be a UrbanPop file")
        (magic, version, num_naics, _num_geoids, _num_agents, record_size, codec,
         index_end) = header_struct.unpack(header)
        if magic != 0x55504F50:
            sys.exit(f"error: {path} is not a UrbanPop .bin (bad magic number)")
        if codec not in (0, 1):
            sys.exit(f"error: {path} uses unsupported codec {codec}")
        index_struct = struct.Struct(f"<QQ III {num_naics}I")
        index = f.read(index_end - header_struct.size)

        names = [name for name, _ in AGENT_FIELD_WIDTHS]
        age_column_offset = sum(width for name, width in AGENT_FIELD_WIDTHS
                                if names.index(name) < names.index("age"))
        expected_record_size = sum(width for _, width in AGENT_FIELD_WIDTHS)
        if record_size != expected_record_size:
            sys.exit(f"error: {path} has {record_size}-byte agent records, expected "
                     f"{expected_record_size} -- UrbanPopAgentStruct.H has changed, so "
                     f"AGENT_FIELD_WIDTHS in this script needs updating to match")

        histogram = np.zeros(256, dtype=np.int64)
        for i in range(len(index) // index_struct.size):
            _geoid, offset, nbytes, pop, _w_pop = index_struct.unpack_from(
                index, i * index_struct.size)[:5]
            if pop == 0:  # a work-only GEOID: nobody lives there, no frame
                continue
            f.seek(offset)
            blob = f.read(nbytes)
            frame = zlib.decompress(blob) if codec == 1 else blob
            ages = np.frombuffer(
                frame[age_column_offset * pop:(age_column_offset + 1) * pop],
                dtype=np.int8).astype(np.int64)
            histogram += np.bincount(ages, minlength=256)[:256]

    counts = np.zeros(N_AGE)
    lower = 0
    for i, upper in enumerate(AGE_GROUP_UPPER):
        counts[i] = histogram[lower:upper].sum()
        lower = upper
    total = int(counts.sum())
    if total == 0:
        sys.exit(f"error: read no agents from {path}")
    return counts / total, total


def locate_urbanpop(ini_path, raw, explicit):
    """Find the UrbanPop .bin, which the .ini names relative to the *run*
    directory rather than to itself."""
    if explicit:
        return explicit
    tokens = raw.get("agent.urbanpop_filename")
    if not tokens:
        return None
    named = tokens[0]
    candidates = [named, os.path.join(os.path.dirname(os.path.abspath(ini_path)), named)]
    # The .ini lives in cfgs/ and is run from its parent, so also try resolving
    # relative to a few levels up from the .ini.
    base = os.path.dirname(os.path.abspath(ini_path))
    for _ in range(4):
        base = os.path.dirname(base)
        candidates.append(os.path.join(base, named))
    for candidate in candidates:
        if os.path.isfile(candidate):
            return candidate
    return None


def age_fractions_cache_path(bin_path):
    return bin_path + ".agecounts.json"


def resolve_age_fractions(ini_path, raw, urbanpop=None, age_fractions=None,
                          population=None, use_cache=True):
    """Return (age-group fractions, population, a description of where they came from)."""
    if age_fractions:
        fractions = np.array([float(x) for x in age_fractions.split(",")])
        if fractions.size != N_AGE:
            sys.exit(f"error: --age-fractions needs {N_AGE} comma-separated values")
        return fractions / fractions.sum(), population, "--age-fractions"

    bin_path = locate_urbanpop(ini_path, raw, urbanpop)
    if bin_path is None:
        sys.exit("error: could not find the UrbanPop .bin (the .ini's "
                 "agent.urbanpop_filename is relative to the run directory, not to "
                 "the .ini). Pass --urbanpop, or --age-fractions to skip it.")

    cache = age_fractions_cache_path(bin_path)
    if use_cache and os.path.isfile(cache):
        try:
            if os.path.getmtime(cache) >= os.path.getmtime(bin_path):
                with open(cache) as f:
                    cached = json.load(f)
                return (np.array(cached["fractions"]), cached["population"],
                        f"{bin_path} (cached)")
        except (OSError, ValueError, KeyError):
            pass  # a stale or corrupt cache is not worth failing over

    fractions, population = read_urbanpop_age_fractions(bin_path)
    if use_cache:
        try:
            with open(cache, "w") as f:
                json.dump({"fractions": list(fractions), "population": population}, f)
        except OSError:
            pass  # read-only data directory; just recompute next time
    return fractions, population, bin_path


# ---------------------------------------------------------------------------
# Agent lifecycle


def sample_from_cdf(cdf, rng, n):
    """Inverse-CDF sampling, matching sampleFromCDF() in DiseaseParm.H."""
    return np.searchsorted(np.asarray(cdf), rng.random(n), side="left").astype(float)


def draw_periods(p, rng, n):
    """Replay setInfected() (DiseaseParm.H) for n agents.

    Note that beta is the gamma *scale*, not a rate: ExaEpi calls
    amrex::RandomGamma, which is std::gamma_distribution, so the mean of a drawn
    period is alpha * beta (+ loc).
    """
    if p.boolean("compare_to_epicast"):
        latent = sample_from_cdf(LATENT_PERIOD_CDF, rng, n)
        incubation = latent + 1.0
        infectious = sample_from_cdf(INFECTIOUS_PERIOD_CDF, rng, n)
        hospital_delay = np.full(n, 2.0)
    else:
        latent = rng.gamma(p.scalar("latent_length_alpha"),
                           p.scalar("latent_length_beta"), n)
        # Pre-symptomatic period (infectiousness -> symptom onset), possibly
        # negative; alpha <= 0 means exactly presymptomatic_length_loc days.
        presymp_alpha = p.scalar("presymptomatic_length_alpha")
        if presymp_alpha > 0:
            presymptomatic = rng.gamma(presymp_alpha,
                                       p.scalar("presymptomatic_length_beta"), n)
        else:
            presymptomatic = np.zeros(n)
        presymptomatic = presymptomatic + p.scalar("presymptomatic_length_loc")
        infectious = rng.gamma(p.scalar("infectious_length_alpha"),
                               p.scalar("infectious_length_beta"), n) \
            + p.scalar("infectious_length_loc")
        # RandomGamma needs alpha > 0; ExaEpi reads alpha <= 0 as "no random
        # component", leaving a deterministic delay of hospital_delay_length_loc.
        delay_alpha = p.scalar("hospital_delay_length_alpha")
        if delay_alpha > 0:
            hospital_delay = rng.gamma(delay_alpha,
                                       p.scalar("hospital_delay_length_beta"), n)
        else:
            hospital_delay = np.zeros(n)
        hospital_delay = hospital_delay + p.scalar("hospital_delay_length_loc")
        latent = np.maximum(latent, 1.0)
        # incubation >= 1, enforced by clamping the pre-symptomatic period
        # (not to >= 0, only to >= 1 - latent).
        incubation = latent + np.maximum(presymptomatic, 1.0 - latent)
        infectious = np.maximum(infectious, 1.0)

    # setInfected() extends the infectious period if hospitalisation would
    # otherwise land after recovery, so that a symptomatic agent is always still
    # infected when its CHR check fires.
    hosp_check_day = np.round(incubation) + np.round(hospital_delay)
    infectious = np.where(np.round(latent + infectious) < hosp_check_day,
                          hosp_check_day - latent, infectious)
    return latent, incubation, infectious, hospital_delay


def derive(p, age_fractions, rng, n):
    """Return per-compartment dwell-time statistics for n simulated agents.

    The dwell times are counted the way ExaEpi's own compartment diagnostics
    count them, in whole days off the agent's disease_counter: an agent is
    latent while counter <= round(latent_period), infectious from the next day
    until the day before it recovers or is admitted, and hospitalised for
    t_hosp_days days after admission.
    """
    age = rng.choice(N_AGE, size=n, p=age_fractions)
    latent, incubation, infectious, hospital_delay = draw_periods(p, rng, n)

    # An agent turns symptomatic (or not) at the end of its incubation period,
    # and only a symptomatic one is ever tested against CHR -- so the asymptomatic
    # fraction never reaches hospital.  DiseaseStatus.H:158-190.
    symptomatic = rng.random(n) >= p.scalar("p_asymp")
    CHR = p.array("CHR")
    hospitalised = symptomatic & (rng.random(n) < CHR[age])

    latent_days = np.round(latent)
    recovery_day = np.round(latent + infectious)
    admission_day = np.round(incubation) + np.round(hospital_delay)
    exit_day = np.where(hospitalised, admission_day, recovery_day)
    infectious_days = np.maximum(exit_day - latent_days - 1.0, 0.0)

    # Hospital stay: a constant per age group, or a gamma draw rounded to whole
    # days and clamped to at least 1.  checkHospitalization(), DiseaseParm.H.
    if p.string("hospital_stay_type") == "random":
        alpha = p.array("hospitalization_days_alpha")
        beta = p.array("hospitalization_days_beta")
        stay = np.maximum(np.trunc(rng.gamma(alpha[age], beta[age])), 1.0)
    else:
        stay = p.array("hospitalization_days")[age]

    # An agent escalates hospital -> ICU -> ventilator, and is tested for death
    # once, against the fatality rate of the highest level it reached.
    # HospitalModel.H:167-195.
    CIC, CVE = p.array("CIC"), p.array("CVE")
    in_icu = rng.random(n) < CIC[age]
    on_vent = in_icu & (rng.random(n) < CVE[age])
    fatality = np.where(on_vent, p.array("ventCVF")[age],
                        np.where(in_icu, p.array("icuCVF")[age],
                                 p.array("hospCVF")[age]))
    died = hospitalised & (rng.random(n) < fatality)

    def stats(values):
        mean = float(values.mean())
        # The Erlang chain that best matches a dwell time reproduces its
        # coefficient of variation: an Erlang(k) has CV = 1/sqrt(k).
        cv = float(values.std() / mean) if mean > 0 else float("nan")
        return mean, cv

    latent_mean, latent_cv = stats(latent_days)
    infectious_mean, infectious_cv = stats(infectious_days)
    stay_mean, stay_cv = stats(stay[hospitalised])

    return {
        "latent_mean": latent_mean,
        "latent_cv": latent_cv,
        "infectious_mean": infectious_mean,
        "infectious_cv": infectious_cv,
        "hosp_stay_mean": stay_mean,
        "hosp_stay_cv": stay_cv,
        "p_hosp": float(hospitalised.mean()),
        "p_death_given_hosp": float(died.sum() / max(hospitalised.sum(), 1)),
        "ifr": float(died.mean()),
        "hosp_age_fractions": np.bincount(age[hospitalised], minlength=N_AGE)
        / max(hospitalised.sum(), 1),
    }


# compare_to_epicast.py adds one ODE compartment per Erlang sub-stage, so the shapes
# need an upper bound; by 50 stages the dwell time is near-deterministic (CV 0.14)
# and more stages buy nothing visible.
MAX_ERLANG_SHAPE = 50


def erlang_shape(cv):
    """Erlang stages reproducing a dwell time's spread: Erlang(k) has CV = 1/sqrt(k).

    Note which way a near-constant dwell time clamps. CV 0 is the k -> infinity
    limit, not k = 1 -- k = 1 is the exponential, the most variable case there is,
    so clamping the wrong way would turn the least variable dwell time into the
    most variable one.
    """
    if not np.isfinite(cv) or cv <= 1.0 / np.sqrt(MAX_ERLANG_SHAPE):
        return MAX_ERLANG_SHAPE
    return max(1, round(1.0 / cv ** 2))


def to_rates(d):
    """Convert dwell times and branch probabilities into SEIRHD rates."""
    sigma = 1.0 / d["latent_mean"]
    exit_i = 1.0 / d["infectious_mean"]
    hosp_rate = exit_i * d["p_hosp"]
    gamma = exit_i - hosp_rate
    exit_h = 1.0 / d["hosp_stay_mean"]
    mu = exit_h * d["p_death_given_hosp"]
    gamma_h = exit_h - mu
    return {
        "sigma": sigma,
        "gamma": gamma,
        "hosp_rate": hosp_rate,
        "gamma_h": gamma_h,
        "mu": mu,
        "kE": erlang_shape(d["latent_cv"]),
        "kI": erlang_shape(d["infectious_cv"]),
        # No kD: H is deliberately left as a single exponential compartment. kD is
        # the one shape that does not reach the plotted curves much -- admissions
        # come from the I chain and carry no kD at all, and on the death curve the
        # difference between kD=1 and the shape-matching value is under 3% of the
        # peak and one day of timing, because that curve's ~54-day width is set by
        # the spread of admission times rather than by a stay whose sd is 2 days.
        # See the reported hospital-stay CV for what it would have been.
    }


# ---------------------------------------------------------------------------
# Library entry point


DEFAULT_NUM_AGENTS = 4_000_000
DEFAULT_RNG_SEED = 0


class Derivation(NamedTuple):
    """Everything params_from_ini() worked out, for callers that want more than
    the rates -- seirhd_params' own report, and compare_to_epicast.py's
    --seir_from_ini."""

    rates: dict[str, Any]          # sigma/gamma/hosp_rate/gamma_h/mu/kE/kI, plus N
    dwell: dict[str, Any]          # mean and CV of each compartment's dwell time
    inputs: "Params"               # every parameter read, and where it came from
    age_fractions: np.ndarray
    population: int | None
    age_source: str


def params_from_ini(ini_path, overrides=(), prefix="disease", urbanpop=None,
                    age_fractions=None, population=None,
                    num_agents=DEFAULT_NUM_AGENTS, rng_seed=DEFAULT_RNG_SEED,
                    use_cache=True) -> Derivation:
    """Derive SEIRHD rates from an ExaEpi .ini and its UrbanPop population.

    This is the whole job of this module, and the entry point compare_to_epicast.py
    calls for --seir_from_ini; the CLI below is a thin report over it.
    """
    raw = parse_ini(ini_path, list(overrides))
    inputs = Params(raw, prefix)
    fractions, population, age_source = resolve_age_fractions(
        ini_path, raw, urbanpop=urbanpop, age_fractions=age_fractions,
        population=population, use_cache=use_cache)
    dwell = derive(inputs, fractions, np.random.default_rng(rng_seed), num_agents)
    rates = to_rates(dwell)
    if population:
        rates["N"] = population
    return Derivation(rates, dwell, inputs, fractions, population, age_source)


# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Report the SEIRHD rates an ExaEpi .ini implies.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Every rate comes from the .ini and the UrbanPop population file; "
               "no simulation output is read.  beta is not derivable from disease "
               "parameters -- use --r0 to turn a fitted R0 into the matching beta.  "
               "To plot a curve from these rates, pass the .ini to "
               "compare_to_epicast.py --seir_from_ini instead; it derives them through "
               "the same code.")
    parser.add_argument("ini", help="ExaEpi input file")
    parser.add_argument("overrides", nargs="*",
                        help="key=value overrides, as passed on the agent command line")
    parser.add_argument("--prefix", default="disease",
                        help="ParmParse prefix for the disease parameters "
                             "(default: %(default)s)")
    parser.add_argument("--urbanpop", help="UrbanPop .bin (default: from the .ini)")
    parser.add_argument("--age-fractions",
                        help="comma-separated population fractions for "
                             + "/".join(AGE_GROUP_NAMES) + ", instead of reading the "
                             ".bin")
    parser.add_argument("--population", type=int,
                        help="total population the SEIRHD model is run at (default: "
                             "from the .bin)")
    parser.add_argument("--r0", type=float,
                        help="fitted R0, converted to beta = R0 * (gamma + hosp_rate)")
    parser.add_argument("-n", "--num-agents", type=int, default=DEFAULT_NUM_AGENTS,
                        help="agents to simulate (default: %(default)s)")
    parser.add_argument("--seed", type=int, default=DEFAULT_RNG_SEED,
                        help="RNG seed, so the output is reproducible "
                             "(default: %(default)s)")
    parser.add_argument("--no-cache", action="store_true",
                        help="ignore and do not write the .agecounts.json cache "
                             "beside the .bin")
    parser.add_argument("--show-inputs", action="store_true",
                        help="also list every parameter read, and whether it came "
                             "from the .ini or from the DiseaseParm.H default")
    args = parser.parse_args()

    derivation = params_from_ini(
        args.ini, overrides=args.overrides, prefix=args.prefix,
        urbanpop=args.urbanpop, age_fractions=args.age_fractions,
        population=args.population, num_agents=args.num_agents,
        rng_seed=args.seed, use_cache=not args.no_cache)
    r, d, p = derivation.rates, derivation.dwell, derivation.inputs
    age_fractions, population = derivation.age_fractions, derivation.population
    age_source = derivation.age_source
    beta = args.r0 * (r["gamma"] + r["hosp_rate"]) if args.r0 is not None else None

    print(f"SEIRHD parameters derived from {args.ini}")
    print(f"  age composition: {age_source}")
    print("  " + "  ".join(f"{name}={frac:.4f}"
                           for name, frac in zip(AGE_GROUP_NAMES, age_fractions)))
    if population:
        print(f"  population: {population}")
    print(f"  {args.num_agents} simulated agents, seed {args.seed}")

    if args.show_inputs:
        print("\nInputs:")
        for name, (value, source) in p.used.items():
            shown = (np.array2string(value, precision=6, separator=" ")
                     if isinstance(value, np.ndarray) else value)
            print(f"  {name:32s} {str(shown):48s} [{source}]")

    print("\nDwell times (days) and branch probabilities:")
    print(f"  latent period                      {d['latent_mean']:8.3f}"
          f"   CV {d['latent_cv']:.3f}")
    print(f"  infectious period                  {d['infectious_mean']:8.3f}"
          f"   CV {d['infectious_cv']:.3f}")
    print(f"  hospital stay                      {d['hosp_stay_mean']:8.3f}"
          f"   CV {d['hosp_stay_cv']:.3f}  (kD would be "
          f"{erlang_shape(d['hosp_stay_cv'])}; see to_rates)")
    print(f"  P(hospitalised | infected)         {d['p_hosp']:8.4f}")
    print(f"  P(dead | hospitalised)             {d['p_death_given_hosp']:8.4f}")
    print(f"  infection fatality rate            {d['ifr']:8.4f}")
    print("  hospitalisations by age: "
          + "  ".join(f"{name}={frac:.3f}"
                      for name, frac in zip(AGE_GROUP_NAMES, d["hosp_age_fractions"])))

    print("\nSEIRHD rates:")
    print(f"  sigma     (E->I progression)       {r['sigma']:8.4f}")
    print(f"  gamma     (I->R direct recovery)   {r['gamma']:8.4f}")
    print(f"  hosp_rate (I->H)                   {r['hosp_rate']:8.4f}")
    print(f"  gamma_h   (H->R recovery)          {r['gamma_h']:8.4f}")
    print(f"  mu        (H->D death)             {r['mu']:8.4f}")
    print(f"  kE / kI   (Erlang shapes)          {r['kE']:4d} {r['kI']:4d}"
          f"        (H is left exponential: kD=1)")
    if beta is not None:
        print(f"  beta      (for R0 = {args.r0:.3f})          {beta:8.4f}")
    else:
        print("\nbeta is not derivable from disease parameters -- it depends on the "
              "contact\nnetwork. Fit it, or pass --r0 to convert a fitted R0 into "
              "the matching beta.")


if __name__ == "__main__":
    main()
