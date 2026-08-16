#!/usr/bin/env python

"""Read each agent's static home/work grid-cell assignment from an ExaEpi AMReX particle plotfile.

ExaEpi writes a raw AMReX particle plotfile in the "agents" subdirectory of every pltNNNNN
directory (see src/IO.cpp's writePlotFile -> pc.WritePlotFile(..., "agents", ...)). Most of its int
fields (withdrawn, status, symptomatic, ...) are written every step, but home_i/home_j/work_i/work_j
are static per agent (an agent's home and work locations don't change), so ExaEpi only writes them
at step 0 (write_int_comp.push_back(static_cast<int>(step == 0)) in IO.cpp) -- later steps' "agents"
plotfiles omit these four fields from their Header entirely. So a single step-0 snapshot (e.g.
plt00000) is both necessary and sufficient to recover every agent's home/work grid cell.

This is a from-scratch reader (not yt) because yt's BoxLib/AMReX frontend only auto-discovers
particle subdirectories that the main plotfile Header references, and "agents" isn't -- it's a
second, independent WritePlotFile call with its own Header/Level_0/DATA_* layout. The format itself
(AMReXParticleHeader "Version_Two_Dot_Zero_<real_type>") is the same one yt's own AMReX frontend
parses; see yt/frontends/amrex/data_structures.py's AMReXParticleHeader and
yt/frontends/amrex/io.py's IOHandlerBoxlib._read_particle_fields for the reference implementation
this was written against.

Binary layout (per grid, within a Level_<n>/DATA_<file_number> file, starting at that grid's byte
offset from Header): npart particles x num_int int32 fields, particle-major (agent 0's fields, then
agent 1's, ...), immediately followed by npart x num_real real fields (not read here -- we only need
int fields, which come first).
"""

import glob
import os

import numpy as np


def _parse_particle_header(agents_dir):
    """Parse <agents_dir>/Header (AMReXParticleHeader format) into its int-field layout and the
    per-grid (file_number, num_particles, byte_offset) triples needed to read the DATA files.
    """
    with open(os.path.join(agents_dir, "Header")) as f:
        lines = [line.rstrip("\n") for line in f]

    i = 0
    _version_string = lines[i]
    i += 1
    dim = int(lines[i])
    i += 1
    num_real_extra = int(lines[i])
    i += 1
    i += num_real_extra  # real component names -- unused, we only read int fields
    num_int_extra = int(lines[i])
    i += 1
    int_names = lines[i : i + num_int_extra]
    i += num_int_extra
    is_checkpoint = bool(int(lines[i]))
    i += 1
    _num_particles = int(lines[i])
    i += 1
    i += 1  # max_next_id -- unused
    finest_level = int(lines[i])
    i += 1
    num_levels = finest_level + 1

    num_int_base = 2 if is_checkpoint else 0  # particle_id, particle_cpu
    num_int = num_int_base + num_int_extra
    num_real = dim + num_real_extra  # unused here, kept for documentation of the layout
    int_field_names = (["particle_id", "particle_cpu"] if is_checkpoint else []) + int_names

    data_map = {}
    for level in range(num_levels):
        ngrids = int(lines[i])
        i += 1
        entries = []
        for _ in range(ngrids):
            file_number, npart, offset = (int(v) for v in lines[i].split())
            i += 1
            entries.append((file_number, npart, offset))
        data_map[level] = entries

    return {
        "num_int": num_int,
        "num_real": num_real,
        "int_field_names": int_field_names,
        "data_map": data_map,
    }


def _data_filename_width(level_dir):
    """DATA file names are zero-padded to a fixed but undocumented width (observed: 5 digits) --
    detect it once per level directory from whatever DATA_* files actually exist, rather than
    assuming a specific width.
    """
    sample = sorted(glob.glob(os.path.join(level_dir, "DATA_*")))
    if not sample:
        raise FileNotFoundError(f"No DATA_* files found in {level_dir}")
    return len(os.path.basename(sample[0])) - len("DATA_")


def read_agent_fields(plot_dir, field_names):
    """
    Read arbitrary int fields for every agent from <plot_dir>/agents.

    Parameters
    ----------
    plot_dir : str
        An ExaEpi plotfile directory. Most per-agent fields (home_i/home_j/work_i/work_j, naics,
        school_id, ...) are static and only written at step 0 -- see module docstring -- so this
        must usually be a step-0 directory (e.g. "plt00000").
    field_names : iterable of str
        Int field names to read (see the "agents" plotfile's Header for what's available, e.g.
        via _parse_particle_header(os.path.join(plot_dir, "agents"))["int_field_names"]).

    Returns
    -------
    dict of numpy int32 arrays, one per agent, keyed by the requested field names.
    """
    agents_dir = os.path.join(plot_dir, "agents")
    header = _parse_particle_header(agents_dir)
    int_field_names = header["int_field_names"]

    field_names = tuple(field_names)
    missing = [name for name in field_names if name not in int_field_names]
    if missing:
        raise SystemExit(
            f"{missing} not found in {agents_dir}/Header -- most per-agent fields are only "
            "written at step 0 (they're static per agent), so pass the step-0 plotfile directory "
            "(e.g. plt00000), not a later step."
        )
    col_idx = {name: int_field_names.index(name) for name in field_names}
    num_int = header["num_int"]

    chunks = {name: [] for name in field_names}
    for level, entries in header["data_map"].items():
        level_dir = os.path.join(agents_dir, f"Level_{level}")
        width = _data_filename_width(level_dir)
        for file_number, npart, offset in entries:
            if npart == 0:
                continue
            data_fn = os.path.join(level_dir, f"DATA_{file_number:0{width}d}")
            with open(data_fn, "rb") as f:
                f.seek(offset)
                idata = np.fromfile(f, dtype=np.int32, count=npart * num_int)
            idata = idata.reshape(npart, num_int)
            for name, idx in col_idx.items():
                chunks[name].append(idata[:, idx])

    return {name: np.concatenate(arrs) for name, arrs in chunks.items()}


def read_agent_home_work(plot_dir):
    """Convenience wrapper around read_agent_fields for the four home/work fields -- see that
    function for details.
    """
    return read_agent_fields(plot_dir, ("home_i", "home_j", "work_i", "work_j"))
