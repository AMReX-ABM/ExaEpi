#!/usr/bin/env python

"""Draw the schematic figures of the paper's model description, which have no data behind them:

  - disease-states: the disease progression of an infected agent (DiseaseStatus.H, HospitalModel.H),
    for the usual case of a positive pre-symptomatic period (DiseaseParm.H), so that agents become
    infectious before they show symptoms (or turn out to be asymptomatic).
  - transmission-hierarchy: the groups agents interact in, at night in their home community
    (households, household clusters, neighborhoods, the community) and by day in the community they
    work or study in (workgroups, schools and their classes, the neighborhoods and the community;
    AgentContainer::interactDay and interactNight).
  - seirhd_diagram: the compartments and rates of the SEIRHD model the ABMs are compared to, with
    Erlang chains of k_E exposed and k_I infectious stages (the equations in the paper's methods,
    and compare_to_epicast.py).

All are drawn at full page width with the paper's shared style (plos_compbio_style.py).
"""

import argparse
import os
import sys

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerTuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plos_compbio_style import FONT_LEGEND, FONT_TITLE, FULL_PAGE_WIDTH_IN, apply_style  # noqa: E402

EDGE = "0.2"
LABEL = "0.3"


def new_axes(height_in):
    """A figure the full page wide whose data coordinates are inches, so boxes and gaps can be laid
    out directly in physical units."""
    fig = plt.figure(figsize=(FULL_PAGE_WIDTH_IN, height_in))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, FULL_PAGE_WIDTH_IN)
    ax.set_ylim(0, height_in)
    ax.set_axis_off()
    return fig, ax


def box(ax, cx, cy, w, h, fc, ec, text=None, ls="-", lw=1.0, round_=0.06, z=1, **text_kw):
    ax.add_patch(mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h, boxstyle=f"round,pad=0,rounding_size={round_}",
                                         fc=fc, ec=ec, ls=ls, lw=lw, zorder=z))
    if text:
        ax.text(cx, cy, text, ha="center", va="center", zorder=z + 1, **text_kw)


def arrow(ax, start, end, rad=0.0, connection=None, ls="-", color=EDGE, lw=1.0, z=3):
    ax.annotate("", xy=end, xytext=start, zorder=z,
                arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=color, lw=lw, ls=ls,
                                shrinkA=0, shrinkB=0, connectionstyle=connection or f"arc3,rad={rad}"))


def disease_states(output):
    fig, ax = new_axes(3.12)
    h = 0.4
    # (x, y, width, label, fill, edge) of each state
    states = {
        "S": (0.48, 2.0, 0.9, "Susceptible", "#d0e2f3", "#3d85c6"),
        "E": (1.755, 2.0, 0.75, "Exposed", "#fff2cc", "#bf9000"),
        "P": (3.105, 2.0, 1.08, "Pre-symptomatic", "#f4cccc", "#cc4125"),
        "A": (4.58, 2.5, 1.0, "Asymptomatic", "#f4cccc", "#cc4125"),
        "Y": (4.58, 1.5, 1.0, "Symptomatic", "#f4cccc", "#cc4125"),
        "R": (6.005, 2.0, 0.95, "Recovered\n(immune)", "#d9ead3", "#6aa84f"),
        "H": (2.95, 0.45, 0.95, "Hospitalized", "#d9d2e9", "#8e7cc3"),
        "C": (3.95, 0.45, 0.55, "ICU", "#d9d2e9", "#8e7cc3"),
        "V": (4.85, 0.45, 0.9, "Ventilated", "#d9d2e9", "#8e7cc3"),
        "D": (6.17, 0.45, 0.6, "Dead", "#efefef", "#999999"),
    }

    def edge(k, side):
        x, y, w = states[k][:3]
        return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]

    def note(x, y, text, **kw):
        ax.text(x, y, text, fontsize=FONT_LEGEND, color=LABEL, style="italic", **{"ha": "center", "va": "center",
                                                                                     **kw})

    # The infectious states and the hospital states, each grouped in a dashed box
    ax.add_patch(mpatches.Rectangle((2.5, 1.2), 2.68, 1.6, fc="none", ec="0.5", ls="--", lw=0.8))
    note(2.58, 1.28, "Infectious", ha="left", va="bottom")
    ax.add_patch(mpatches.Rectangle((2.4, 0.15), 2.97, 0.6, fc="none", ec="0.5", ls="--", lw=0.8))
    note(2.33, 0.45, "In hospital", ha="right")

    for x, y, w, label, fc, ec in states.values():
        box(ax, x, y, w, h, fc, ec, label, lw=1.2, z=2, fontsize=FONT_LEGEND)

    arrow(ax, edge("S", "r"), edge("E", "l"))
    note(1.155, 2.33, "infection")
    arrow(ax, edge("E", "r"), edge("P", "l"))
    note(2.355, 2.4, "latent\nperiod")
    # At the end of the pre-symptomatic period, the agent shows symptoms or turns out to be asymptomatic
    arrow(ax, edge("P", "r"), edge("A", "l"), rad=-0.2)
    arrow(ax, edge("P", "r"), edge("Y", "l"), rad=0.2)
    # Recovery at the end of the infectious period
    arrow(ax, edge("A", "r"), (6.005 - 0.3, 2.0 + h / 2))
    arrow(ax, edge("Y", "r"), (6.005 - 0.3, 2.0 - h / 2))
    # Hospitalization, a few days after symptoms start, and the escalation within hospital
    arrow(ax, (4.35, 1.5 - h / 2), (3.15, 0.45 + h / 2))
    note(3.7, 0.9, "hospitalization", ha="left")
    arrow(ax, edge("H", "r"), edge("C", "l"))
    arrow(ax, edge("C", "r"), edge("V", "l"))
    # The outcome at the end of the hospital stay, from any of the hospital states
    arrow(ax, (5.37, 0.45), edge("D", "l"))
    note(5.62, 0.5, "death", va="bottom")
    arrow(ax, (5.25, 0.75), (5.9, 2.0 - h / 2))
    note(5.66, 1.18, "discharge", ha="left")
    # Waning immunity
    arrow(ax, edge("R", "t"), edge("S", "t"), connection="bar,fraction=0.13", ls=(0, (3, 2)))
    note(3.24, 2.95, "loss of immunity", va="bottom")

    print("Plotting results to", output)
    fig.savefig(output)


def agents(ax, x0, y, n, colors, dx=0.11, size=4.5, z=5):
    """A row of n agents centred on x0; returns their positions."""
    xs = [x0 + (i - (n - 1) / 2) * dx for i in range(n)]
    for x, c in zip(xs, colors):
        ax.plot(x, y, "o", ms=size, mfc=c, mec="0.15", mew=0.5, zorder=z)
    return list(zip(xs, [y] * n))


def transmission_hierarchy(output):
    fig, ax = new_axes(3.25)
    stay, worker, student = "0.6", "#3d85c6", "#cc4125"
    clusters = ["#fce5cd", "#d9ead3", "#d0e2f3"]
    cluster_edges = ["#e69138", "#6aa84f", "#3d85c6"]
    nbh_w, nbh_gap, comm_pad = 0.95, 0.06, 0.06
    comm_w = 3 * nbh_w + 2 * nbh_gap + 2 * comm_pad
    comm_gap = FULL_PAGE_WIDTH_IN - 2 * comm_w - 0.04
    comm_x = [0.02 + comm_w / 2, FULL_PAGE_WIDTH_IN - 0.02 - comm_w / 2]
    comm_y0, comm_y1 = 0.42, 3.22
    nbh_y0, nbh_y1 = comm_y0 + comm_pad, comm_y1 - 0.32
    rows = [2.25, 1.6, 0.95]  # centres of the groups within a neighborhood
    grp_w, grp_h = 0.8, 0.46

    def nbh_x(c, j):
        return comm_x[c] - comm_w / 2 + comm_pad + nbh_w / 2 + j * (nbh_w + nbh_gap)

    def group(x, y, label, fc, ec, roles, w=grp_w, hgt=grp_h, z=2, dx=0.12):
        box(ax, x, y, w, hgt, fc, ec, lw=0.9, z=z)
        ax.text(x, y + 0.1, label, ha="center", va="center", fontsize=FONT_LEGEND, color=LABEL, zorder=z + 1)
        colors = [{"w": worker, "s": student, "x": stay}[r] for r in roles]
        return agents(ax, x, y - 0.1, len(roles), colors, dx=dx)

    titles = ["Home community (night)", "Work or school community (day)"]
    for c in range(2):
        box(ax, comm_x[c], (comm_y0 + comm_y1) / 2, comm_w, comm_y1 - comm_y0, "#f3f3f3", "0.35", lw=1.0,
            round_=0.1, z=0)
        ax.text(comm_x[c], comm_y1 - 0.16, titles[c], ha="center", va="center", fontsize=FONT_TITLE, weight="bold")
        for j in range(3):
            box(ax, nbh_x(c, j), (nbh_y0 + nbh_y1) / 2, nbh_w, nbh_y1 - nbh_y0, "white", "0.55", lw=0.8, z=1)
            ax.text(nbh_x(c, j), nbh_y1 - 0.14, "Neighborhood", ha="center", va="center", fontsize=FONT_LEGEND,
                    color=LABEL)

    # Night: three households per neighborhood, each also in a household cluster (its fill color),
    # which can span neighborhoods. The commuters are the last agent of a household.
    layout = [
        [(0, "xsw"), (1, "xx"), (0, "xxss")],
        [(1, "xs"), (2, "xw"), (0, "x")],
        [(2, "xxs"), (1, "xxw"), (2, "xxs")],
    ]
    homes = {}
    for j, households in enumerate(layout):
        for i, (cl, roles) in enumerate(households):
            homes[(j, i)] = group(nbh_x(0, j), rows[i], "Household", clusters[cl], cluster_edges[cl], roles)

    # Day: workgroups, a school with its classes, and residents who stay home
    wg = group(nbh_x(1, 0), rows[1], "Workgroup", "#e8e8e8", "0.4", "www")
    group(nbh_x(1, 0), rows[0], "Workgroup", "#e8e8e8", "0.4", "ww")
    sx = nbh_x(1, 1)
    school_top, school_bottom = rows[0] + grp_h / 2 + 0.17, rows[2] - grp_h / 2 - 0.08
    box(ax, sx, (school_top + school_bottom) / 2, 0.88, school_top - school_bottom, "#fff2cc", "#bf9000", lw=0.9,
        z=2)
    ax.text(sx, school_top - 0.13, "School", ha="center", va="center", fontsize=FONT_LEGEND, color=LABEL,
            zorder=3)
    group(sx, rows[0] - 0.12, "Class", "white", "#bf9000", "sss", w=0.72, z=3)
    cls = group(sx, rows[2], "Class", "white", "#bf9000", "sss", w=0.72, z=3)
    group(nbh_x(1, 2), rows[0], "Workgroup", "#e8e8e8", "0.4", "ww")
    # Households only interact at night: by day, residents who stay home are only in their
    # neighborhood and community
    ax.text(nbh_x(1, 2), rows[2] + 0.1, "Residents\nat home", ha="center", va="bottom", fontsize=FONT_LEGEND,
            color=LABEL)
    agents(ax, nbh_x(1, 2), rows[2], 3, [stay] * 3)

    # Commutes from the home community, along the gaps between the groups
    for (x0, y0), (x1, y1), color in [(homes[(2, 1)][-1], wg[0], worker), (homes[(2, 2)][-1], cls[0], student)]:
        arrow(ax, (x0 + 0.04, y0), (x1 - 0.05, y1), color=color, lw=1.1, z=6)

    dot = dict(ls="", marker="o", ms=4.5, mec="0.15", mew=0.5)
    handles = [plt.Line2D([], [], mfc=worker, label="Worker", **dot),
               plt.Line2D([], [], mfc=student, label="Student", **dot),
               plt.Line2D([], [], mfc=stay, label="Agent who stays home", **dot),
               tuple(mpatches.Patch(fc=f, ec=e, lw=0.9) for f, e in zip(clusters, cluster_edges))]
    labels = [hd.get_label() for hd in handles[:3]] + ["Household clusters"]
    ax.legend(handles, labels, handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)}, loc="lower center",
              bbox_to_anchor=(0.5, -0.01), ncol=4, fontsize=FONT_LEGEND, frameon=False, handletextpad=0.4,
              columnspacing=1.5, handlelength=2.4)

    print("Plotting results to", output)
    fig.savefig(output)


def seirhd_diagram(output):
    fig, ax = new_axes(2.15)
    h, y_top, y_bot = 0.38, 1.55, 0.45
    big, small = 0.62, 0.4
    # (x, y, width, label, fill, edge) of each compartment
    states = {
        "S": (0.4, y_top, big, "$S$", "#d0e2f3", "#3d85c6"),
        "E1": (2.0, y_top, small, "$E_1$", "#fff2cc", "#bf9000"),
        "Ek": (2.8, y_top, small, "$E_{k_E}$", "#fff2cc", "#bf9000"),
        "I1": (3.88, y_top, small, "$I_1$", "#f4cccc", "#cc4125"),
        "Ik": (4.68, y_top, small, "$I_{k_I}$", "#f4cccc", "#cc4125"),
        "R": (6.0, y_top, big, "$R$", "#d9ead3", "#6aa84f"),
        "H": (4.68, y_bot, big, "$H$", "#d9d2e9", "#8e7cc3"),
        "D": (6.0, y_bot, big, "$D$", "#efefef", "#999999"),
    }

    def edge(k, side):
        x, y, w = states[k][:3]
        return {"l": (x - w / 2, y), "r": (x + w / 2, y), "t": (x, y + h / 2), "b": (x, y - h / 2)}[side]

    def note(x, y, text, italic=True, **kw):
        ax.text(x, y, text, fontsize=FONT_LEGEND, color=LABEL, style="italic" if italic else "normal",
                **{"ha": "center", "va": "center", **kw})

    def rate(x, y, text, **kw):
        ax.text(x, y, text, fontsize=FONT_TITLE, color=EDGE, **{"ha": "center", "va": "center", **kw})

    # The Erlang chains, each in a dashed box
    for x0, x1, name in [(1.72, 3.08, "Exposed"), (3.6, 4.96, "Infectious")]:
        ax.add_patch(mpatches.Rectangle((x0, y_top - 0.3), x1 - x0, 0.6, fc="none", ec="0.5", ls="--", lw=0.8))
        note((x0 + x1) / 2, y_top + 0.38, name)
    for a, b in [("E1", "Ek"), ("I1", "Ik")]:
        xa, xb = edge(a, "r")[0], edge(b, "l")[0]
        mid = (xa + xb) / 2
        arrow(ax, (xa, y_top), (mid - 0.1, y_top), color="0.4", lw=0.8)
        ax.text(mid, y_top, r"$\cdots$", ha="center", va="center", fontsize=FONT_TITLE, color="0.4")
        arrow(ax, (mid + 0.1, y_top), (xb, y_top), color="0.4", lw=0.8)

    for k, (x, y, w, label, fc, ec) in states.items():
        box(ax, x, y, w, h, fc, ec, label, lw=1.2, z=2, fontsize=FONT_TITLE)
    for k, name in [("S", "Susceptible"), ("R", "Recovered"), ("H", "Hospitalized"), ("D", "Dead")]:
        note(states[k][0], states[k][1] - h / 2 - 0.12, name, italic=False)

    arrow(ax, edge("S", "r"), (1.72, y_top))
    rate(1.215, y_top + 0.17, r"$\beta S\,\Sigma_j I_j / N$")
    arrow(ax, (3.08, y_top), (3.6, y_top))
    rate(3.34, y_top + 0.15, r"$k_E \sigma$")
    arrow(ax, edge("Ik", "r"), edge("R", "l"))
    rate(5.3, y_top + 0.15, r"$k_I \gamma$")
    arrow(ax, edge("Ik", "b"), edge("H", "t"))
    rate(4.82, (y_top + y_bot) / 2, "$k_I h$", ha="left")
    arrow(ax, edge("H", "r"), edge("D", "l"))
    rate(5.35, y_bot - 0.14, r"$\mu$")
    arrow(ax, (4.99, y_bot + h / 2), (6.0 - big / 2, y_top - 0.1))
    rate(5.3, (y_top + y_bot) / 2 + 0.12, r"$\gamma_h$", ha="right")

    print("Plotting results to", output)
    fig.savefig(output)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--output_dir", "-o", default=".", help="Directory for the figures")
    args = parser.parse_args()
    apply_style()
    disease_states(os.path.join(args.output_dir, "disease-states.png"))
    transmission_hierarchy(os.path.join(args.output_dir, "transmission-hierarchy.png"))
    seirhd_diagram(os.path.join(args.output_dir, "seirhd_diagram.png"))


if __name__ == "__main__":
    main()
