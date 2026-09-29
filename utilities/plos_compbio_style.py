"""Shared matplotlib styling for every plot_geo*/plot_group_size*/plot_gini_timeseries.py figure
going into the emerge paper (a PLOS Computational Biology submission), so they all look like one
consistent set rather than each script's own ad hoc font sizes and aspect ratios.

Font sizes (8-12pt, using the very bottom of that range: 8pt ticks/legends/axis labels, 9pt titles --
Arial reads noticeably larger than the Computer Modern of the paper's 9pt sub-captions at the same
point size, so anything bigger makes the figure text look larger than its captions), font family (Arial -- PLOS allows "Arial, Times,
or Symbol" only), and line width (~0.2mm, PLOS's own guidance) come from PLOS's figure spec
(https://journals.plos.org/ploscompbiol/s/figures). But figure WIDTH here is calibrated against how
the figures are actually placed in the paper's LaTeX source, NOT PLOS's spec for the submitted
file's own resolution -- those are two different numbers, and conflating them is what makes text
come out too small. PLOS's 5.2in/7.5in figures describe the physical size PLOS's own production
pipeline re-embeds a submitted file at (from its pixel dimensions and DPI) -- that's irrelevant to
how the file displays in the author's own compiled draft, which is governed entirely by whatever
\\includegraphics[width=...] fraction of \\linewidth the author actually uses. Since fonts are drawn
into the raster/vector at a fixed point size, LaTeX scaling the whole image down to fit that width
scales the fonts down with it -- so a figure must be authored at (or very near) its ACTUAL final
display width for an 8-12pt font to still read as 8-12pt on the page.

This paper places third-page figures at 0.32\\linewidth (three sub-figures across), half-page ones
at 0.48\\linewidth and full-page ones at \\linewidth, in the PLOS LaTeX template (\\linewidth ~= 6.5in
for that template's single-column body text):
  - third-page width = 0.32 * 6.5in ~= 2.08in
  - half-page width  = 0.48 * 6.5in ~= 3.12in
  - full-page width  = 1.00 * 6.5in ~= 6.5in
A figure authored at one width and placed at another has its fonts scaled by the ratio -- a
half-page figure placed three-across comes out at 2/3 size, well under PLOS's 8pt floor -- so a
script whose output can go in either slot takes a --width option (see PAGE_WIDTHS_IN).
If the actual compiled PDF ever shows these noticeably larger or smaller than intended, that means
\\linewidth in the real document differs from 6.5in -- recalculate both widths from the actual value
rather than adjusting font sizes to compensate (that would just reintroduce the same mismatch this
module exists to avoid).

HALF_PAGE_HEIGHT_IN gives every ordinary (non-map) single-panel half-page plot the same 4:3 aspect
ratio, so they read as one consistent set rather than each script's own inherited shape -- a
choropleth/map script is the one legitimate exception (its aspect ratio is fixed by real geography,
not a free style choice), and multi-panel figures instead pick a height from their own per-panel
content.

Usage: call apply_style() once near the top of main(), before creating any figure, and use
HALF_PAGE_WIDTH_IN / FULL_PAGE_WIDTH_IN / THIRD_PAGE_WIDTH_IN (plus the matching *_HEIGHT_IN where
relevant) as the figure size when calling plt.subplots(figsize=...).
"""

import matplotlib.pyplot as plt

# See the module docstring for why these are NOT simply PLOS's raw submission-file spec numbers.
_ASSUMED_LATEX_LINEWIDTH_IN = 6.5
THIRD_PAGE_WIDTH_IN = 0.32 * _ASSUMED_LATEX_LINEWIDTH_IN
HALF_PAGE_WIDTH_IN = 0.48 * _ASSUMED_LATEX_LINEWIDTH_IN
FULL_PAGE_WIDTH_IN = _ASSUMED_LATEX_LINEWIDTH_IN
HALF_PAGE_HEIGHT_IN = HALF_PAGE_WIDTH_IN * 0.75  # shared 4:3 aspect for ordinary single-panel plots
# Squarer than 4:3: at 2.08in wide, a 4:3 plot is only ~1.56in tall, and the fixed-size text
# (title, x label, ticks) would leave almost no height for the data.
THIRD_PAGE_HEIGHT_IN = THIRD_PAGE_WIDTH_IN * 0.9

# --width choices for scripts whose figure can be placed in more than one slot: (width, height) in
# inches for a single-panel plot.
PAGE_WIDTHS_IN = {
    "third": (THIRD_PAGE_WIDTH_IN, THIRD_PAGE_HEIGHT_IN),
    "half": (HALF_PAGE_WIDTH_IN, HALF_PAGE_HEIGHT_IN),
}

FONT_TICK = 8
FONT_LABEL = 8
FONT_LEGEND = 8
FONT_TITLE = 9

# PLOS's ~0.2mm line-width guidance, rounded to a value matplotlib's linewidth (in points) can hit
# exactly; thin spines/ticks read as crisp rather than heavy at these figure sizes.
AXES_LINEWIDTH = 0.5


def apply_style():
    """Set the shared rcParams for a PLOS-Computational-Biology-ready figure. Call once, before
    creating any figure. Data line widths (ax.plot(..., lw=...), etc.) are each script's own call
    and not overridden here, since those vary meaningfully by plot (e.g. a trend line vs. a thin
    reference line) -- only the shared baseline (fonts, spine/tick width, output DPI) is set.
    """
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            # Arial is PLOS's specified sans-serif; Helvetica/DejaVu Sans are near-identical
            # fallbacks so rendering still looks right on a system without Arial installed.
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": FONT_LABEL,
            "axes.titlesize": FONT_TITLE,
            "axes.labelsize": FONT_LABEL,
            "xtick.labelsize": FONT_TICK,
            "ytick.labelsize": FONT_TICK,
            "legend.fontsize": FONT_LEGEND,
            "axes.linewidth": AXES_LINEWIDTH,
            "xtick.major.width": AXES_LINEWIDTH,
            "ytick.major.width": AXES_LINEWIDTH,
            "grid.linewidth": AXES_LINEWIDTH,
            # PLOS requires 300-600 dpi; every script's plt.savefig(...) picks this up automatically
            # even where the call site doesn't pass dpi= explicitly.
            "savefig.dpi": 300,
        }
    )
