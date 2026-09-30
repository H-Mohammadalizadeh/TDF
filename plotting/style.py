"""Publication style of the paper's figures (IEEE two-column, Times New Roman).

Figures are drawn at their printed size (column width 3.5 in), so the text keeps its point size
when the PDF is included at 100 %. `save` checks the layout before it writes a figure: no text may
extend beyond the figure, and no framed legend may cover a data line or marker.

Text extents depend on the font. Without Times New Roman, matplotlib substitutes another serif
font; the figures then differ from the published ones and a failed layout check is reported as a
warning instead of an error.
"""
import functools
import sys

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.ticker import NullLocator  # noqa: E402
from matplotlib.transforms import Bbox  # noqa: E402

COLW = 3.5  # IEEEtran conference \columnwidth = 252 pt
FS = 7.5    # base font size in points (captions are 8 pt)
FONT = 'Times New Roman'

plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': [FONT, 'Times', 'Nimbus Roman', 'STIXGeneral', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': FS, 'axes.labelsize': FS, 'axes.titlesize': FS,
    'legend.fontsize': FS - 0.5, 'xtick.labelsize': FS - 0.5, 'ytick.labelsize': FS - 0.5,
    'axes.linewidth': .5, 'lines.linewidth': .9, 'lines.markersize': 4.4,
    'lines.markeredgewidth': .7,
    'xtick.major.size': 2, 'ytick.major.size': 2, 'xtick.minor.size': 1.2, 'ytick.minor.size': 1.2,
    'xtick.major.width': .5, 'ytick.major.width': .5, 'xtick.minor.width': .4, 'ytick.minor.width': .4,
    'xtick.major.pad': 1.5, 'ytick.major.pad': 1.5, 'axes.labelpad': 1.5, 'axes.titlepad': 2.5,
    'xtick.direction': 'out', 'ytick.direction': 'out',
    'legend.frameon': False, 'legend.handlelength': 1.5, 'legend.handletextpad': .4,
    'legend.columnspacing': .9, 'legend.borderaxespad': .2, 'legend.labelspacing': .25,
    'legend.borderpad': .3,
    'pdf.fonttype': 42, 'ps.fonttype': 42, 'savefig.facecolor': 'white', 'axes.axisbelow': True,
    'grid.color': '#e4e4e4', 'grid.linewidth': .4,
})

# Larger text at the same printed size, used by all three figures.
LARGE = {'axes.labelsize': FS + 1.5, 'xtick.labelsize': FS + 1, 'ytick.labelsize': FS + 1,
         'legend.fontsize': FS + .5}

# Reconfiguration-time distributions: legend names and marker shapes (Pareto-3/2 shares the Pareto marker).
NAME = {'det': 'Deterministic', 'exp': 'Exponential', 'gamma': 'Gamma',
        'tnorm': 'Truncated normal', 'lognormal': 'Lognormal', 'pareto': 'Pareto',
        'pareto_heavy': 'Pareto 3/2'}
MARK = {'det': 's', 'exp': 'o', 'gamma': '^', 'tnorm': 'D', 'lognormal': 'v', 'pareto': 'P',
        'pareto_heavy': 'P'}

INK = '#222222'
BLUE = '#2E5C8A'  # steel blue: the window constraint; the certified rate range; E[tau_12] < E[tau_21]
RED = '#9E2A2B'   # brick red: the phase constraint; the OJF counterexample; E[tau_12] > E[tau_21]
# The target: a thin black line with small dashes, above shaded zones and below data.
TARGET_STYLE = dict(color=INK, lw=.6, ls=(0, (1.4, 1.1)), zorder=1.5)
FRAMED = dict(frameon=True, fancybox=False, framealpha=1, facecolor='white', edgecolor='#555555')

PDF_METADATA = {'Creator': 'tail-delay-feasibility', 'CreationDate': None}  # no date: reproducible PDFs


def font_available():
    """Whether matplotlib finds Times New Roman, the font of the published figures."""
    try:
        font_manager.findfont(font_manager.FontProperties(family=FONT), fallback_to_default=False)
    except ValueError:
        return False
    return True


EXACT_FONT = font_available()


def figure(width, height, **adjust):
    """A figure of the printed size in inches, with the subplot parameters `adjust`."""
    fig = plt.figure(figsize=(width, height))
    fig.subplots_adjust(**adjust)
    return fig


def large_text(draw):
    """Decorator: draw and save the figure with the LARGE text sizes."""
    @functools.wraps(draw)
    def wrapped(*args, **kwargs):
        with plt.rc_context(LARGE):
            return draw(*args, **kwargs)
    return wrapped


def tint(color, a=.3):
    """`color` at opacity `a` over white, as a solid color (overlaps stay clean)."""
    rgb = [int(color[i:i + 2], 16) for i in (1, 3, 5)]
    return '#%02X%02X%02X' % tuple(round(255 - a * (255 - v)) for v in rgb)


def tidy(ax):
    """Horizontal grid lines behind the data; no top and right spines."""
    ax.grid(False)
    ax.grid(axis='y', which='major')
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def no_minor(ax):
    """No minor ticks on either axis."""
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())


def target_line(ax):
    """The 1 % target on an axis in percent."""
    ax.axhline(1.0, **TARGET_STYLE)


def framed_legend(fig, handles, ncol, **kw):
    """One framed legend with a thin frame, centered at the top of the figure."""
    legend = fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, .995), ncol=ncol,
                        borderaxespad=0, **kw, **FRAMED)
    legend.get_frame().set_linewidth(.5)


def _clipped_texts(fig):
    """The visible texts (labels, tick labels in view, legend entries) that extend beyond the figure."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += [ax.xaxis.label, ax.yaxis.label, ax.title]
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            texts += [t.label1 for t in axis.get_major_ticks() if lo <= t.get_loc() <= hi]
        if ax.get_legend() is not None:
            texts += ax.get_legend().get_texts()
    for legend in fig.legends:
        texts += legend.get_texts()
    clipped = []
    for text in texts:
        if text.get_visible() and text.get_text():
            box = text.get_window_extent(renderer)
            if (box.x0 < -.5 or box.y0 < -.5 or box.x1 > fig.bbox.width + .5
                    or box.y1 > fig.bbox.height + .5):
                clipped.append((text, box))
    return clipped


def legend_covers(fig, legend, axes, pad_pt=1.0):
    """The first data line of `axes` whose stroke or markers come within `pad_pt` points of the frame of
    `legend`, else None."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    scale = fig.dpi / 72
    box = legend.get_frame().get_window_extent(renderer)
    for line in (line for ax in axes for line in ax.lines):
        if not line.get_visible():
            continue
        path = line.get_transform().transform_path(line.get_path())
        has_marker = line.get_marker() not in (None, 'None', '', ' ', 'none')
        reach = (max(line.get_linewidth(), line.get_markersize() if has_marker else 0) / 2 + pad_pt) * scale
        grown = Bbox.from_extents(box.x0 - reach, box.y0 - reach, box.x1 + reach, box.y1 + reach)
        drawn = line.get_linestyle() not in ('None', 'none', '', ' ')
        hit = drawn and len(path.vertices) > 1 and path.intersects_bbox(grown, filled=False)
        if has_marker and not hit:
            hit = any(grown.contains(x, y) for x, y in path.vertices)
        if hit:
            return line
    return None


def check_layout(fig, name):
    """Raise on clipped text or on a framed legend over data (an axes legend is checked against its axes,
    a figure legend against all axes); with a substitute font, report the problems as warnings."""
    problems = [f'Clipped text in {name}: {text.get_text()!r}; bbox={box.bounds}'
                for text, box in _clipped_texts(fig)]
    pairs = [(ax.get_legend(), [ax]) for ax in fig.axes] + [(legend, fig.axes) for legend in fig.legends]
    for legend, axes in pairs:
        if legend is None or not legend.get_frame_on():
            continue
        line = legend_covers(fig, legend, axes)
        if line is not None:
            problems.append(f'Legend covers data in {name}: line {line.get_label()!r} color={line.get_color()!r}')
    if problems and EXACT_FONT:
        raise ValueError('; '.join(problems))
    for problem in problems:
        print(f'warning: {problem} (substitute font)', file=sys.stderr)


def save(fig, folder, name):
    """Check the layout, then write `name`.pdf and a 300 dpi `name`.png into `folder`."""
    check_layout(fig, name)
    fig.savefig(folder / f'{name}.pdf', metadata={'Title': name, **PDF_METADATA})
    fig.savefig(folder / f'{name}.png', dpi=300)
    plt.close(fig)
