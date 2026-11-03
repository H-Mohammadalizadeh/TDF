"""The three figures of Section VI, drawn from the tables of scripts/tables.sh (plotting only).

    python3 plotting/figures.py [--tables results/tables] [--out figures]

fig_verdicts  Fig. 1: verdicts for four classes and a 1 % target. Left: mean reconfiguration time
              0.5 s, edges for deterministic (solid) and Pareto (dashed) reconfiguration times.
              Right: the feasible edge as reconfiguration becomes 2, 10 and infinitely many times
              faster (deterministic times).
fig_gap       Fig. 2: the gap between the certificates for four classes at load 0.9 against the mean
              reconfiguration time. Left: deterministic reconfiguration, I/4 <= d <= 3I/4 (shaded),
              line at d = I/2. Right: d = I for three distributions.
fig_nclasses  Fig. 3: verdicts for a 1 % target against the number of classes (deterministic
              reconfiguration, mean 0.5 s) at loads 0.5 and 0.9, with the mean time away I (dotted).
"""
import argparse
import csv
from fractions import Fraction
from pathlib import Path

import numpy as np

import style as s
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.patheffects import withStroke

INK, BLUE, RED = s.INK, s.BLUE, s.RED
FILL = {'infeasible': s.tint(RED, .28), 'undetermined': s.tint('#7E8587', .26),
        'feasible': s.tint(BLUE, .28)}


def rows(folder, name):
    with (folder / name).open() as handle:
        return list(csv.DictReader(handle))


# ---- Fig. 1 ----

EDGE = dict(color=INK, lw=.7, zorder=3)
DASH = (0, (3, 1.5))
HALO = [withStroke(linewidth=1.6, foreground='white')]
# feasible edges of deterministic reconfiguration: mean [s], contour label, load of the label
SPEEDUP = ((.5, r'1$\times$', .50), (.25, r'2$\times$', .62), (.05, r'10$\times$', .74))
LABEL_AT = (.50, .62, .74, .86)  # loads of the labels 1x, 2x, 10x and infinity (where the lines are apart)
BANDS = (.28, .28, .28, .28)     # one blue for every speed-up band; the contour lines tell them apart
X0, X1 = .05, .95
Y0, Y1 = 1, 100


def edge(table, law, mean, side):
    """Loads and interpolated 1 % thresholds of one edge (bracketed crossings only)."""
    pts = sorted((float(r['rho']), float(r['d_cross'])) for r in table
                 if r['law'] == law and abs(float(r['mean']) - mean) < 1e-12 and r['side'] == side
                 and r['status'] == 'bracketed')
    return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])


def on_grid(x, y, grid):
    """An edge on a fine load grid, linear in log(d) between the computed loads."""
    return np.exp(np.interp(grid, x, np.log(y)))


def regions(ax, x, lo, hi):
    """Certified infeasible below `lo`, undetermined between, certified feasible above `hi`, with
    solid edges."""
    ax.fill_between(x, Y0, lo, color=FILL['infeasible'], lw=0, zorder=0)
    ax.fill_between(x, lo, hi, color=FILL['undetermined'], lw=0, zorder=0)
    ax.fill_between(x, hi, Y1, color=FILL['feasible'], lw=0, zorder=0)
    ax.plot(x, lo, **EDGE)
    ax.plot(x, hi, **EDGE)


def axes_style(ax, label, top=Y1, ticks=(1, 3, 10, 30, 100), xlim=(X0, X1), xticks=(.1, .3, .5, .7, .9)):
    ax.set_xlim(*xlim)
    ax.set_ylim(Y0, top)
    ax.set_yscale('log')
    ax.set_xticks(xticks, [f'{t:g}' for t in xticks])
    ax.set_yticks(ticks, [f'{t:g}' for t in ticks])
    s.no_minor(ax)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_xlabel(r'$\rho$')
    ax.text(.5, .97, label, transform=ax.transAxes, ha='center', va='top', color=INK, zorder=6)


@s.large_text
def fig_verdicts(tables, out):
    table = rows(tables, 'edges.csv')
    closed = rows(tables, 'edges_closed.csv')
    fig = s.figure(s.COLW, 1.35, left=.115, right=.985, bottom=.235, top=.69, wspace=.08)
    axa, axb = fig.subplots(1, 2, sharey=True)
    grid = np.linspace(X0, X1, 400)

    # left: the verdicts at mean 0.5 s (deterministic), and the two edges for Pareto (dashed)
    regions(axa, grid, on_grid(*edge(table, 'det', .5, 'bound'), grid),
            on_grid(*edge(table, 'det', .5, 'witness'), grid))
    for side in ('bound', 'witness'):
        axa.plot(grid, on_grid(*edge(table, 'pareto', .5, side), grid), ls=DASH, **EDGE)
    axes_style(axa, r'$\bar\tau=0.5$ s')
    axa.set_ylabel(r'$d$ [s]')

    # the limits come first: the contour labels take their angle and gap from the final axes
    axes_style(axb, r'$\bar\tau\to0$')
    # right: the verdicts as the reconfiguration times shrink to zero, and where certified feasibility
    # starts at the current reconfiguration time and after reconfiguring 2 and 10 times faster
    rho = np.array([float(r['rho']) for r in closed])
    nec = np.interp(grid, rho, [float(r['d_nec']) for r in closed])
    suf = np.interp(grid, rho, [float(r['d_suf']) for r in closed])
    # the speed-up that certified feasibility needs, as a field over (load, threshold): the log of the
    # factor, interpolated in log d between the feasible edges at 1x, 2x and 10x and the edge of instant
    # reconfiguration (level 100, labelled infinity), larger below it. Below the edge of instant
    # reconfiguration nothing is filled.
    edges = {mean: on_grid(*edge(table, 'det', mean, 'witness'), grid) for mean, _, _ in SPEEDUP}
    logd = np.linspace(np.log10(Y0), np.log10(Y1), 500)
    field = np.empty((len(logd), len(grid)))
    for k in range(len(grid)):
        knots = np.log10([suf[k], edges[.05][k], edges[.25][k], edges[.5][k]])
        field[:, k] = np.interp(logd, np.concatenate([[knots[0] - 2], knots, [knots[-1] + 2]]),
                                np.log10([1e4, 100, 10, 2, 1, 1e-2]))
    levels = np.log10([1, 2, 10, 100])
    axb.contourf(grid, 10 ** logd, field, levels=np.concatenate([[field.min()], levels]),
                 colors=[s.tint(BLUE, a) for a in BANDS], zorder=0)
    # below the edge of instant reconfiguration: undetermined even then, and infeasible at every speed
    axb.fill_between(grid, Y0, nec, color=FILL['infeasible'], lw=0, zorder=0)
    axb.fill_between(grid, nec, suf, color=FILL['undetermined'], lw=0, zorder=0)
    axb.plot(grid, nec, **EDGE)
    # the speed-up contours in dark gray; the last one (infinity) is the solid edge of the undetermined region
    contours = axb.contour(grid, 10 ** logd, field, levels=levels, colors=['#4A4A4A'] * 3 + [INK],
                           linewidths=[.6, .6, .6, EDGE['lw']], zorder=3)
    texts = [text for _, text, _ in SPEEDUP] + [r'$\infty$']
    lines = [edges[mean] for mean, _, _ in SPEEDUP] + [suf]
    at = [(x, np.interp(x, grid, line)) for x, line in zip(LABEL_AT, lines)]
    axb.clabel(contours, levels, inline=True, inline_spacing=2, fontsize=.8 * (s.FS - .5), colors=INK, manual=at,
               fmt=dict(zip(contours.levels, texts)))

    # the legend fills its three columns top to bottom: verdicts in the first row, edge styles in the second
    handles = [Patch(facecolor=FILL['infeasible'], edgecolor=INK, lw=.5, label='certified infeasible'),
               Line2D([], [], label='deterministic', **EDGE),
               Patch(facecolor=FILL['undetermined'], edgecolor=INK, lw=.5, label='undetermined'),
               Line2D([], [], ls=DASH, label='Pareto', **EDGE),
               Patch(facecolor=FILL['feasible'], edgecolor=INK, lw=.5, label='certified feasible')]
    s.framed_legend(fig, handles, 3, columnspacing=1.0, handlelength=1.6)
    s.save(fig, out, 'fig_verdicts')


# ---- Fig. 2 ----

COLOR = {'det': '#2E5C8A', 'exp': '#9E2A2B', 'pareto': '#C9A227'}
MEANS = ['1/2', '5', '50', '500']
SHOWN = ['1/4', '1/2', '3/4']
MS = 3.6


def series(table, law, frac, rho='9/10'):
    """The rows of one (law, d/I) at load `rho`, ordered by the mean reconfiguration time."""
    found = {r['mean']: r for r in table if r['law'] == law and r['rho'] == rho and r['d_over_i'] == frac}
    return [float(Fraction(m)) for m in MEANS], [found[m] for m in MEANS]


def marked(law):
    """Line and marker style of one distribution: filled circles with a thin white edge (the color
    tells the distributions apart)."""
    return dict(color=COLOR[law], lw=.9, marker='o', ms=MS, mfc=COLOR[law], mec='white', mew=.25)


def log_axes(ax, label):
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(.36, 700)
    ax.set_ylim(.6, 50)
    ax.set_xticks([.5, 5, 50, 500], ['0.5', '5', '50', '500'])
    ax.set_yticks([1, 3, 10, 30], ['1', '3', '10', '30'])
    s.no_minor(ax)
    ax.set_xlabel(r'$\bar\tau$ [s]')
    ax.text(.5, .97, label, transform=ax.transAxes, ha='center', va='top', color=INK)


@s.large_text
def fig_gap(tables, out):
    table = rows(tables, 'gap.csv')
    fig = s.figure(s.COLW, 1.15, left=.115, right=.985, bottom=.30, top=.765, wspace=.10)
    axa, axb = fig.subplots(1, 2, sharey=True)

    # left: deterministic reconfiguration, thresholds below the mean time away
    x, _ = series(table, 'det', '1/2')
    widths = {f: [float(r['width']) for r in series(table, 'det', f)[1]] for f in SHOWN}
    lo = [min(widths[f][i] for f in SHOWN) for i in range(len(x))]
    hi = [max(widths[f][i] for f in SHOWN) for i in range(len(x))]
    axa.fill_between(x, lo, hi, color=s.tint(COLOR['det'], .45), lw=0, zorder=2)
    axa.plot(x, widths['1/2'], zorder=3, **marked('det'))
    log_axes(axa, r'$I/4\leq d\leq3I/4$')
    axa.set_ylabel('Gap [pp]')

    # right: threshold equal to the mean time away, three distributions
    for law in ('det', 'exp', 'pareto'):
        xs, rows_b = series(table, law, '1')
        axb.plot(xs, [float(r['width']) for r in rows_b], zorder=3, **marked(law))
    log_axes(axb, r'$d=I$')
    axb.tick_params(axis='y', left=False)

    handles = [Line2D([], [], label=s.NAME[law], **marked(law)) for law in ('det', 'exp', 'pareto')]
    s.framed_legend(fig, handles, 3, columnspacing=1.0, handlelength=1.6)
    s.save(fig, out, 'fig_gap')


# ---- Fig. 3 ----

DOT = (0, (1, 1.3))
N0, N1 = 2, 16
PANELS = ((.5, 24, [0, 10, 20]), (.9, 120, [0, 50, 100]))  # load, top [s], ticks


def class_edge(table, rho, side, status='bracketed'):
    """Rows of one edge with the given status, sorted by N: bracketed crossings (N, d_cross), wide
    brackets (N, d_lo, d_hi), or edges above the computed range (N, largest computed d)."""
    out = []
    for r in table:
        if abs(float(r['rho']) - rho) > 1e-9 or r['side'] != side or r['status'] != status:
            continue
        if status == 'bracketed':
            out.append((int(r['n']), float(r['d_cross'])))
        elif status == 'wide':
            out.append((int(r['n']), float(r['d_lo']), float(r['d_hi'])))
        else:
            out.append((int(r['n']), float(r['d_lo'])))
    return sorted(out)


@s.large_text
def fig_nclasses(tables, out):
    table = rows(tables, 'classes.csv')
    closed = rows(tables, 'classes_closed.csv')
    fig = s.figure(s.COLW, 1.35, left=.115, right=.965, bottom=.235, top=.69, wspace=.30)
    axes = fig.subplots(1, 2)
    for ax, (rho, top, ticks) in zip(axes, PANELS):
        bound, witness = class_edge(table, rho, 'bound'), class_edge(table, rho, 'witness')
        bx, by = np.array([p[0] for p in bound], float), np.array([p[1] for p in bound])
        wx, wy = np.array([p[0] for p in witness], float), np.array([p[1] for p in witness])
        ax.set_facecolor(FILL['undetermined'])
        ax.fill_between(bx, 0, by, color=FILL['infeasible'], lw=0, zorder=0)
        ax.fill_between(wx, wy, top, color=FILL['feasible'], lw=0, zorder=0)
        ax.plot(bx, by, color=INK, lw=.7, marker='o', ms=2.0, zorder=3, clip_on=False)
        ax.plot(wx, wy, color=INK, lw=.7, marker='o', ms=2.0, zorder=3, clip_on=False)
        for n_w, lo_w, hi_w in class_edge(table, rho, 'bound', 'wide'):
            ax.errorbar([n_w], [(lo_w + hi_w) / 2], yerr=[[(hi_w - lo_w) / 2]], fmt='none', ecolor=INK,
                        elinewidth=.7, capsize=1.5, capthick=.6, zorder=4, clip_on=False)
        for n_a, d_a in class_edge(table, rho, 'witness', 'above'):
            ax.plot([n_a], [d_a], ls='none', marker='^', ms=3.0, mfc='white', mec=INK, mew=.6, zorder=4,
                    clip_on=False)
        n = np.array([float(r['n']) for r in closed if abs(float(r['rho']) - rho) < 1e-9])
        away = np.array([float(r['I']) for r in closed if abs(float(r['rho']) - rho) < 1e-9])
        ax.plot(n, away, color=INK, lw=.7, ls=DOT, zorder=2)
        ax.set_xlim(N0, N1)
        ax.set_ylim(0, top)
        ax.set_xticks([2, 4, 8, 12, 16], ['2', '4', '8', '12', '16'])
        ax.set_yticks(ticks, [f'{t:g}' for t in ticks])
        s.no_minor(ax)
        for side in ('top', 'right'):
            ax.spines[side].set_visible(False)
        ax.set_xlabel(r'$N$')
        ax.text(.5, .97, rf'$\rho={rho:g}$', transform=ax.transAxes, ha='center', va='top', color=INK, zorder=6)
    axes[0].set_ylabel(r'$d$ [s]')
    handles = [Patch(facecolor=FILL['infeasible'], edgecolor=INK, lw=.5, label='certified infeasible'),
               Patch(facecolor=FILL['undetermined'], edgecolor=INK, lw=.5, label='undetermined'),
               Patch(facecolor=FILL['feasible'], edgecolor=INK, lw=.5, label='certified feasible'),
               Line2D([], [], color=INK, lw=.7, ls=DOT, label=r'mean time away $I$')]
    s.framed_legend(fig, handles, 2, columnspacing=1.0, handlelength=1.6)
    s.save(fig, out, 'fig_nclasses')


if __name__ == '__main__':
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--tables', type=Path, default=root / 'results' / 'tables')
    parser.add_argument('--out', type=Path, default=root / 'figures')
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    for draw in (fig_verdicts, fig_gap, fig_nclasses):
        draw(args.tables, args.out)
