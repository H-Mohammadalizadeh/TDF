"""The paper's three evaluation figures, drawn from the stored results at their printed size.

fig3_where        where targets are infeasible: L_LP of the six distributions at three loads over the
                  zones that the window constraint and the phase constraint rule out alone, and L_LP
                  versus the deadline for gamma reconfiguration
fig4_bracket      the certified interval [L_safe, U^+] of the closed-form case with the simulated OJF
                  tails, and the interval width for three distributions, with U^+ and with U_0
fig5_scheduling   the certified range of the total reconfiguration rate versus the target with the OJF
                  counterexample, and kappa_L of the directional case under service and reconfiguration
                  changes
"""
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

import data
import style as s
from style import BLUE, COLW, INK, MARK, NAME, RED


# where targets are infeasible

# Equation numbers of the window constraint and the phase constraint in the manuscript; the legend
# names the two zones after them.
EQ_WINDOW = 6
EQ_PHASE = 9

BAND = '#f0f0f0'  # the shaded window behind each load group
MS = 4.3          # marker size in points


def _zone(ax, x, top, color, z):
    """A tinted zone below the bound `top` (what one constraint rules out alone), with a thin edge on top."""
    ax.fill_between(x, [1e-12] * len(x), top, color=s.tint(color), lw=0, zorder=z)
    ax.plot(x, top, color=color, lw=.7, solid_capstyle='butt', zorder=z + .01)


def _mark(infeasible):
    """A filled marker with a thin white rim if the target is certified infeasible, else an open marker."""
    if infeasible:
        return dict(color=INK, mfc=INK, mec='white', mew=.25, ms=MS + .5)
    return dict(color=INK, mfc='white', mec=INK, mew=.6, ms=MS)


@s.large_text
def fig3_where(res, out):
    """Left: L_LP at d = 2 of the six distributions at three loads (markers), over the zones that the
    window constraint (blue; it depends on the mean reconfiguration time only) and the phase constraint
    (red) rule out alone. Right: L_LP versus the deadline for gamma reconfiguration at the same loads.
    N = 4 and the 1 % target throughout."""
    rows = [r for r in res.core() if r['n'] == '4']
    loads = ('1/10', '1/2', '9/10')
    fig = s.figure(COLW, 1.60, left=.135, right=.995, bottom=.206, top=.666, wspace=.36)
    axa, axb = fig.subplots(1, 2, gridspec_kw={'width_ratios': [1.15, 1]})
    for ax in (axa, axb):
        s.tidy(ax)
        ax.set_yscale('log')
        s.target_line(ax)

    # left: one group per load, the six distributions side by side over their zones
    width = .145  # spacing within a group; groups are one unit apart
    for g, rho in enumerate(loads):
        cells = [data.one(rows, profile=p, d='2', rho=rho) for p in data.PROFILES]
        window = data.bound(cells[0], 'base_windows')  # one zone: it depends on the mean only
        assert all(abs(data.bound(r, 'base_windows') - window) <= 1e-9 * max(1, window) for r in cells), rho
        phase = [data.bound(r, 'base_phases') for r in cells]
        axa.axvspan(g - .45, g + .45, color=BAND, lw=0, zorder=0)
        z_window, z_phase = (1.3, 1.2) if window < max(phase) else (1.2, 1.3)  # the lower zone on top
        _zone(axa, [g - .42, g + .42], [window, window], BLUE, z_window)
        for k, (p, r) in enumerate(zip(data.PROFILES, cells)):
            x = g + (k - 2.5) * width
            _zone(axa, [x - .055, x + .055], [phase[k]] * 2, RED, z_phase)
            axa.plot(x, data.bound(r), marker=MARK[p], ls='none', zorder=3, **_mark(data.excluded(r)))
    axa.set_xlim(-.5, len(loads) - .5)
    axa.set_xticks(range(len(loads)), [f'{data.q(rho):g}' for rho in loads])
    axa.set_xlabel(r'$\rho$')
    axa.set_ylabel(r'$L_{\mathrm{LP}}$ [%]')
    axa.set_ylim(.02, 60)
    axa.set_yticks([.1, 1, 10], [r'$10^{-1}$', r'$10^{0}$', r'$10^{1}$'])
    s.no_minor(axa)

    # right: L_LP versus the deadline for gamma reconfiguration, one line per load
    floor = 1e-6  # the lowest tick; smaller values are drawn there
    xb = np.arange(len(data.DS))
    styles = [(0, (3.2, 1.2, 1, 1.2)), (0, (4, 1.5)), '-']  # dash-dot, dashed, solid: none like the target
    for rho, ls in zip(loads, styles):
        cells = [data.one(rows, profile='gamma', d=d, rho=rho) for d in data.DS]
        y = [max(data.bound(r), floor) for r in cells]
        axb.plot(xb, y, color=INK, ls=ls, lw=.8, zorder=2)
        for xi, yi, r in zip(xb, y, cells):
            axb.plot(xi, yi, marker=MARK['gamma'], ls='none', zorder=3, **_mark(data.excluded(r)))
    axb.set_xticks(xb, data.DS)
    axb.set_xlabel('$d$')
    axb.set_ylim(floor * .6, 60)
    axb.set_yticks([1e-6, 1e-4, 1e-2, 1], [r'$10^{-6}$', r'$10^{-4}$', r'$10^{-2}$', r'$10^{0}$'])
    s.no_minor(axb)

    # one framed legend for both panels: the distributions, the two zones and the target, the loads
    handles = [Line2D([], [], color=INK, marker=MARK[p], ls='none', ms=MS, label=NAME[p]) for p in data.PROFILES]
    handles += [Patch(facecolor=s.tint(BLUE), edgecolor=BLUE, lw=.7, label=f'Eq. ({EQ_WINDOW})'),
                Patch(facecolor=s.tint(RED), edgecolor=RED, lw=.7, label=f'Eq. ({EQ_PHASE})'),
                Line2D([], [], label=r'$\varepsilon_i=1\%$', **s.TARGET_STYLE)]
    handles += [Line2D([], [], color=INK, ls=ls, lw=.8, label=rf'$\rho={data.q(rho):g}$')
                for rho, ls in zip(loads, styles)]
    s.framed_legend(fig, handles, 4, handlelength=1.4)
    s.save(fig, out, 'fig3_where')


# certified intervals

DEADLINE_COLOR = {'2': '#7E8587', '3': '#C9A227'}  # slate gray, mustard yellow: not the constraint colors
GAP_N, GAP_D = 2, '2'  # the setting of the interval widths in the right panel
GAP_MS = MS + .5       # as large as the filled markers of `_mark`


@s.large_text
def fig4_bracket(res, out):
    """Left: the certified interval [L_safe, U^+] of the closed-form case (N = 2, exponential
    reconfiguration) versus the load, one band per deadline, with the simulated OJF tails. Right: the
    width U^+ - L_safe at N = 2 and d = 2 for three distributions, and the width with U_0 in place of U^+."""
    rows = res.brackets(data.BENCHMARK)
    sims = res.ojf_rows()
    fig = s.figure(COLW, 1.60, left=.13, right=.985, bottom=.218, top=.747, wspace=.47)
    axa, axb = fig.subplots(1, 2)

    # left: the interval as a band per deadline (its upper edge U^+ is the exact miss probability of OJF)
    # and the simulated OJF tails (x)
    s.tidy(axa)
    axa.set_xscale('log')
    axa.set_yscale('log')
    for d, color in DEADLINE_COLOR.items():
        sel = sorted([r for r in rows if r['d'] == d], key=lambda r: r['rho'])
        x = np.array([r['rho'] for r in sel])
        lower = np.array([r['safe'] for r in sel])
        upper = np.array([r['u_exact'] for r in sel])
        axa.fill_between(x, lower, upper, color=s.tint(color, .35), lw=0, zorder=1)
        axa.plot(x, lower, color=color, lw=.9, zorder=3)
        axa.plot(x, upper, color=color, lw=.9, zorder=3)
        for v in sims:
            tail = next(t for t in v['tails'] if abs(t['threshold_f64'] - float(d)) < 1e-9)
            axa.plot([v['rho_f64']], [100 * tail['estimate']], marker='x', ms=3.2, mew=.6, color=INK, ls='none',
                     zorder=5)
    s.target_line(axa)
    axa.set_xlim(6e-5, .6)
    axa.set_ylim(.1, 100)
    axa.set_xticks([1e-4, 1e-3, 1e-2, 1e-1], [r'$10^{-4}$', r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    axa.set_yticks([.1, 1, 10, 100], [r'$0.1$', r'$1$', r'$10$', r'$100$'])
    s.no_minor(axa)
    axa.set_xlabel(r'$\rho$')
    axa.set_ylabel(r'$m$ [%]')

    # right: the width U^+ - L_safe, one line per distribution, over the width U_0 - L_safe; U_0 is drawn
    # once, for gamma: within the axes the three distributions agree within 5 %, and coinciding lines
    # would draw bolder
    s.tidy(axb)
    axb.set_xscale('log')
    axb.set_yscale('log')
    series = data.gap_series(res.brackets(data.LIGHT_LOAD), res.brackets(data.FINITE_LOAD), GAP_D)
    dists = ('det', 'gamma', 'pareto_heavy')
    loads = sorted({k[2] for k in series if k[1] == GAP_N})
    at = [[series[(p, GAP_N, x)] for p in dists] for x in loads]
    u0 = [[r['u_simple'] - r['safe'] for r in rs] for rs in at]
    assert all(max(v) <= 1.05 * min(v) for v in u0 if max(v) <= 25), 'U_0 widths differ visibly'
    axb.plot(loads, [v[dists.index('gamma')] for v in u0], color=INK, lw=.8, zorder=2)
    for k, p in enumerate(dists):
        axb.plot(loads, [rs[k]['gap'] for rs in at], color=INK, lw=.8, marker=MARK[p], ms=GAP_MS,
                 mfc=INK, mec='white', mew=.25, zorder=4)
    axb.set_xlim(7e-4, .45)
    axb.set_ylim(.08, 25)  # U_0 leaves the top at high load; the three widths get the room
    axb.set_xticks([1e-3, 1e-2, 1e-1], [r'$10^{-3}$', r'$10^{-2}$', r'$10^{-1}$'])
    axb.set_yticks([.1, 1, 10], [r'$0.1$', r'$1$', r'$10$'])
    s.no_minor(axb)
    axb.set_xlabel(r'$\rho$')
    axb.set_ylabel('width [pp]')

    # one legend for both panels in two rows; its columns fill top-down: the deadlines, the left panel's
    # marks, then the right panel (distributions and U_0)
    handles = [Patch(facecolor=s.tint(color, .35), edgecolor=color, lw=.7, label=f'$d={d}$')
               for d, color in DEADLINE_COLOR.items()]
    handles += [Line2D([], [], color=INK, marker='x', ms=3.2, mew=.6, ls='none', label='simulated'),
                Line2D([], [], label=r'$\varepsilon_i=1\%$', **s.TARGET_STYLE)]
    handles += [Line2D([], [], color=INK, lw=.8, marker=MARK[p], ms=GAP_MS, mfc=INK, mec='white', mew=.25,
                       label=NAME[p]) for p in dists]
    handles.append(Line2D([], [], color=INK, lw=.8, label=r'$U_0$'))
    s.framed_legend(fig, handles, 4, columnspacing=1.0, handlelength=1.6)
    s.save(fig, out, 'fig4_bracket')


# scheduler requirements

@s.large_text
def fig5_scheduling(res, out):
    """Left (N = 2, exponential reconfiguration, rho = 0.05, d = 3): the certified range of the total
    reconfiguration rate versus the common target (band; dots at its endpoints), the excluded targets below
    L_safe (gray), the exact miss probability of OJF (dashed), and the counterexample: the rates of OJF
    (square) at the smallest target whose relaxation admits them although OJF misses it, with a segment up
    to the miss probability of OJF. Right: kappa_L of the directional case at rho = 0.1, h = 3/4 under
    faster class-1 service and under shorter reconfiguration times, for both reconfiguration directions
    (gray: kappa_L above 1, the targets are certified infeasible)."""
    fig = s.figure(COLW, 1.60, left=.13, right=.985, bottom=.262, top=.720, wspace=.40)
    axa, axb = fig.subplots(1, 2, width_ratios=[.70, 1.30])  # the right panel needs the width for its decimal ticks
    gray = '#e3e3e3'

    # left: the operating region in the (rate, target) plane
    region = res.region()
    cex = data.counterexample(res.sweep_rows())
    s.tidy(axa)
    axa.set_xscale('log')
    axa.set_yscale('log')
    rows = [r for r in region['region'] if not r['excluded']]
    eps = np.array([100 * r['eps_f64'] for r in rows])
    nu_lo = np.array([r['nu_lo'] for r in rows])
    nu_hi = np.array([r['nu_hi'] for r in rows])
    ymin = .15
    l_safe = 100 * region['bracket']['l_safe']['f64']
    u_plus = 100 * region['bracket']['u_plus']['f64']
    axa.axhspan(ymin, l_safe, color=gray, lw=0, zorder=0)
    axa.fill_betweenx(eps, nu_lo, nu_hi, color=s.tint(BLUE, .35), lw=0, zorder=1)
    axa.plot(nu_lo, eps, color=BLUE, lw=.8, zorder=3)
    axa.plot(nu_hi, eps, color=BLUE, lw=.8, zorder=3)
    axa.plot(np.r_[nu_lo, nu_hi], np.r_[eps, eps], ls='none', marker='o', ms=1.4, mew=0, color=BLUE, zorder=4)
    axa.axhline(u_plus, color=INK, lw=.7, ls=(0, (3, 1.3)), zorder=2)
    nu_ojf = cex['reference_rates']['nu_total']['f64']
    eps_cex = 100 * cex['eps_f64']
    axa.plot([nu_ojf, nu_ojf], [eps_cex, u_plus], color=RED, lw=.8, zorder=5)
    axa.plot([nu_ojf], [eps_cex], marker='s', ms=3.4, mfc='white', mec=INK, mew=.8, ls='none', zorder=6)
    axa.set_xlim(8e-3, 3.0)
    axa.set_ylim(ymin, 20)
    axa.set_xticks([.01, .1, 1], ['0.01', '0.1', '1'])
    axa.set_yticks([.2, 1, 10], ['0.2', '1', '10'])
    s.no_minor(axa)
    axa.set_xlabel(r'$\nu$')
    axa.set_ylabel(r'$\varepsilon$ [%]')

    # right: kappa_L of the base design (factor 1) under two separate families of changes, each given as
    # (variants in order, x positions, tick labels, group title)
    s.tidy(axb)
    axb.set_yscale('log')
    dj = res.directional()
    iv = {(r['orientation'], r['variant']): r for r in dj['interventions']}
    groups = [(['base', 'service_class1_x5/4', 'service_class1_x3/2', 'service_class1_x2'], [0, 1, 2, 3],
               ['1', '1.25', '1.5', '2'], r'$\mu_1\times$'),
              (['base', 'reconfiguration_x4/5', 'reconfiguration_x3/5', 'reconfiguration_x2/5'], [3.9, 4.9, 5.9, 6.9],
               ['1', '0.8', '0.6', '0.4'], r'$\tau\times$')]
    looks = {'matrix': dict(color=BLUE, mfc=BLUE, ls='-'), 'transpose': dict(color=RED, mfc='white', ls=(0, (3, 1.2)))}
    axb.axhspan(1.0, 100, color=gray, lw=0, zorder=0)
    for variants, xs, _, _ in groups:
        for orientation, look in looks.items():
            y = [iv[(orientation, v)]['kappa_lower'] for v in variants]
            axb.plot(xs, y, color=look['color'], lw=.8, ls=look['ls'], zorder=2)
            axb.plot(xs, y, marker='o', ms=3.0, mfc=look['mfc'], mec=look['color'], mew=.8, ls='none', zorder=5)
    axb.axvline(3.45, color='#bbbbbb', lw=.5, zorder=0)
    s.target_line(axb)
    axb.set_xlim(-.35, 7.3)
    axb.set_ylim(.1, 3)
    axb.set_xticks(groups[0][1] + groups[1][1], groups[0][2] + groups[1][2])  # decimal factors
    trans = axb.get_xaxis_transform()
    for _, xs, _, title in groups:  # the group titles act as axis labels
        axb.text((xs[0] + xs[-1]) / 2, -.26, title, transform=trans, ha='center', va='top',
                 fontsize=s.LARGE['axes.labelsize'])
    axb.set_yticks([.1, .3, 1, 3], ['0.1', '0.3', '1', '3'])
    s.no_minor(axb)
    axb.set_ylabel(r'$\kappa_L$')

    # one legend for both panels; its columns fill top-down: the shaded regions, the two OJF entries, the
    # two directions
    handles = [Patch(facecolor=s.tint(BLUE, .35), edgecolor=BLUE, lw=.7, label=r'$[\nu_{\min},\nu_{\max}]$'),
               Patch(facecolor=gray, edgecolor='#9a9a9a', lw=.5, label='excluded'),
               Line2D([], [], color=INK, lw=.7, ls=(0, (3, 1.3)), label=r'$\widetilde m$ of OJF'),
               Line2D([], [], color=INK, marker='s', ms=3.4, mfc='white', mew=.8, ls='none', label='OJF rates'),
               Line2D([], [], color=BLUE, marker='o', ms=3.0, mew=.8, lw=.8,
                      label=r'$\mathbb{E}[\tau_{12}]<\mathbb{E}[\tau_{21}]$'),
               Line2D([], [], color=RED, marker='o', ms=3.0, mfc='white', mew=.8, lw=.8, ls=(0, (3, 1.2)),
                      label=r'$\mathbb{E}[\tau_{12}]>\mathbb{E}[\tau_{21}]$')]
    s.framed_legend(fig, handles, 3, fontsize=s.LARGE['axes.labelsize'], columnspacing=1.0, handlelength=1.6)
    s.save(fig, out, 'fig5_scheduling')


FIGURES = (fig3_where, fig4_bracket, fig5_scheduling)
