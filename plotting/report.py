"""The numbers report: every result-derived number that the paper prints, with its value and source.

The macros come from `macros.compute`. The numbers written directly into the manuscript are
recomputed here with the rounding the paper uses and compared with the printed text; statements that
summarize results (for example, that no verdict is inconclusive) are checked the same way.
"""
import math
import re
from collections import Counter, namedtuple
from fractions import Fraction
from itertools import takewhile

import data
from display import pct
from macros import HIGH, MID, unique

# A number or statement of the text: as printed, as computed from the results, and whether they match.
Literal = namedtuple('Literal', 'printed value match what source timing')

# The parts of the paper in print order: (key, heading).
SECTIONS = [
    ('setup', 'Evaluation: Experimental Setup'),
    ('exclusion', 'Evaluation: When and Why Targets Are Excluded'),
    ('gap', 'Evaluation: Gap to Achievable Performance'),
    ('scheduling', 'Evaluation: Scheduler Requirements'),
    ('limits', 'Evaluation: Limitations and Cost'),
    ('cost', 'Appendix: Cost and Numerical Checks'),
    ('workload', 'Appendix: The Workload Bound'),
    ('holbmw', 'Appendix: The Adapted HOL-BMW Policy'),
]

WORDS = dict(enumerate('one two three four five six seven eight nine'.split(), start=1))  # counts as words
TIMING = ('coefficients', 'windows', 'lp', 'certificates', 'verification')  # the parts of one setting's run time
RUN_TIMES = 'numerics/runtime_N*.json'
DIRECTIONAL_ARGS = 'directional/manifest.json'
FINITE = f'{data.FINITE_LOAD}/bracket.json'


def number(printed, value, what, source, timing=False):
    """A number as printed in the paper and as computed from the results."""
    return Literal(printed, value, printed == value, what, source, timing)


def statement(printed, holds, evidence, what, source, timing=False):
    """A statement of the paper, whether the results support it, and the evidence."""
    return Literal(printed, evidence, bool(holds), what, source, timing)


def g(x):
    """`x` as a float in 'g' notation, e.g. 0.05."""
    return f'{float(x):g}'


def joined(values, form=str):
    """`values` formatted by `form`, separated by commas."""
    return ', '.join(form(v) for v in values)


def e_notation(v, digits=2):
    """`v` in e-notation as the paper prints it, e.g. 4.0e-3."""
    mantissa, exponent = f'{v:.{digits - 1}e}'.split('e')
    return f'{mantissa}e{int(exponent)}'


def cv_text(cv):
    """A coefficient of variation as the distribution table prints it."""
    if cv is None:
        return '∞'
    if abs(cv - 1 / math.sqrt(3)) < 1e-12:
        return '1/√3'
    return g(cv)


def per_target(grid):
    """The distinct numbers of grid rows per target."""
    return sorted(set(Counter(r['eps'] for r in grid).values()))


def _setup(res):
    grid = res.grid_rows()
    summary = res.read_json(data.GRID_SUMMARY)
    models = [res.certificate(name)['model'] for name in res.certificate_names()]
    certificates = f'{data.CERTIFICATES}/*.json'
    arrivals = all(Fraction(x) == 4 * Fraction(m['rho']) / m['n'] for m in models for x in m['lambda'])
    small = sorted({r['grid_step'] for r in grid if int(r['n']) <= 4})
    large = sorted({r['grid_step'] for r in grid if int(r['n']) > 4})
    out = [
        number('1890', joined(per_target(grid)), 'settings per target', data.GRID),
        number('99.9, 99, 95', joined((100 - 100 * data.q(e) for e in summary['epsilons']), g),
               'delay percentiles that the three targets correspond to', data.GRID_SUMMARY),
        number('2, 3, 4, 6, 8', joined(summary['ns']), 'configurations N', data.GRID_SUMMARY),
        number('0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.85, 0.9, 0.95', joined(map(data.q, summary['rhos']), g), 'loads ρ',
               data.GRID_SUMMARY),
        number('1, 2, 3, 4, 6, 8, 12', joined(summary['ds']), 'deadlines d [s]', data.GRID_SUMMARY),
        number('0.1, 1, 5', joined((100 * data.q(e) for e in summary['epsilons']), g), 'targets ε [%]',
               data.GRID_SUMMARY),
        number('4', joined(sorted({Fraction(x) for m in models for x in m['mu']}), g),
               'service rate μ_i [1/s] of every certificate', certificates),
        statement('4ρ/N', arrivals, f'λ_i = 4ρ/N in all {len(models)} certificates', 'arrival rate λ_i [1/s]',
                  certificates),
        number('1/2000 for N≤4, 1/1000 for N=6,8', f'{joined(small)} for N≤4, {joined(large)} for N=6,8',
               'grid step g [s]', data.GRID),
    ]
    # the distribution table: row name, printed parameters and CV, parameters from the results
    profiles = {p['profile']: p for p in res.read_json(data.DISTRIBUTIONS)['profiles']}
    means = sorted({p['law']['mean'][0] for p in profiles.values()} | {p['engine_mean'] for p in profiles.values()})
    law = {name: p['law']['exact'] for name, p in profiles.items()}
    tn, ln, pa, ph = law['tnorm'], law['lognormal'], law['pareto'], law['pareto_heavy']
    b2 = math.log(1 + data.q(ln['cv2']))  # ln(tau) ~ N(a, b^2) with the prescribed mean and CV^2
    a = math.log(data.q(ln['mean'])) - b2 / 2
    table = {
        'det': ('Deterministic', '1/2; CV 0', law['det']['value']),
        'exp': ('Exponential', 'rate 2; CV 1', f"rate {law['exp']['rate']}"),
        'gamma': ('Gamma', 'shape 3, rate 6; CV 1/√3', f"shape {law['gamma']['shape']}, rate {law['gamma']['rate']}"),
        'tnorm': ('Truncated normal', 'N(0.43, 0.34²); CV 1/√3',
                  f"N({data.q(tn['mu0']):.2f}, {data.q(tn['sigma0']):.2f}²)"),
        'lognormal': ('Lognormal', 'N(-0.84, 0.54²); CV 1/√3', f'N({a:.2f}, {math.sqrt(b2):.2f}²)'),
        'pareto': ('Pareto', 'shape 3, min. 1/3; CV 1/√3', f"shape {pa['alpha']}, min. {pa['xm']}"),
        'pareto_heavy': ('Pareto-3/2', 'shape 3/2, min. 1/6; CV ∞', f"shape {ph['alpha']}, min. {ph['xm']}"),
    }
    out.append(number('0.5', joined(means, g), 'mean reconfiguration time [s] of every distribution',
                      data.DISTRIBUTIONS))
    out += [number(printed, f"{value}; CV {cv_text(profiles[p]['cv'])}",
                   f'{row}: parameters and coefficient of variation', data.DISTRIBUTIONS)
            for p, (row, printed, value) in table.items()]
    # the protocol of the policy screen
    parts = res.screen_parts()
    envs = [e for part in parts for e in part['environments']]
    lam_min = [min(data.q(x) for x in e['model']['lambda']) for e in envs]
    runs = {p['runs'] for e in envs for p in e['policies'] if p['policy'] in data.PAPER_POLICIES}
    windows = all(e['window']['measure'] == max(20000, 5000 / lm) for e, lm in zip(envs, lam_min))
    out += [
        number('32', joined(sorted({len(part['seeds']) for part in parts} | runs)),
               'replications per policy and setting', data.SCREEN_FILES),
        number('2000', joined(sorted({e['window']['warmup'] for e in envs}), g), 'warm-up [s]', data.SCREEN_FILES),
        statement('max{20000, 5000/λ_min}', windows, f'holds in all {len(envs)} settings', 'observation window [s]',
                  data.SCREEN_FILES),
        number('5000', g(min(e['window']['measure'] * lm for e, lm in zip(envs, lam_min))),
               'fewest expected arrivals of the least frequent class in the observation window', data.SCREEN_FILES),
        number('95', joined(sorted({100 * (1 - part['alpha']) for part in parts}), g), 'bootstrap interval level [%]',
               data.SCREEN_FILES),
    ]
    return out


def _exclusion(res):
    grid = res.grid_rows()
    summary = res.read_json(data.GRID_SUMMARY)
    n4 = [r for r in res.core() if r['n'] == '4']

    def first_excluded_load(profile):
        return next(g(data.q(rho)) for rho in data.RHOS
                    if data.excluded(data.one(n4, profile=profile, d='2', rho=rho)))

    def excluded_through(rho):
        """The largest deadline up to which the gamma target is excluded at every deadline."""
        ds = list(takewhile(lambda d: data.excluded(data.one(n4, profile='gamma', d=d, rho=rho)), data.DS))
        return ds[-1] if ds else None

    ten = {p: data.bound(data.one(n4, profile=p, d='2', rho='1/10')) for p in ('gamma', 'tnorm', 'lognormal', 'pareto')}
    ordered = min(ten, key=ten.get) == 'tnorm' and max(ten, key=ten.get) == 'pareto'
    ninety = [data.bound(data.one(n4, profile=p, d='2', rho='9/10')) for p in data.PROFILES]
    largest_target = 100 * max(data.q(e) for e in data.EPSS)
    moments = {r['profile']: r['bracket']['verdict'] for r in res.bracket_rows(data.MOMENTS)}
    args = res.manifest(data.MOMENTS)
    counts = [sum(data.excluded(r) for r in grid if r['eps'] == e) for e in data.EPSS]
    return [
        number('0.05', first_excluded_load('exp'), 'first excluded load, exponential, N = 4, d = 2', data.GRID),
        number('0.7', first_excluded_load('det'), 'first excluded load, deterministic, N = 4, d = 2', data.GRID),
        statement('range from truncated normal to Pareto, on both sides of the target',
                  ordered and ten['tnorm'] < 1 < ten['pareto'], joined(f'{p} {pct(v)} %' for p, v in ten.items()),
                  'L_LP of the four distributions with equal mean and CV, N = 4, ρ = 0.1, d = 2', data.GRID),
        number('1/4000, 15/8', f"{joined(args['rhos'])}, {joined(args['ds'])}",
               'load and deadline of the moment-matched pair', f'{data.MOMENTS}/manifest.json'),
        statement('certified feasible for gamma, infeasible for Pareto',
                  moments == {'gamma': 'certified_feasible', 'pareto': 'certified_infeasible'},
                  joined(f'{p} {v}' for p, v in moments.items()), 'verdicts of the moment-matched pair',
                  f'{data.MOMENTS}/bracket.json'),
        statement('all six distributions give the same bound at ρ = 0.9', max(ninety) - min(ninety) < 1e-9,
                  f'spread {max(ninety) - min(ninety):.1e} pp', 'L_LP, N = 4, ρ = 0.9, d = 2', data.GRID),
        statement('above all three targets', min(ninety) > largest_target,
                  f'{min(ninety):.2f} % > {g(largest_target)} %', 'L_LP, N = 4, ρ = 0.9, d = 2', data.GRID),
        number('1', excluded_through('1/10'), 'largest excluded deadline, gamma, N = 4, ρ = 0.1', data.GRID),
        number('2', excluded_through('1/2'), 'largest excluded deadline, gamma, N = 4, ρ = 0.5', data.GRID),
        number('8', excluded_through('9/10'), 'largest excluded deadline, gamma, N = 4, ρ = 0.9', data.GRID),
        number('0.2', g(data.q(res.certificate(data.CERTIFICATE)['model']['rho'])),
               'load of the stored combination-only certificate', f'{data.CERTIFICATES}/{data.CERTIFICATE}.json'),
        number('963, 808, 656 of 1890', f'{joined(counts)} of {joined(per_target(grid))}',
               'excluded settings at the targets 0.1 %, 1 %, 5 %', data.GRID),
        statement('none is inconclusive', summary['inconclusive'] == 0,
                  f"{summary['inconclusive']} of {summary['points']}", 'inconclusive verdicts of the grid',
                  data.GRID_SUMMARY),
    ]


def _gap(res):
    bench = res.bracket_rows(data.BENCHMARK)
    args = res.manifest(data.BENCHMARK)
    finite = res.brackets(data.FINITE_LOAD)
    det = data.one(finite, profile='det', n=2, rho_exact='1/10', d='2')
    series = data.gap_series(res.brackets(data.LIGHT_LOAD), finite, '2')
    ratios = [series[('pareto_heavy', 2, x)]['gap'] / series[('gamma', 2, x)]['gap']
              for x in sorted({k[2] for k in series if k[1] == 2}) if .01 - 1e-12 <= x <= .1 + 1e-12]
    d2 = [r['bracket']['verdict'] for r in bench if r['d'] == '2']
    fin_args = res.manifest(data.FINITE_LOAD)
    sharper = sum(r['u_workload'] < r['u_simple'] for r in finite)
    mu = sorted({r['bracket']['homogeneous']['mu'] for r in bench})
    return [
        number('2, 0.5, 4', f"{joined(args['ns'])}, {g(data.q(args['mean']))}, {joined(mu)}",
               'N, mean reconfiguration time [s] and service rate [1/s] of the closed-form case',
               f'{data.BENCHMARK}/manifest.json, bracket.json'),
        statement('infeasible at every load (d = 2)', all(v == 'certified_infeasible' for v in d2),
                  f"{d2.count('certified_infeasible')} of {len(d2)} loads certified infeasible",
                  'verdicts of the closed-form case at d = 2', f'{data.BENCHMARK}/bracket.json'),
        statement('an order of magnitude', det['u_simple'] / det['u_workload'] >= 10,
                  f"U_0 / U^+ = {det['u_simple'] / det['u_workload']:.0f}",
                  'tightening by the workload bound, deterministic, N = 2, ρ = 0.1, d = 2', FINITE),
        number('2.1–2.5', f'{min(ratios):.2g}–{max(ratios):.2g}',
               'interval width of Pareto-3/2 over that of gamma, N = 2, d = 2, loads 0.01 to 0.1',
               'bracket/{light_load,finite_load}/bracket.json'),
        number('N ∈ {2, 4}, d ∈ {2, 3}', f"N ∈ {{{joined(fin_args['ns'])}}}, d ∈ {{{joined(fin_args['ds'])}}}",
               'configurations and deadlines of the finite-load set', f'{data.FINITE_LOAD}/manifest.json'),
        statement('the workload bound tightens U^+ in all settings', sharper == len(finite),
                  f'{sharper} of {len(finite)}', 'workload bound below U_0', FINITE),
    ]


def _scheduling(res):
    settings = {f"{a['n']}, {g(data.q(a['rho']))}, {a['d']}"
                for a in (res.manifest('rate_region'), res.manifest('target_sweep'))}
    region = [r for r in res.region()['region'] if not r['excluded']]
    exact_ends = sum(r['endpoints_exact'] for r in region)
    cex = data.counterexample(res.sweep_rows())
    dj = res.directional()
    args = res.manifest('directional')
    means = args['mean_matrix_rows_from']
    forward = {means[0][1], means[1][2], means[2][0]}  # 1 -> 2 -> 3 -> 1
    reverse = {means[1][0], means[2][1], means[0][2]}
    arrivals = sum(all(Fraction(x) == Fraction(r['rho']) * Fraction(a) * Fraction(m)
                       for x, a, m in zip(r['lambda'], args['shares'], args['mu'])) for r in dj['rows'])
    flips = sum(c['verdicts_differ'] for c in dj['comparison'])
    ivs = {(r['orientation'], r['variant']): r for r in dj['interventions']}
    orientations = ('matrix', 'transpose')

    def lowering(family, orientation):
        """The factors of a family of changes that lower kappa_L below one."""
        factors = {Fraction(v.split('_x')[1]) for _, v in ivs if v.startswith(family)}
        return [f for f in factors if ivs[(orientation, f'{family}_x{f}')]['kappa_lower'] < 1]

    service = max(min(lowering('service_class1', o)) for o in orientations)
    reconfiguration = sorted((max(lowering('reconfiguration', o)) for o in orientations), reverse=True)
    base = data.one(dj['rows'], orientation='matrix', rho='1/10', h='3/4')
    single_class = next(level for level in base['levels'] if level['level'] == 'base')['lb_max']
    mu1 = data.q(ivs[('matrix', 'service_class1_x5/4')]['mu'][0])
    upgraded = math.exp(-(mu1 - data.q(base['lambda'][0])) * data.q(base['deadlines'][0]))  # single-class bound
    return [
        number('2, 0.05, 3', ' / '.join(sorted(settings)), 'N, ρ, d of the rate range and the target sweep',
               'rate_region/manifest.json, target_sweep/manifest.json'),
        statement('all endpoints are exactly verified', exact_ends == len(region),
                  f'{exact_ends} of {len(region)} targets', 'endpoints of the rate range', data.REGION),
        number('5.4', f"{-math.log(cex['bracket']['u_plus']['f64']):.1f}",
               'exponent of the OJF miss probability at the counterexample', data.SWEEP),
        number('(4, 5, 6)', f"({joined(args['mu'])})", 'service rates μ of the directional case', DIRECTIONAL_ARGS),
        number('(1/2, 1/3, 1/6)', f"({joined(args['shares'])})", 'load shares a', DIRECTIONAL_ARGS),
        statement('λ_i = ρ a_i μ_i', arrivals == len(dj['rows']), f"{arrivals} of {len(dj['rows'])} settings",
                  'arrival rates of the directional case', data.DIRECTIONAL),
        number('h (2, 5/2, 3)', f"h ({joined(args['base_deadlines'])})", 'deadlines', DIRECTIONAL_ARGS),
        number('(0.5, 1, 2)', f"({joined((100 * data.q(e) for e in args['eps']), g)})", 'targets [%]',
               DIRECTIONAL_ARGS),
        number('3', re.search(r'shape (\S+),', args['law']).group(1), 'gamma shape of the reconfiguration times',
               DIRECTIONAL_ARGS),
        number('0.25, 1', f'{joined(map(data.q, forward), g)}, {joined(map(data.q, reverse), g)}',
               'mean reconfiguration time [s] along 1→2→3→1 and in reverse', DIRECTIONAL_ARGS),
        number('{0.05, 0.1, 0.2}, {0.5, 0.75, 1, 1.25, 1.5, 2}',
               f"{{{joined(map(data.q, args['rhos']), g)}}}, {{{joined(map(data.q, args['scales']), g)}}}",
               'loads ρ and deadline scales h', DIRECTIONAL_ARGS),
        statement('does not change the feasibility verdict', flips == 0,
                  f"{flips} of {len(dj['comparison'])} pairs differ", 'verdicts of the two directions',
                  data.DIRECTIONAL),
        number('1.25', g(service), 'smallest class-1 service factor with κ_L < 1 in both directions', data.DIRECTIONAL),
        number('0.8 or 0.6', ' or '.join(map(g, reconfiguration)),
               'largest reconfiguration-time factor with κ_L < 1, per direction', data.DIRECTIONAL),
        number('0.33', pct(100 * single_class, 2), 'largest single-class bound [%] (class 1), ρ = 0.1, h = 0.75',
               data.DIRECTIONAL),
        number('0.5', g(100 * data.q(base['eps'][0])), 'class-1 target [%]', data.DIRECTIONAL),
        number('0.075', pct(100 * upgraded, 2), 'single-class bound [%] of class 1 with μ_1 × 1.25', data.DIRECTIONAL),
    ]


def _run_times(res):
    """The run times of one setting over N, which the limitations paragraph and the appendix print: the
    shared numbers, the first and last N, and the shares of the coefficients and of the LP."""
    t = res.runtimes()
    first, last = min(t), max(t)
    total = {n: sum(times[k] for k in TIMING) for n, times in t.items()}
    coefficients = [times['coefficients'] / 1000 for times in t.values()]
    verification = max(times['certificates'] + times['verification'] for times in t.values()) / 1000
    return {
        'first': first, 'last': last,
        'coefficient_share': t[first]['coefficients'] / total[first], 'lp_share': t[last]['lp'] / total[last],
        'coefficients': number('2–4', f'{min(coefficients):.0f}–{max(coefficients):.0f}', 'coefficient time [s] over N',
                               RUN_TIMES, True),
        'total_first': number('2.5', f'{total[first] / 1000:.1f}', f'run time [s] of one setting at N = {first}',
                              RUN_TIMES, True),
        'total_last': number('37', f'{total[last] / 1000:.0f}', f'run time [s] of one setting at N = {last}',
                             RUN_TIMES, True),
        'lp_first': number('1', f"{t[first]['lp']:.0f}", f'LP time [ms] at N = {first}', RUN_TIMES, True),
        'lp_last': number('31', f"{t[last]['lp'] / 1000:.0f}", f'LP time [s] at N = {last}', RUN_TIMES, True),
        'verification': number('2.7', f'{verification:.1f}',
                               f'largest exact-verification time [s] over N (stored {verification:.3f})',
                               RUN_TIMES, True),
    }


def _limits(res):
    envs, _ = res.screen()
    cell = {(data.setting(e), e['profile']): e for e in envs}
    higher = [(key, p) for key in (MID, HIGH) for p in ('exp', 'gamma')]
    bounds = {k: data.env_bound(cell[k]) for k in higher}
    ec = {k: data.worst(next(p for p in cell[k]['policies'] if p['policy'] == 'ec')) for k in higher}
    ends = ((MID, 'gamma'), (HIGH, 'exp'))
    pairs = data.pairs(envs)
    violations = sum(data.worst(p) < data.env_bound(e) for e, p in pairs)
    run = _run_times(res)
    return [
        statement('the bounds fall to [(4, 0.5, 4) gamma]–[(2, 0.7, 3) exponential]',
                  (min(bounds, key=bounds.get), max(bounds, key=bounds.get)) == ends, 'lowest and highest of the four',
                  'certified bounds of the two higher-load settings', data.SCREEN_FILES),
        statement('EC misses [(4, 0.5, 4) gamma]–[(2, 0.7, 3) exponential]',
                  (min(ec, key=ec.get), max(ec, key=ec.get)) == ends, 'lowest and highest of the four',
                  'EC estimates of the two higher-load settings', data.SCREEN_FILES),
        statement('no miss probability falls below its bound', violations == 0, f'{violations} of {len(pairs)} pairs',
                  'EC and HOL-BMW estimates against the bounds', data.SCREEN_FILES),
        run['coefficients'],
        run['total_first'],
        statement('mostly for the coefficients', run['coefficient_share'] > .5,
                  f"{run['coefficient_share']:.0%} coefficients", f"run time at N = {run['first']}", RUN_TIMES, True),
        run['total_last'],
        statement('mostly for the LP', run['lp_share'] > .5, f"{run['lp_share']:.0%} LP",
                  f"run time at N = {run['last']}", RUN_TIMES, True),
        run['verification'],
    ]


def _cost(res):
    grid = res.grid_rows()
    summary = res.read_json(data.GRID_SUMMARY)
    run = _run_times(res)
    core_ms = sum(float(r['ms_coefficients']) + float(r['ms_windows']) + float(r['ms_lp']) + float(r['ms_certificates'])
                  + float(r['ms_verification']) for r in grid)
    step = {s: res.numerics(f'step_exp_{s.replace("/", "_")}') for s in ('1/100', '1/2000', '1/4000')}
    b = {s: 100 * data.point_bound(p) for s, p in step.items()}
    model = step['1/100']['model']
    width = max(float(r[f'grid_max_{k}_width']) for r in grid for k in 'habc')
    change = max(float(r['grid_last_sweep_change']) for r in grid)
    precision = {(p, bits): data.point_bound(res.numerics(f'precision_{p}_{bits}'))
                 for p in ('exp', 'gamma', 'tnorm', 'lognormal', 'pareto') for bits in (64, 96, 128, 192)}
    bits = sorted({bits for _, bits in precision})
    rel = max(abs(float((v - precision[(p, 192)]) / precision[(p, 192)])) for (p, _), v in precision.items())
    subsets = [(res.numerics(f'subsets_all_N{n}_rho{rho}'), res.numerics(f'subsets_pairs_N{n}_rho{rho}'))
               for n in (5, 6, 8) for rho in ('1_2', '17_20')]
    same = sum(data.point_bound(a) == data.point_bound(p) for a, p in subsets)
    cost = max(sum(a['evaluation']['timing_ms'].values()) / sum(p['evaluation']['timing_ms'].values())
               for a, p in subsets)
    fine = res.brackets(data.BENCHMARK_FINE)
    shortfall = float(max(r['floor_q'] - r['lp_q'] for r in fine if r['lp_q'] < r['floor_q']))
    runs = (data.BENCHMARK, data.BENCHMARK_FINE, data.LIGHT_LOAD, data.FINITE_LOAD, data.MOMENTS)
    tails = [r['bracket']['tail'] for run_dir in runs for r in res.bracket_rows(run_dir)]
    tails += [r['bracket']['tail'] for rows in res.refinement().values() for r in rows]
    closed = [tail for tail in tails if tail['closed_form'] is not None]
    intersect = sum(max(Fraction(tail['closed_form']['lo']), Fraction(tail['grid']['lo']))
                    <= min(Fraction(tail['closed_form']['hi']), Fraction(tail['grid']['hi'])) for tail in closed)
    checks = res.read_json(data.VALIDATION)['checks']
    met = [r['fcfs_rates_point_in_relaxation'] for r in res.sweep_rows() if r['reference_meets_target']]
    return [
        statement('none was inconclusive', summary['inconclusive'] == 0,
                  f"{summary['inconclusive']} of {summary['points']}", 'inconclusive verdicts of the grid',
                  data.GRID_SUMMARY),
        run['coefficients'],
        run['lp_first'],
        run['lp_last'],
        statement('where it dominates', run['lp_share'] > .5, f"{run['lp_share']:.0%} LP",
                  f"run time at N = {run['last']}", RUN_TIMES, True),
        run['total_last'],
        run['verification'],
        number('22', f'{core_ms / 3.6e6:.0f}', 'core-hours of the whole grid', data.GRID, True),
        number('(4, 0.5, 4)', f"({model['n']}, {g(data.q(model['rho']))}, {model['deadlines'][0]})",
               'setting of the grid-step check (N, ρ, d)', 'numerics/step_exp_*.json'),
        number('0.049', f"{float(b['1/100']):.2g}", 'exponential bound [%] at g = 1/100',
               'numerics/step_exp_1_100.json'),
        number('0.124', f"{float(b['1/2000']):.3g}", 'exponential bound [%] at g = 1/2000',
               'numerics/step_exp_1_2000.json'),
        number('3', f"{100 * float(b['1/4000'] / b['1/2000'] - 1):.0f}", 'further rise [% of the value] at g = 1/4000',
               'numerics/step_exp_1_4000.json'),
        number('5670', str(len(grid)) if len(grid) == len(res.certificate_names()) else 'rows and files differ',
               'certificates of the grid (rows and stored files)', f'{data.GRID}, {data.CERTIFICATES}/'),
        number('4.0e-3', e_notation(width), 'largest enclosure width of P, A, B, C over the grid', data.GRID),
        statement('a third refinement sweep changes no reported result', change == 0,
                  f'largest last-sweep change {change:g}', 'last refinement sweep of every certificate', data.GRID),
        number('64 to 192', f'{bits[0]} to {bits[-1]}', 'arithmetic precisions compared [bits]',
               'numerics/precision_*.json'),
        number('1.3e-11', e_notation(rel), 'largest relative change of the bound from 64 to 192 bits',
               'numerics/precision_*.json'),
        statement('enumerating all class subsets returns identical bounds', same == len(subsets),
                  f'{same} of {len(subsets)} settings', 'all subsets against the restricted subsets',
                  'numerics/subsets_*.json'),
        number('44', f'{cost:.0f}', 'largest cost factor of enumerating all subsets', 'numerics/subsets_*.json', True),
        number('0.0004', f'{shortfall:.4f}',
               f'largest shortfall [pp] of L_LP below the lower enclosure of m_0 at g = 1/4000 (exact {shortfall:.2g}; '
               'the printed value is the difference of the four-decimal displays)',
               f'{data.BENCHMARK_FINE}/bracket.json'),
        statement('the closed-form and grid enclosures of Pr(τ+S>d) intersect', intersect == len(closed),
                  f"{intersect} of {len(closed)} rows with a closed form; {sum(c['passed'] for c in checks)} of "
                  f'{len(checks)} closed-form checks of the checker pass',
                  'tail enclosures of every bracket row (deterministic, exponential, gamma)',
                  f'bracket/*/bracket.json, {data.VALIDATION}'),
        statement('the exact vector of OJF satisfies every row for every target it meets', all(met),
                  f'{sum(met)} of {len(met)} targets', 'OJF rates against the relaxation of the target sweep',
                  data.SWEEP),
    ]


def _workload(res):
    bench = res.bracket_rows(data.BENCHMARK)
    contains = 0
    for r in bench:
        w, miss = r['bracket']['u_workload'], r['bracket']['benchmark']['miss']
        contains += (data.exact(w['u_k_minus']) <= Fraction(miss['lo'])
                     and Fraction(miss['hi']) <= data.exact(w['u_k_plus']))
    refine = {(r['profile'], step): data.bracket_numbers(r) for step, rows in res.refinement().items() for r in rows}
    steps = sorted({step for _, step in refine})
    profiles = sorted({p for p, _ in refine})
    halvings = [(p, a, c) for p in profiles for a, c in zip(steps, steps[1:])]
    nested = all(refine[(p, a)]['u_minus_q'] <= refine[(p, c)]['u_minus_q']
                 and refine[(p, c)]['u_workload_q'] <= refine[(p, a)]['u_workload_q'] for p, a, c in halvings)
    ratios = [refine[(p, a)]['num_width'] / refine[(p, c)]['num_width'] for p, a, c in halvings]
    args = res.manifest(f'{data.REFINEMENT}/step_{steps[0]}')
    coarse, fine = refine[('det', steps[0])], refine[('det', steps[-1])]
    refine_source = f'{data.REFINEMENT}/step_*/bracket.json'
    finite = res.brackets(data.FINITE_LOAD)
    settled = sum(r['verdict'] != 'undetermined' for r in finite)
    two_thirds = 'two thirds' if Fraction(settled, len(finite)) == Fraction(2, 3) else f'{settled} of {len(finite)}'
    light = [r for r in finite if r['profile'] in ('det', 'gamma') and r['d'] == '3' and r['rho'] <= .1 + 1e-12]
    heavy = [r for r in finite if r['profile'] == 'pareto_heavy' and r['d'] == '3']
    as_stated = sum((r['verdict'] == 'certified_infeasible') == (r['n'] != 2 or r['rho'] >= .1 - 1e-12) for r in heavy)
    undetermined = Counter(r['rho'] for r in finite if r['verdict'] == 'undetermined')
    loads = sorted({r['rho'] for r in finite})
    by_load = [undetermined[x] for x in loads]
    table_steps = sorted({r['bracket']['u_workload']['step'] for r in res.bracket_rows(data.FINITE_LOAD)
                          if r['n'] == 2 and r['d'] == '2'})
    return [
        statement('every interval contains the exact m_F', contains == len(bench),
                  f'{contains} of {len(bench)} settings', 'workload enclosure against the closed form',
                  f'{data.BENCHMARK}/bracket.json'),
        statement('nested intervals whose width roughly halves with each halving of g', nested,
                  f'nested; width ratios {min(ratios):.2f}–{max(ratios):.2f}', 'grid refinement of the workload bound',
                  refine_source),
        number('2, 0.1, 2', f"{joined(args['ns'])}, {g(data.q(args['rhos'][0]))}, {joined(args['ds'])}",
               'N, ρ, d of the refinement', f'{data.REFINEMENT}/step_{steps[0]}/manifest.json'),
        statement('contains the 1 % target', coarse['u_minus_q'] <= 1 <= coarse['u_workload_q'],
                  f"[{float(coarse['u_minus_q']):.4f}, {float(coarse['u_workload_q']):.4f}] %",
                  f'deterministic interval at g = 1/{steps[0]}', refine_source),
        statement('which certifies it', fine['u_workload_q'] < 1 and fine['verdict'] == 'certified_feasible',
                  f"U^+ = {float(fine['u_workload_q']):.4f} %, {fine['verdict']}",
                  f'deterministic interval at g = 1/{steps[-1]}', refine_source),
        number('two thirds', two_thirds, 'settled settings of the finite-load set', FINITE),
        statement('deterministic and gamma stay certified feasible up to ρ = 0.1 (d = 3)',
                  all(r['verdict'] == 'certified_feasible' for r in light), f'{len(light)} of {len(light)} settings',
                  'verdicts at loads up to 0.1', FINITE),
        statement('Pareto-3/2 is excluded except at N = 2 below ρ = 0.1 (d = 3)', as_stated == len(heavy),
                  f'{as_stated} of {len(heavy)} settings', 'verdicts of Pareto-3/2 at d = 3', FINITE),
        number('one', WORDS.get(undetermined[.01], str(undetermined[.01])), 'undetermined settings at ρ = 0.01',
               FINITE),
        number('seven', WORDS.get(undetermined[.3], str(undetermined[.3])), 'undetermined settings at ρ = 0.3',
               FINITE),
        statement('rising', by_load == sorted(by_load), joined(by_load),
                  f'undetermined settings at ρ = {joined(loads, g)}', FINITE),
        number('1/2000', joined(table_steps), 'grid step g of the enclosure table (N = 2, d = 2)', FINITE),
    ]


def _holbmw(res):
    alpha = {p['spec']['params']['alpha'] for part in res.screen_parts() for p in part['policies']
             if p['spec']['name'] == 'hol_bmw'}
    return [number('0.1', joined(sorted(alpha), g), 'exponent α of the adapted HOL-BMW policy', data.SCREEN_FILES)]


def literals(res):
    """{part of the paper: [Literal]} for the numbers and statements written into the manuscript."""
    return {'setup': _setup(res), 'exclusion': _exclusion(res), 'gap': _gap(res), 'scheduling': _scheduling(res),
            'limits': _limits(res), 'cost': _cost(res), 'workload': _workload(res), 'holbmw': _holbmw(res)}


INTRO = """\
Every number of the evaluation section and the appendices that is derived from a stored run,
grouped by the part of the paper that prints it. *Macros* are defined in `reference_numbers.tex`
and printed by the manuscript as they are. *Text* lists the numbers written directly into the
manuscript, each recomputed from the stored runs with the rounding the paper uses and compared with
the printed text; statements that summarize the results are checked the same way. Percentages are
in percent and pp means percentage points. Rows marked (timing) are run times, which depend on the
machine; their values are the ones measured in the stored runs. Sources are relative to `results/`.
Method constants that the runs do not record (the window lengths and offsets of the evaluation
grid) are not listed."""


def _source(source):
    """Paths as code; a source that is not a stored file (a closed form) as text."""
    return f'`{source}`' if '/' in source else source


def write(path, macro_sections, literal_sections):
    """Write numbers.md: a summary with the mismatches, then per part of the paper its macros and text numbers."""
    macros = unique(macro_sections)
    texts = [(key, x) for key, xs in literal_sections.items() for x in xs]
    wrong = [(key, x) for key, x in texts if not x.match]
    titles = dict(SECTIONS)
    lines = ['# Numbers printed in the paper', '', 'Generated by `plotting/paper.py` from `results/`; do not edit.', '',
             INTRO, '',
             f'{len(macros)} macros; {len(texts)} numbers and statements in the text, '
             f'{len(texts) - len(wrong)} of which match the results.', '']
    if wrong:
        lines += ['Not matching:', '']
        lines += [f'- {titles[key]}: printed {x.printed}, from the results {x.value} ({x.what})' for key, x in wrong]
        lines.append('')
    for key, title in SECTIONS:
        lines += [f'## {title}', '']
        if macro_sections.get(key):
            lines += ['Macros:', '', '| Macro | Value | What | Source |', '|---|---|---|---|']
            lines += [f'| `\\{m.name}` | `{m.value}` | {m.what} | {_source(m.source)} |' for m in macro_sections[key]]
            lines.append('')
        if literal_sections.get(key):
            lines += ['Text:', '', '| Printed | From the results | Match | What | Source |', '|---|---|---|---|---|']
            for x in literal_sections[key]:
                timing = ' (timing)' if x.timing else ''
                lines.append(f"| {x.printed} | {x.value} | {'yes' if x.match else '**no**'} | {x.what}{timing} | "
                             f'{_source(x.source)} |')
            lines.append('')
    path.write_text('\n'.join(lines))
