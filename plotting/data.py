"""Loaders for the stored runs under results/ and helpers to select from them.

Nothing is recomputed with the model: every value comes from a CSV or JSON file written by the
`tdf` commands. `bound`, `env_bound`, `worst` and `bracket_numbers` return probabilities in percent.
"""
import csv
import json
from fractions import Fraction
from pathlib import Path

# The evaluation grid: 6 profiles x N in {2, 3, 4, 6, 8} x 9 loads x 7 deadlines = 1890 settings,
# each tested at three targets.
PROFILES = ['det', 'exp', 'gamma', 'tnorm', 'lognormal', 'pareto']
RHOS = ['1/20', '1/10', '1/5', '3/10', '1/2', '7/10', '17/20', '9/10', '19/20']
DS = ['1', '2', '3', '4', '6', '8', '12']
EPSS = ['1/1000', '1/100', '1/20']
GRID_TESTS = 5670

# The policies of the policy screen that the paper reports; other policies in a stored run are ignored.
PAPER_POLICIES = ('ec', 'hol_bmw')

# Stored files and run directories, relative to the results directory.
GRID = 'grid/map.csv'
GRID_SUMMARY = 'grid/summary.json'
CERTIFICATES = 'grid/certificates'
CERTIFICATE = 'gamma_N_4_rho_1_5_d_2_eps_1_100'  # the certificate of the combination-only exclusion in the text
DISTRIBUTIONS = 'distributions/profiles.json'
VALIDATION = 'validation/validate.json'
NUMERICS = 'numerics'
SCREEN_PARTS = ('a', 'b', 'c')
SCREEN = 'screen/{part}/screen.json'
SCREEN_FILES = SCREEN.format(part='{a,b,c}')
BENCHMARK = 'bracket/benchmark'
BENCHMARK_FINE = 'bracket/benchmark_fine'
LIGHT_LOAD = 'bracket/light_load'
FINITE_LOAD = 'bracket/finite_load'
MOMENTS = 'bracket/moments'
REFINEMENT = 'bracket/refinement'
OJF = 'ojf/validate.json'
SWEEP = 'target_sweep/sweep.json'
REGION = 'rate_region/region.json'
DIRECTIONAL = 'directional/directional.json'


def q(text):
    """A rational string such as '1/20' as a float."""
    return float(Fraction(text))


def exact(v):
    """The exact value of a stored number (a dict with 'exact' or 'f64', or a rational string); floats stay."""
    if isinstance(v, dict):
        v = v['exact'] if 'exact' in v else v['f64']
    return Fraction(v) if isinstance(v, str) else v


class Results:
    """Read access to the stored runs under one results directory."""

    def __init__(self, root):
        self.root = Path(root)

    def read_json(self, relative):
        return json.loads((self.root / relative).read_text())

    def read_csv(self, relative):
        with (self.root / relative).open() as handle:
            rows = list(csv.DictReader(handle))
        assert rows and all(None not in row for row in rows), relative
        return rows

    # evaluation grid

    def grid_rows(self):
        """Every row of the grid: one per setting and target, fields as strings."""
        rows = self.read_csv(GRID)
        assert len(rows) == GRID_TESTS
        return rows

    def core(self):
        """The grid rows at the 1 % target."""
        return [r for r in self.grid_rows() if r['eps'] == '1/100']

    def certificate(self, name):
        """A stored certificate of the grid, by file name without '.json'."""
        return self.read_json(f'{CERTIFICATES}/{name}.json')

    def certificate_names(self):
        return sorted(p.stem for p in (self.root / CERTIFICATES).glob('*.json'))

    # policy screen

    def screen_parts(self):
        """The stored parts of the policy screen, each a run over some of its settings."""
        return [self.read_json(SCREEN.format(part=part)) for part in SCREEN_PARTS]

    def screen(self):
        """The policy screen merged over its parts, with the paper's policies only:
        (environments, number of replications)."""
        parts = self.screen_parts()
        envs = [e for part in parts for e in part['environments']]
        for e in envs:
            e['policies'] = [p for p in e['policies'] if p['policy'] in PAPER_POLICIES]
        return envs, len(parts[0]['seeds'])

    # single-setting checks

    def numerics(self, name):
        return self.read_json(f'{NUMERICS}/{name}.json')

    def runtimes(self):
        """{N: timing in ms by part} of the single-setting runs without a coefficient cache."""
        runs = {int(p.stem.split('_N')[1]): p.stem for p in (self.root / NUMERICS).glob('runtime_N*.json')}
        return {n: self.numerics(runs[n])['evaluation']['timing_ms'] for n in sorted(runs)}

    # certified intervals, OJF, rate ranges

    def bracket_rows(self, run):
        return self.read_json(f'{run}/bracket.json')['rows']

    def brackets(self, run):
        """The rows of a bracket run as `bracket_numbers`."""
        return [bracket_numbers(r) for r in self.bracket_rows(run)]

    def manifest(self, run):
        """The arguments of a stored run."""
        return self.read_json(f'{run}/manifest.json')['args']

    def refinement(self):
        """{S: rows} of the grid-refinement runs with step 1/S."""
        return {int(path.name.split('_')[1]): self.bracket_rows(f'{REFINEMENT}/{path.name}')
                for path in (self.root / REFINEMENT).glob('step_*')}

    def ojf_rows(self):
        return self.read_json(OJF)['rows']

    def sweep_rows(self):
        return self.read_json(SWEEP)['rows']

    def region(self):
        return self.read_json(REGION)

    def directional(self):
        return self.read_json(DIRECTIONAL)


def one(rows, **fields):
    """The single row whose `fields` have the given values (compared as strings)."""
    out = [r for r in rows if all(str(r[k]) == str(v) for k, v in fields.items())]
    assert len(out) == 1, (fields, len(out))
    return out[0]


# grid rows

def excluded(row, level='full'):
    """Whether the constraint level certifies the row's target infeasible."""
    category = row[level + '_category']
    assert category in ('certified_infeasible', 'undetermined'), (row['id'], category)
    return category == 'certified_infeasible'


def bound(row, level='full'):
    """The certified lower bound on max_i m_i of the constraint level, in percent."""
    return 100 * float(row[level + '_lb_max'])


def point_bound(point):
    """The certified lower bound on max_i m_i of a single-setting run, as an exact fraction."""
    level = next(v for v in point['evaluation']['levels'] if v['level'] == 'full')
    return Fraction(level['min_max']['certified_value_exact'])


# policy screen

def setting(env):
    """(N, rho, d) of a screen environment."""
    m = env['model']
    return int(m['n']), m['rho'], m['deadlines'][0]


def worst(policy):
    """The pooled worst-class miss probability of a policy in percent, or None without an estimate."""
    pooled = policy.get('pooled')
    if not pooled or pooled.get('worst') is None:
        return None
    return 100 * pooled['worst']


def env_bound(env):
    """The certified lower bound on max_i m_i of a screen environment, in percent."""
    return 100 * env['certified']['full_lb_max']


def pairs(envs):
    """(environment, policy) pairs with a pooled estimate."""
    return [(e, p) for e in envs for p in e['policies'] if worst(p) is not None]


# certified intervals

def bracket_numbers(row):
    """The quantities of one bracket row, probabilities in percent (except `truncation`): floats for
    drawing and comparing, and exact rationals (`*_q`) for outward rounding."""
    b = row['bracket']
    bench = b['benchmark']
    w = b['u_workload']
    return {
        'rho': row['rho_f64'], 'rho_exact': row['rho'], 'd': row['d'], 'n': row['n'], 'profile': row['profile'],
        'lp': 100 * b['lp_lower']['f64'], 'safe': 100 * b['l_safe']['f64'], 'floor': 100 * b['floor_b']['lo_f64'],
        'u_simple': 100 * b['u_simple']['f64'],
        'u_exact': None if bench is None else 100 * bench['miss']['hi_f64'],
        'u_workload': 100 * w['u_k_plus']['f64'],
        'gap': 100 * b['gap_safe']['f64'],
        'verdict': b['verdict'],
        'lp_q': 100 * exact(b['lp_lower']), 'safe_q': 100 * exact(b['l_safe']),
        'floor_q': 100 * Fraction(b['floor_b']['lo']),
        'u_simple_q': 100 * exact(b['u_simple']),
        'u_workload_q': 100 * exact(w['u_k_plus']),
        'u_minus_q': 100 * exact(w['u_k_minus']),
        'num_width': 100 * w['numeric_width'], 'truncation': w['truncation'],
    }


def gap_series(light_load, finite_load, d):
    """Bracket numbers at deadline d keyed by (profile, N, load); the finite-load set overrides the light-load set."""
    return {(r['profile'], r['n'], round(r['rho'], 9)): r for r in light_load + finite_load if r['d'] == d}


def counterexample(sweep_rows):
    """The sweep row of the rate-membership counterexample: the smallest target whose relaxation admits the
    rates of OJF while OJF misses the target."""
    for r in sorted(sweep_rows, key=lambda r: r['eps_f64']):
        if r['fcfs_rates_point_in_relaxation'] and not r['reference_meets_target']:
            return r
    raise RuntimeError('no rate-membership counterexample in the sweep')
