#!/usr/bin/env python3
"""Draw the paper's figures and write its number macros and numbers report from the stored results.

    python3 plotting/paper.py [--results results] [--out figures] [--install MANUSCRIPT_DIR]

Reads only the stored runs under --results and writes into --out:

    fig3_where, fig4_bracket, fig5_scheduling   the evaluation figures (.pdf, with a .png preview)
    reference_numbers.tex                       the number macros that the manuscript includes
    numbers.md                                  every result-derived number of the paper, with its source

With --install, the three PDFs are also copied to MANUSCRIPT_DIR/assets/figures/ and the macros to
MANUSCRIPT_DIR/tex/generated/reference_numbers.tex. Nothing else is written outside --out.
"""
import argparse
import shutil
import sys
from pathlib import Path

import data
import figures
import macros
import report
import style

ROOT = Path(__file__).resolve().parent.parent  # the repository


def install(out, manuscript):
    """Copy the figure PDFs and the macro file from `out` into the manuscript directory."""
    assets = manuscript / 'assets' / 'figures'
    generated = manuscript / 'tex' / 'generated'
    for folder in (assets, generated):
        folder.mkdir(parents=True, exist_ok=True)
    for draw in figures.FIGURES:
        shutil.copyfile(out / f'{draw.__name__}.pdf', assets / f'{draw.__name__}.pdf')
    shutil.copyfile(out / 'reference_numbers.tex', generated / 'reference_numbers.tex')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument('--results', type=Path, default=ROOT / 'results',
                        help='directory of the stored runs (default: results/ of the repository)')
    parser.add_argument('--out', type=Path, default=ROOT / 'figures',
                        help='output directory (default: figures/ of the repository)')
    parser.add_argument('--install', type=Path, metavar='MANUSCRIPT_DIR',
                        help='also copy the figures and the macros into this manuscript directory')
    args = parser.parse_args(argv)
    if not (args.results / data.GRID).is_file():
        parser.error(f'--results: no stored runs in {args.results}')
    if args.install is not None and not args.install.is_dir():
        parser.error(f'--install: {args.install} is not a directory')
    if not style.EXACT_FONT:
        print(f'warning: {style.FONT} not found; the figures use a substitute font and differ from the published ones',
              file=sys.stderr)
    res = data.Results(args.results)
    args.out.mkdir(parents=True, exist_ok=True)
    for draw in figures.FIGURES:
        draw(res, args.out)
    macro_sections = macros.compute(res)
    macros.write_tex(args.out / 'reference_numbers.tex', macro_sections)
    literal_sections = report.literals(res)
    report.write(args.out / 'numbers.md', macro_sections, literal_sections)
    texts = [x for xs in literal_sections.values() for x in xs]
    print(f'{args.out}: {", ".join(draw.__name__ for draw in figures.FIGURES)} (.pdf, .png), '
          f'reference_numbers.tex ({len(macros.unique(macro_sections))} macros), '
          f'numbers.md ({sum(x.match for x in texts)} of {len(texts)} text numbers match)')
    if args.install is not None:
        install(args.out, args.install)
        print(f'{args.install}: installed the figures and reference_numbers.tex')


if __name__ == '__main__':
    main()
