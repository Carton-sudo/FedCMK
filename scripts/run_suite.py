"""Print a training run matrix; execute sequentially only with --execute."""
from pathlib import Path
import argparse
import csv
import shlex
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.dont_write_bytecode = True
sys.path.insert(0, str(ROOT))
from run import AVAILABLE, ALIASES


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--suite', choices=['benchmark', 'resources'], required=True)
    p.add_argument('--method', default='FedCMK', help='Exact method or available to select all runnable methods.')
    p.add_argument('--scene')
    p.add_argument('--seed', type=int)
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], default='cpu')
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    p.add_argument('--output-root', type=Path, default=ROOT / 'runs')
    p.add_argument('--save-model-artifacts', choices=['original', 'true', 'false'], default='original')
    p.add_argument('--execute', action='store_true')
    args = p.parse_args()
    method = ALIASES.get(args.method.lower(), args.method.lower())
    rows = list(csv.DictReader((ROOT / 'configs/index.csv').open()))
    rows = [r for r in rows if r['category'] == args.suite and (r['method'] in AVAILABLE if method == 'available' else r['method'] == method)]
    if args.scene: rows = [r for r in rows if r['scene'] == args.scene]
    if args.seed is not None: rows = [r for r in rows if int(r['seed']) == args.seed]
    if not rows: raise SystemExit('No recorded configuration matches.')
    unavailable = sorted({r['method'] for r in rows if r['method'] not in AVAILABLE})
    if unavailable: raise SystemExit(f'Implementation unavailable: {unavailable}; choose a method listed in README.md.')
    for r in rows:
        relative = Path(r['config']).relative_to('configs').with_suffix('')
        output = args.output_root.expanduser().resolve() / relative
        cmd = [sys.executable, str(ROOT / 'run.py'), '--config', r['config'], '--device', args.device,
               '--data-root', str(args.data_root.expanduser().resolve()), '--output', str(output),
               '--save-model-artifacts', args.save_model_artifacts]
        print(shlex.join(cmd), flush=True)
        if args.execute: subprocess.run(cmd, check=True)


if __name__ == '__main__': main()
