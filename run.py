"""Training entry for the included federated learning configurations. No training on import."""
from pathlib import Path
import argparse
import contextlib
import csv
import json
import os
import sys

ROOT = Path(__file__).resolve().parent
AVAILABLE = {'fedcmk', 'fedavg', 'fedprox', 'moon', 'fedntd', 'fedlmd'}
ALIASES = {}


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, help='Exact exported JSON; relative paths are repository-relative.')
    p.add_argument('--suite', choices=['benchmark', 'resources'], default='benchmark')
    p.add_argument('--method', default='FedCMK')
    p.add_argument('--scene', help='e.g. c10-s2, c10-lda01, c100-s10, m-s2')
    p.add_argument('--seed', type=int, default=2022)
    p.add_argument('--participants', type=int, help='Select a recorded resource setting; does not change its budget.')
    p.add_argument('--local-epochs', type=int, help='Select a recorded resource setting; does not change its budget.')
    p.add_argument('--data-root', type=Path, help='Parent of mnist/, cifar10/, cifar100/.')
    p.add_argument('--output', type=Path, help='New output directory; never overwritten.')
    p.add_argument('--device', choices=['cpu', 'mps', 'cuda'], help='Explicit override; default is the recorded device.')
    p.add_argument('--resume', type=Path, help='Trusted full-state checkpoint in --output; not an untrusted download.')
    p.add_argument('--save-model-artifacts', choices=['original', 'true', 'false'], default='original', help='Original saved policy by default; false saves metrics only (cannot resume).')
    p.add_argument('--verbose', action='store_true', help='Print original per-client progress; no duplicate console log is written.')
    p.add_argument('--list', action='store_true', help='List matching recorded configurations without importing torch.')
    p.add_argument('--dry-run', action='store_true', help='Resolve and validate paths/configuration without models or datasets.')
    return p


def repo_path(path):
    p = Path(path).expanduser()
    return p.resolve() if p.is_absolute() else (ROOT / p).resolve()


def catalog():
    with (ROOT / 'configs/index.csv').open(newline='') as f:
        return list(csv.DictReader(f))


def select_config(args):
    if args.config:
        return repo_path(args.config)
    method = ALIASES.get(args.method.lower(), args.method.lower())
    rows = [r for r in catalog() if r['category'] == args.suite and r['method'] == method and int(r['seed']) == args.seed]
    if args.scene:
        rows = [r for r in rows if r['scene'] == args.scene]
    for flag in ('participants', 'local_epochs'):
        val = getattr(args, flag)
        if val is not None:
            rows = [r for r in rows if int(r[flag]) == val]
    if args.list:
        for r in rows:
            print(f"{r['method']:24} {r['scene']:12} seed={r['seed']} K={r['participants']} E={r['local_epochs']} {r['runtime_status']}\n  {r['config']}")
        return None
    if len(rows) != 1:
        raise ValueError(f'Matched {len(rows)} configurations. Specify --scene and, for resources, --participants/--local-epochs; use --list or --config for an exact choice.')
    return repo_path(rows[0]['config'])


def resolved_config(args, path):
    cfg = json.loads(path.read_text())
    cfg['data_setups']['root'] = str(repo_path(args.data_root or cfg['data_setups']['root']))
    part = cfg['data_setups']['partition']
    if part.get('index_path'):
        part['index_path'] = str(repo_path(part['index_path']))
        if not Path(part['index_path']).is_file():
            raise FileNotFoundError(part['index_path'])
    train = cfg['train_setups']
    log = train.setdefault('logging', {})
    log['client_schedule_path'] = str(repo_path(log['client_schedule_path']))
    if not Path(log['client_schedule_path']).is_file():
        raise FileNotFoundError(log['client_schedule_path'])
    log['output_dir'] = str(repo_path(args.output or log['output_dir']))
    if args.device:
        train['scenario']['device'] = args.device
    if args.save_model_artifacts != 'original':
        log['save_model_artifacts'] = args.save_model_artifacts == 'true'
    if args.resume:
        checkpoint = repo_path(args.resume)
        if not checkpoint.is_file() or checkpoint.parent != Path(log['output_dir']):
            raise ValueError('--resume must be an existing trusted checkpoint inside --output.')
        log['resume_from'] = str(checkpoint)
    cfg['wandb_setups']['mode'] = 'disabled'
    return cfg


def execute(args):
    path = select_config(args)
    if path is None:
        return
    cfg = resolved_config(args, path)
    method = cfg['train_setups']['algo']['name']
    if args.dry_run:
        print(json.dumps({'runtime_available': method in AVAILABLE, 'config': cfg}, indent=2))
        return
    if method not in AVAILABLE:
        raise ValueError(f'{method}: algorithm is not registered. Choose a method listed in README.md.')
    os.environ['WANDB_MODE'] = 'disabled'
    os.environ.setdefault('WANDB_SILENT', 'true')
    sys.dont_write_bytecode = True
    sys.path.insert(0, str(ROOT / 'src'))
    import torch
    from utils import ConfLoader
    import runtime
    device = cfg['train_setups']['scenario']['device']
    if device == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable; explicitly choose --device cpu or mps.')
    if device == 'mps' and not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable; explicitly choose --device cpu or cuda.')
    output = Path(cfg['train_setups']['logging']['output_dir'])
    if output.exists() and not args.resume:
        raise FileExistsError(f'Refusing existing output directory: {output}')
    output.mkdir(parents=True, exist_ok=bool(args.resume))
    saved = output / 'config.json'
    if not args.resume:
        with saved.open('x') as f:
            json.dump(cfg, f, indent=2)
    # Retain the original attribute-dict interface without a private config loader.
    opt = json.loads(json.dumps(cfg), object_hook=ConfLoader.DictWithAttributeAccess)
    print(f"Starting {method}, {cfg['data_setups']['dataset_name']}, seed {cfg['train_setups']['seed']}, {device}\nOutput: {output}", flush=True)
    runtime.wandb.init(config=opt, **opt.wandb_setups)
    runtime.wandb.config.log_interval = 10
    try:
        if args.verbose:
            runtime.main(opt)
        else:
            with open(os.devnull, 'w') as silent, contextlib.redirect_stdout(silent):
                runtime.main(opt)
    finally:
        runtime.wandb.finish()
    print(f'Completed. Saved measurements: {output / "round_metrics.csv"}', flush=True)


if __name__ == '__main__':
    try:
        execute(parser().parse_args())
    except (ValueError, FileNotFoundError, FileExistsError, RuntimeError) as exc:
        print(f'Error: {exc}', file=sys.stderr)
        raise SystemExit(2)
