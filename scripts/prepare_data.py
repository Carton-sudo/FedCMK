"""Download the three public benchmark datasets using the retained torchvision loaders' layout."""
from pathlib import Path
import argparse

ROOT = Path(__file__).resolve().parents[1]


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--dataset', choices=['mnist', 'cifar10', 'cifar100', 'all'], required=True)
    p.add_argument('--data-root', type=Path, default=ROOT / 'data')
    args = p.parse_args()
    from torchvision.datasets import MNIST, CIFAR10, CIFAR100
    datasets = {'mnist':MNIST, 'cifar10':CIFAR10, 'cifar100':CIFAR100}
    for name, cls in datasets.items():
        if args.dataset not in {'all', name}: continue
        location = args.data_root.expanduser().resolve() / name
        for train in [True, False]: cls(str(location), train=train, download=True)
        print(f'{name}: {location}')
