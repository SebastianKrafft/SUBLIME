"""
experiments/hparam_sweep.py

Runs a hyperparameter sweep over (temperature, c) and optionally lr.
All other hyperparameters are taken from the dataset's paper baseline.

Usage:
    sublime-python -m experiments.hparam_sweep \
        -dataset cora \
        -experiment joint \
        -ntrials 3

-experiment choices:
    temperature   Experiment 1: temperature only, c fixed at baseline
    c             Experiment 2: c only, temperature fixed at 0.2
    joint         Experiment 3: full (temperature, c) grid
    lr            Experiment 4: lr only, pass -temperature and -c explicitly
    final         Experiment 5: paper baseline + best config, ntrials=5
"""

import argparse
import copy
import itertools
import os
import sys
import time

import main as sublime_main

# ------------------------------------------------------------------
# Per-dataset base configurations (from scripts/*.sh)
# ------------------------------------------------------------------
BASE_CONFIGS = {
    "cora": dict(
        dataset="cora", ntrials=3, sparse=0,
        epochs_cls=200, lr_cls=0.001, w_decay_cls=0.0005,
        hidden_dim_cls=32, dropout_cls=0.5, dropedge_cls=0.75,
        nlayers_cls=2, patience_cls=10,
        epochs=4000, lr=0.01, w_decay=0.0,
        hidden_dim=512, rep_dim=256, proj_dim=256,
        dropout=0.5, dropedge_rate=0.5, nlayers=2,
        type_learner="fgp", k=30, sim_function="cosine",
        activation_learner="relu",
        gsl_mode="structure_refinement", eval_freq=50,
        tau=0.9999,
        maskfeat_rate_learner=0.7, maskfeat_rate_anchor=0.6,
        contrast_batch_size=0, gpu=0,
        downstream_task="classification",
        gamma=0.9,
    ),
    "citeseer": dict(
        dataset="citeseer", ntrials=3, sparse=0,
        epochs_cls=200, lr_cls=0.001, w_decay_cls=0.05,
        hidden_dim_cls=32, dropout_cls=0.5, dropedge_cls=0.5,
        nlayers_cls=2, patience_cls=10,
        epochs=1000, lr=0.001, w_decay=0.0,
        hidden_dim=512, rep_dim=256, proj_dim=256,
        dropout=0.5, dropedge_rate=0.25, nlayers=2,
        type_learner="att", k=20, sim_function="cosine",
        activation_learner="tanh",
        gsl_mode="structure_refinement", eval_freq=20,
        tau=0.9999,
        maskfeat_rate_learner=0.6, maskfeat_rate_anchor=0.8,
        contrast_batch_size=0, gpu=0,
        downstream_task="classification",
        gamma=0.9,
    ),
}

# ------------------------------------------------------------------
# Grids
# ------------------------------------------------------------------
GRIDS = {
    "temperature": {
        "temperature": [0.05, 0.1, 0.2, 0.5, 1.0],
        "c":           [0],
    },
    "c": {
        "temperature": [0.2],
        "c":           [0, 1, 5, 10, 25, 50, 100],
    },
    "joint": {
        "temperature": [0.05, 0.1, 0.2, 0.5, 1.0],
        "c":           [0, 10, 50, 100],
    },
}


def make_args(base_cfg, overrides):
    """Build a Namespace from base config dict + override dict."""
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def run_single(args, log_path):
    """Run one training configuration, redirecting stdout to log_path."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    # Skip if already done
    if os.path.exists(log_path) and os.path.getsize(log_path) > 100:
        print(f"  [skip] {log_path}", flush=True)
        return

    print(f"  [run]  {log_path}", flush=True)
    t0 = time.time()

    # Redirect stdout to log file
    orig_stdout = sys.stdout
    with open(log_path, "w") as f:
        sys.stdout = f
        try:
            sublime_main.args = args
            sublime_main.Experiment().train(args)
        finally:
            sys.stdout = orig_stdout

    elapsed = time.time() - t0
    print(f"  [done] {elapsed/60:.1f} min", flush=True)


def sweep(dataset, experiment, overrides=None):
    base = BASE_CONFIGS[dataset]
    log_dir = f"logs/sweep/{dataset}/{experiment}"

    if experiment in GRIDS:
        grid = GRIDS[experiment]
        keys = list(grid.keys())
        values = list(grid.values())
        combos = list(itertools.product(*values))
        total = len(combos)
        for i, combo in enumerate(combos, 1):
            params = dict(zip(keys, combo))
            tag = "_".join(f"{k}{v}" for k, v in params.items())
            log_path = f"{log_dir}/{tag}.log"
            print(f"\n[{i}/{total}] {dataset} {experiment}: {params}", flush=True)
            args = make_args(base, params)
            run_single(args, log_path)

    elif experiment == "lr":
        assert overrides is not None, "Pass -temperature and -c for lr experiment"
        lr_grid = [0.0001, 0.001, 0.01, 0.1]
        total = len(lr_grid)
        for i, lr in enumerate(lr_grid, 1):
            params = {"lr": lr, **overrides}
            tag = f"lr{lr}_t{overrides['temperature']}_c{overrides['c']}"
            log_path = f"{log_dir}/{tag}.log"
            print(f"\n[{i}/{total}] {dataset} lr sweep: lr={lr}", flush=True)
            args = make_args(base, params)
            run_single(args, log_path)

    elif experiment == "final":
        assert overrides is not None, "Pass best temperature, c, lr for final experiment"
        configs = {
            "paper_baseline": {"temperature": 0.2, "c": 0, "lr": base["lr"]},
            "best_config":    overrides,
        }
        for name, params in configs.items():
            # final uses ntrials=5
            params["ntrials"] = 5
            log_path = f"{log_dir}/{name}.log"
            print(f"\nfinal {dataset} {name}: {params}", flush=True)
            args = make_args(base, params)
            run_single(args, log_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-dataset",     type=str, required=True,
                   choices=["cora", "citeseer"])
    p.add_argument("-experiment",  type=str, required=True,
                   choices=["temperature", "c", "joint", "lr", "final"])
    p.add_argument("-temperature", type=float, default=None,
                   help="Used for lr and final experiments")
    p.add_argument("-c",           type=int,   default=None,
                   help="Used for lr and final experiments")
    p.add_argument("-lr",          type=float, default=None,
                   help="Used for final experiment")
    cli = p.parse_args()

    overrides = {}
    if cli.temperature is not None: overrides["temperature"] = cli.temperature
    if cli.c           is not None: overrides["c"]           = cli.c
    if cli.lr          is not None: overrides["lr"]          = cli.lr

    sweep(
        dataset=cli.dataset,
        experiment=cli.experiment,
        overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()