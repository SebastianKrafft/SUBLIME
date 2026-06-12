"""
experiments/hparam_tuning_si.py

Hyperparameter tuning for Table 1 (structure inference scenario).

Tunes temperature, tau, and c — the three hyperparameters most relevant to
structure inference. Unlike structure refinement, the anchor starts as an
identity matrix rather than the original graph, so tau (how much the anchor
updates per bootstrapping step) plays a much larger role here.

Usage:
    sublime-python -m experiments.hparam_tuning_si \\
        -dataset cora \\
        -experiment temperature_tau

-experiment choices:
    temperature       Experiment 1: temperature only, tau and c fixed at baseline
    tau               Experiment 2: tau only, temperature and c fixed at baseline
    temperature_tau   Experiment 3: joint (temperature x tau) grid, c fixed at 0
    tau_c             Experiment 4: joint (tau x c) grid, at best temperature from Exp 3
    lr                Experiment 5: lr only, pass -temperature, -tau, -c explicitly
    final             Experiment 6: paper baseline vs best config, ntrials=5

Notes on base configs
---------------------
All non-tuned hyperparameters are taken from scripts/cora_si.sh and
scripts/citeseer_si.sh. Key differences from the SR configs in hparam_tuning.py:

  - gsl_mode = structure_inference  (no original graph used)
  - tau = 1 for Cora (paper disables bootstrapping entirely)
  - tau = 0.9999 for Citeseer
  - Different masking rates, learner types, and epoch counts

Paper baseline tau values (from scripts/*.sh):
  Cora:     tau=1      (bootstrapping disabled)
  Citeseer: tau=0.9999
"""

import argparse
import copy
import itertools
import os
import sys
import time

import main as sublime_main

# ------------------------------------------------------------------
# Per-dataset base configurations (from scripts/cora_si.sh
# and scripts/citeseer_si.sh)
# ------------------------------------------------------------------
BASE_CONFIGS = {
    "cora": dict(
        dataset="cora", ntrials=3, sparse=0,
        epochs_cls=200, lr_cls=0.001, w_decay_cls=0.0005,
        hidden_dim_cls=32, dropout_cls=0.5, dropedge_cls=0.25,
        nlayers_cls=2, patience_cls=10,
        epochs=4000, lr=0.01, w_decay=0.0,
        hidden_dim=512, rep_dim=256, proj_dim=256,
        dropout=0.5, dropedge_rate=0.5, nlayers=2,
        type_learner="fgp", k=30, sim_function="cosine",
        activation_learner="relu",
        gsl_mode="structure_inference", eval_freq=20,
        tau=1,               # paper baseline: bootstrapping disabled
        c=0,
        maskfeat_rate_learner=0.5, maskfeat_rate_anchor=0.7,
        contrast_batch_size=0, gpu=0,
        downstream_task="classification",
        gamma=0.9,
    ),
    "citeseer": dict(
        dataset="citeseer", ntrials=3, sparse=0,
        epochs_cls=200, lr_cls=0.001, w_decay_cls=0.05,
        hidden_dim_cls=32, dropout_cls=0.5, dropedge_cls=0.5,
        nlayers_cls=2, patience_cls=10,
        epochs=1000, lr=0.01, w_decay=0.0,
        hidden_dim=512, rep_dim=256, proj_dim=256,
        dropout=0.5, dropedge_rate=0.25, nlayers=2,
        type_learner="att", k=20, sim_function="cosine",
        activation_learner="tanh",
        gsl_mode="structure_inference", eval_freq=50,
        tau=0.9999,          # paper baseline
        c=0,
        maskfeat_rate_learner=0.8, maskfeat_rate_anchor=0.7,
        contrast_batch_size=0, gpu=0,
        downstream_task="classification",
        gamma=0.9,
    ),
}

# Paper baseline temperature (fixed throughout, same as SR)
PAPER_TEMPERATURE = 0.2

# Paper baseline tau per dataset
PAPER_TAU = {
    "cora":     1,
    "citeseer": 0.9999,
}

# ------------------------------------------------------------------
# Grids
# ------------------------------------------------------------------
GRIDS = {
    # Experiment 1: temperature only
    # tau and c held at paper baseline per dataset
    "temperature": {
        "temperature": [0.05, 0.1, 0.2, 0.5, 1.0],
    },

    # Experiment 2: tau only
    # temperature=0.2, c=0
    "tau": {
        "tau": [1, 0.99999, 0.9999, 0.999, 0.99],
    },

    # Experiment 3: joint temperature x tau
    # c fixed at 0 (decouple from tau_c experiment)
    "temperature_tau": {
        "temperature": [0.05, 0.1, 0.2, 0.5, 1.0],
        "tau":         [1, 0.99999, 0.9999, 0.999, 0.99],
    },

    # Experiment 4: joint tau x c
    # temperature fixed at best value from Experiment 3
    # passed via -temperature CLI flag
    "tau_c": {
        "tau": [1, 0.99999, 0.9999, 0.999, 0.99],
        "c":   [0, 1, 5, 10, 25, 50, 100],
    },
}

# Fixed values applied per experiment when not being swept
EXPERIMENT_FIXED = {
    "temperature":    lambda dataset: {"tau": PAPER_TAU[dataset], "c": 0,
                                       "temperature": PAPER_TEMPERATURE},
    "tau":            lambda dataset: {"temperature": PAPER_TEMPERATURE, "c": 0,
                                       "tau": PAPER_TAU[dataset]},
    "temperature_tau": lambda dataset: {"c": 0},
    "tau_c":          lambda dataset: {"temperature": PAPER_TEMPERATURE},
}


def make_args(base_cfg, overrides):
    """Build a Namespace from base config dict + override dict."""
    cfg = copy.deepcopy(base_cfg)
    cfg.update(overrides)
    return argparse.Namespace(**cfg)


def run_single(args, log_path):
    """Run one training configuration, redirecting stdout to log_path."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    if os.path.exists(log_path) and os.path.getsize(log_path) > 100:
        print(f"  [skip] {log_path}", flush=True)
        return

    print(f"  [run]  {log_path}", flush=True)
    t0 = time.time()

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
    base    = BASE_CONFIGS[dataset]
    log_dir = f"logs/tuning_si/{dataset}/{experiment}"

    if experiment in GRIDS:
        grid   = GRIDS[experiment]
        # Start from the fixed values for this experiment
        fixed  = EXPERIMENT_FIXED[experiment](dataset)
        # Let CLI overrides (e.g. -temperature for tau_c) override fixed values
        if overrides:
            fixed.update(overrides)

        keys   = list(grid.keys())
        values = list(grid.values())
        combos = list(itertools.product(*values))
        total  = len(combos)

        for i, combo in enumerate(combos, 1):
            params = copy.deepcopy(fixed)
            params.update(dict(zip(keys, combo)))
            tag      = "_".join(f"{k}{v}" for k, v in
                                sorted(dict(zip(keys, combo)).items()))
            log_path = f"{log_dir}/{tag}.log"
            print(f"\n[{i}/{total}] {dataset} {experiment}: "
                  f"{dict(zip(keys, combo))}", flush=True)
            args = make_args(base, params)
            run_single(args, log_path)

    elif experiment == "lr":
        assert overrides is not None, (
            "Pass -temperature, -tau, and -c for the lr experiment")
        lr_grid = [0.0001, 0.001, 0.01, 0.1]
        total   = len(lr_grid)
        for i, lr in enumerate(lr_grid, 1):
            params   = {"lr": lr, **overrides}
            tag      = (f"lr{lr}_t{overrides['temperature']}"
                        f"_tau{overrides['tau']}_c{overrides['c']}")
            log_path = f"{log_dir}/{tag}.log"
            print(f"\n[{i}/{total}] {dataset} lr sweep: lr={lr}", flush=True)
            args = make_args(base, params)
            run_single(args, log_path)

    elif experiment == "final":
        assert overrides is not None, (
            "Pass -temperature, -tau, -c, and -lr for the final experiment")
        configs = {
            "paper_baseline": {
                "temperature": PAPER_TEMPERATURE,
                "tau":         PAPER_TAU[dataset],
                "c":           0,
                "lr":          base["lr"],
            },
            "best_config": overrides,
        }
        for name, params in configs.items():
            params["ntrials"] = 5
            log_path          = f"{log_dir}/{name}.log"
            print(f"\nfinal {dataset} {name}: {params}", flush=True)
            args = make_args(base, params)
            run_single(args, log_path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("-dataset",     type=str,   required=True,
                   choices=["cora", "citeseer"])
    p.add_argument("-experiment",  type=str,   required=True,
                   choices=["temperature", "tau", "temperature_tau",
                            "tau_c", "lr", "final"])
    p.add_argument("-temperature", type=float, default=None,
                   help="Best temperature — required for tau_c, lr, final")
    p.add_argument("-tau",         type=float, default=None,
                   help="Best tau — required for lr and final")
    p.add_argument("-c",           type=int,   default=None,
                   help="Best c — required for lr and final")
    p.add_argument("-lr",          type=float, default=None,
                   help="Best lr — required for final")
    cli = p.parse_args()

    overrides = {}
    if cli.temperature is not None: overrides["temperature"] = cli.temperature
    if cli.tau         is not None: overrides["tau"]         = cli.tau
    if cli.c           is not None: overrides["c"]           = cli.c
    if cli.lr          is not None: overrides["lr"]          = cli.lr

    # Validate required overrides per experiment
    if cli.experiment == "tau_c" and "temperature" not in overrides:
        p.error("-temperature is required for the tau_c experiment")
    if cli.experiment == "lr" and not all(
            k in overrides for k in ["temperature", "tau", "c"]):
        p.error("-temperature, -tau, and -c are all required for the lr experiment")
    if cli.experiment == "final" and not all(
            k in overrides for k in ["temperature", "tau", "c", "lr"]):
        p.error("-temperature, -tau, -c, and -lr are all required for the final experiment")

    sweep(
        dataset=cli.dataset,
        experiment=cli.experiment,
        overrides=overrides if overrides else None,
    )


if __name__ == "__main__":
    main()
