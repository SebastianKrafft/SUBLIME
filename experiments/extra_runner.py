"""Runner that lets ``main.Experiment.train`` work on the five extra datasets
shipped in :mod:`experiments.extra_datasets` (Wine, Cancer, Digits, 20news,
ogbn-arxiv).

It monkey-patches the ``load_data`` symbol inside ``main`` so the training loop
transparently sees our loader. Re-uses every CLI flag from ``main.py``.

Examples
--------
    sublime-python -m experiments.extra_runner -dataset wine        -gsl_mode structure_inference \\
        -type_learner fgp  -k 20 -epochs 1000 -ntrials 5
    sublime-python -m experiments.extra_runner -dataset 20news      -gsl_mode structure_inference \\
        -type_learner mlp  -k 30 -epochs 1000 -ntrials 5 -sparse 1
    sublime-python -m experiments.extra_runner -dataset ogbn-arxiv  -gsl_mode structure_inference \\
        -type_learner mlp  -k 15 -epochs 1000 -ntrials 1 -sparse 1 -contrast_batch_size 2000
"""
from __future__ import annotations

import argparse
import sys

import main as sublime_main
from experiments.extra_datasets import load_extra_dataset, LOADERS


def _build_patched_load_data(seed: int):
    def patched(args):
        return load_extra_dataset(args.dataset, sparse=bool(args.sparse), seed=seed)
    return patched


def _build_parser():
    p = argparse.ArgumentParser()
    p.add_argument("-ntrials", type=int, default=5)
    p.add_argument("-sparse", type=int, default=0)
    p.add_argument("-gsl_mode", type=str, default="structure_inference",
                   choices=["structure_inference", "structure_refinement"])
    p.add_argument("-eval_freq", type=int, default=50)
    p.add_argument("-downstream_task", type=str, default="classification",
                   choices=["classification", "clustering"])
    p.add_argument("-dataset", type=str, required=True, choices=sorted(LOADERS))
    p.add_argument("-split_seed", type=int, default=0,
                   help="seed for the random train/val/test split (UCI only).")
    # GCL Module - Framework
    p.add_argument("-epochs", type=int, default=1000)
    p.add_argument("-lr", type=float, default=0.01)
    p.add_argument("-w_decay", type=float, default=0.0)
    p.add_argument("-hidden_dim", type=int, default=512)
    p.add_argument("-rep_dim", type=int, default=256)
    p.add_argument("-proj_dim", type=int, default=256)
    p.add_argument("-dropout", type=float, default=0.5)
    p.add_argument("-contrast_batch_size", type=int, default=0)
    p.add_argument("-nlayers", type=int, default=2)
    # GCL Module - Augmentation
    p.add_argument("-maskfeat_rate_learner", type=float, default=0.5)
    p.add_argument("-maskfeat_rate_anchor", type=float, default=0.5)
    p.add_argument("-dropedge_rate", type=float, default=0.5)
    # GSL Module
    p.add_argument("-type_learner", type=str, default="fgp",
                   choices=["fgp", "att", "mlp", "gnn"])
    p.add_argument("-k", type=int, default=20)
    p.add_argument("-sim_function", type=str, default="cosine",
                   choices=["cosine", "minkowski"])
    p.add_argument("-gamma", type=float, default=0.9)
    p.add_argument("-activation_learner", type=str, default="relu",
                   choices=["relu", "tanh"])
    # Evaluation Network (Classification)
    p.add_argument("-epochs_cls", type=int, default=200)
    p.add_argument("-lr_cls", type=float, default=0.001)
    p.add_argument("-w_decay_cls", type=float, default=0.0005)
    p.add_argument("-hidden_dim_cls", type=int, default=32)
    p.add_argument("-dropout_cls", type=float, default=0.5)
    p.add_argument("-dropedge_cls", type=float, default=0.25)
    p.add_argument("-nlayers_cls", type=int, default=2)
    p.add_argument("-patience_cls", type=int, default=10)
    # Structure Bootstrapping
    p.add_argument("-tau", type=float, default=1.0)
    p.add_argument("-c", type=int, default=0)
    p.add_argument("-gpu", type=int, default=0)

    # Abalation
    p.add_argument('-temperature', type=float, default=0.2)
    return p


def main():
    args = _build_parser().parse_args()

    # ogbn-arxiv: paper says 3-layer GCN with 256 hidden units as evaluator
    if args.dataset == "ogbn-arxiv":
        if args.nlayers_cls < 3:
            args.nlayers_cls = 3
        if args.hidden_dim_cls < 256:
            args.hidden_dim_cls = 256

    sublime_main.args = args
    sublime_main.load_data = _build_patched_load_data(args.split_seed)
    print(f"EXTRA_RUNNER dataset={args.dataset} gsl_mode={args.gsl_mode} "
          f"type_learner={args.type_learner} ntrials={args.ntrials}", flush=True)
    sublime_main.Experiment().train(args)


if __name__ == "__main__":
    main()
