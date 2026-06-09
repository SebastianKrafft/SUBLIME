"""
Helper runner for Figure 3 (training curves) and Table 4 (tau ablation).

Runs SUBLIME in structure-refinement mode on a chosen dataset and a chosen tau,
adding two extra stdout lines per evaluation so the per-eval test accuracy curve
can be parsed back from the log:

    CURVE_EVAL epoch=<E> val=<V> test=<T>

(The base script already emits one ``Epoch ... | CL Loss ...`` line per epoch,
which is enough for the loss curve.)

Usage (run from the repo root via the env wrapper):

    sublime-python -m experiments.curves_runner \\
        -dataset cora -tau 0.9999 -ntrials 1

All argparse flags from ``main.py`` are forwarded as-is.
"""

import argparse
import sys

import main as sublime_main


def _wrap_evaluate_adj_by_cls():
    """Monkey-patch ``Experiment.evaluate_adj_by_cls`` to also print val/test
    accuracy on every call (i.e. every ``eval_freq`` epochs).

    The relative order of these prints is exactly the eval schedule, so a
    downstream parser can recover the epoch by counting them.
    """
    original = sublime_main.Experiment.evaluate_adj_by_cls

    def wrapper(self, Adj, features, nfeats, labels, nclasses,
                train_mask, val_mask, test_mask, args):
        val, test, model = original(
            self, Adj, features, nfeats, labels, nclasses,
            train_mask, val_mask, test_mask, args,
        )
        # tensors -> floats
        v = float(val.item()) if hasattr(val, "item") else float(val)
        t = float(test.item()) if hasattr(test, "item") else float(test)
        print("CURVE_EVAL val={:.6f} test={:.6f}".format(v, t), flush=True)
        return val, test, model

    sublime_main.Experiment.evaluate_adj_by_cls = wrapper


def main():
    _wrap_evaluate_adj_by_cls()

    # Re-use main.py's argparse by re-running it on our argv (drop the module name).
    # main.py's __main__ block builds the parser; we replicate it here by calling
    # the same argument setup. Easiest: exec main's argparse block? Instead,
    # just import it as a module and reuse its parser if it exposes one.
    #
    # main.py builds the parser inside ``if __name__ == "__main__":`` so we
    # cannot import it directly. We rebuild the parser by deferring to a tiny
    # subprocess-free re-parse: we execute main.py's parser-setup region by
    # reading the file. To avoid that fragility, we mimic main.py's CLI here
    # explicitly. If main.py grows new flags, update this list too.
    p = argparse.ArgumentParser()
    # Experimental setup
    p.add_argument("-ntrials", type=int, default=1)
    p.add_argument("-sparse", type=int, default=0)
    p.add_argument("-gsl_mode", type=str, default="structure_refinement",
                   choices=["structure_inference", "structure_refinement"])
    p.add_argument("-eval_freq", type=int, default=50)
    p.add_argument("-downstream_task", type=str, default="classification",
                   choices=["classification", "clustering"])
    p.add_argument("-dataset", type=str, default="cora")
    # GCL Module - Framework
    p.add_argument("-epochs", type=int, default=4000)
    p.add_argument("-lr", type=float, default=0.01)
    p.add_argument("-w_decay", type=float, default=0.0)
    p.add_argument("-hidden_dim", type=int, default=512)
    p.add_argument("-rep_dim", type=int, default=256)
    p.add_argument("-proj_dim", type=int, default=256)
    p.add_argument("-dropout", type=float, default=0.5)
    p.add_argument("-contrast_batch_size", type=int, default=0)
    p.add_argument("-nlayers", type=int, default=2)
    # GCL Module - Augmentation
    p.add_argument("-maskfeat_rate_learner", type=float, default=0.2)
    p.add_argument("-maskfeat_rate_anchor", type=float, default=0.2)
    p.add_argument("-dropedge_rate", type=float, default=0.5)
    # GSL Module
    p.add_argument("-type_learner", type=str, default="fgp",
                   choices=["fgp", "att", "mlp", "gnn"])
    p.add_argument("-k", type=int, default=30)
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
    p.add_argument("-tau", type=float, default=1)
    p.add_argument("-c", type=int, default=0)
    p.add_argument("-gpu", type=int, default=0)

    # Abalation
    p.add_argument('-temperature', type=float, default=0.2)

    args = p.parse_args()

    # main.py's free-floating ``args`` is referenced as a module-level name from
    # inside ``Experiment.loss_gcl``; install ours there before training.
    sublime_main.args = args

    sublime_main.Experiment().train(args)


if __name__ == "__main__":
    main()
