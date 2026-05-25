# ILP with Parallel Tempering

<p align="center">
  <img src="assets/pt_methods.pdf" alt="ILP with Parallel Tempering overview" width="720"/>
</p>

## Overview

`PT_ILP` is a framework for solving 0/1 Integer Linear Programs (ILPs) with
MCMC samplers driven by **Parallel Tempering (PT)**. Constraints are folded into
the objective via a quadratic / hinge penalty and a temperature ladder is run
in parallel, with DEO swaps exchanging states between adjacent chains. The
package ships with several samplers (locally balanced proposals, GWG,
path-auxiliary, DMALA, block Gibbs, hamming-ball, random walk, ...) and a set of
benchmark instance families:

- **MVC** — Minimum Vertex Cover (1000, 2000 vars)
- **MIS** — Maximum Independent Set (1500, 3000 vars)
- **CA**  — Combinatorial Auction (2000, 4000 vars)
- **SC**  — Set Covering (1000, 2000, 4000 vars)

Five temperature schedules are supported: `exp_decay`, `pt`, `pt_exp_decay`,
`pen_pt`, and `pen_pt_exp_decay` (the last two also sweep a per-chain penalty
ladder in addition to temperature).

## Installation Guide

Install JAX with the CUDA backend appropriate for your machine by following
the [official JAX install guide](https://github.com/google/jax#installation).
Then, from the project root, run:

    pip install -e .

The pinned versions used in our experiments are listed in `requirements.txt`
(JAX 0.4.30 with CUDA 12, NumPy <2, TensorFlow 2.19, PySCIPOpt 5.5, ...).

## Test

A single sampling run is launched through one of the shell scripts in
`PT_ILP/scripts/`. The arguments are
`instance_name max_num_vars t_schedule [sampler] [num_flips] [adaptive] [formulation]`.

Run with a fixed step budget:

    bash PT_ILP/scripts/run_sampling_steplimit.sh mis 1500 pen_pt_exp_decay lbp 3 

Run with a wall-clock budget (200 s by default):

    bash PT_ILP/scripts/run_sampling_runtimelimit.sh mis 1500 pen_pt_exp_decay lbp 3 

Results are written under `PT_ILP/results/` and logged to Weights & Biases.

## Acknowledgements

This codebase is built on top of
[DISCS: A Benchmark for Discrete Sampling](https://github.com/google-research/discs)
([paper](https://openreview.net/pdf?id=oi1MUMk5NF)).
