"""Main class that runs sampler on the model to generate chains."""

import functools
import time
from discs.common import math_util as math
from discs.common import utils
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from discs.common.parallel_tempering import swap_samples_deo, swap_samples_seo, swap_samples_reversible
import discs.common.penalty_parallel_tempering as penalty_pt
import wandb
import os
from pyscipopt import Model, quicksum
from jax.experimental import io_callback
from concurrent.futures import ProcessPoolExecutor
import functools
from scipy.optimize import linprog
from joblib import Parallel, delayed

def solve_sub_problem(x_init, cont_indices, int_indices, const_m, rhs, lhs, obj_coeffs):
    """Solve a sub-ILP problem with fixed integer variables and free continuous variables."""
    model = Model("SubProblem")
    model.setParam("limits/time", 10)
    x_vars = [None] * 1083
    for i in range(1083):
        if i in cont_indices:
            x_vars[i] = model.addVar(vtype="C", name=f"x_{i}", lb=None, ub=None)
        if i in int_indices:
            x_vars[i] = model.addVar(name=f"x_{i}", vtype="B", lb=x_init[i].item(), ub=x_init[i].item())

    # set the objective (use the passed-in obj_coeffs parameter)
    model.setObjective(
        quicksum(-obj_coeffs[i] * x_vars[i] for i in range(len(x_vars))),
        "minimize"
    )

    # set the constraints
    for j in range(const_m.shape[1]):
        model.addCons(quicksum(float(const_m[j][h]) * x_vars[h] for h in range(len(x_vars))) <= float(rhs[j]))
        model.addCons(quicksum(float(const_m[j][h]) * x_vars[h] for h in range(len(x_vars))) >= float(lhs[j]))

    # optimize sub-ILP
    model.optimize()
    if model.getNSols() > 0:
        sol = model.getBestSol()
        x_ = np.array([model.getSolVal(sol, v) for v in x_vars])
    else:
        x_ = np.array([x_init[i].item() for i in range(len(x_vars))])
    return x_


def _solve_single(args):
    """Top-level wrapper for ProcessPoolExecutor (must be picklable)."""
    x_single, cont_indices, int_indices, const_m, rhs, lhs, obj_coeffs = args
    return solve_sub_problem(x_single, cont_indices, int_indices, const_m, rhs, lhs, obj_coeffs)


class Experiment:
    """Experiment class that generates chains of samples."""

    def __init__(self, config):
        self.config = config.experiment
        self.config_model = config.model
        self.sampler_name = config.sampler.name
        if "lbp" in self.sampler_name or "path_auxiliary" in self.sampler_name:
            self.sampler_name = (
                self.sampler_name
                + f"_nflip{config.sampler.num_flips}"
                + ("_adaptive" if config.sampler.adaptive else "")
            )
        self.parallel = False
        self.sample_idx = None
        self.num_saved_samples = config.get("nun_saved_samples", 4)
        if jax.local_device_count() != 1 and self.config.run_parallel:
            self.parallel = True

    def _initialize_model_and_sampler(self, rnd, model, sampler):
        """Initializes model params, sampler state and gets the initial samples."""

        if self.config.evaluator == "co_eval":
            sampler_init_state_fn = jax.vmap(sampler.make_init_state)
        else:
            sampler_init_state_fn = sampler.make_init_state
        model_init_params_fn = model.make_init_params
        rng_param, rng_x0, rng_state = jax.random.split(rnd, num=3)
        # params of the model
        params = model_init_params_fn(jax.random.split(rng_param, self.config.num_models))
        # initial samples
        num_samples = self.config.batch_size * self.config.num_models
        x0 = model.get_init_samples(rng_x0, num_samples)
        # initial state of sampler
        state = sampler_init_state_fn(jax.random.split(rng_state, self.config.num_models))
        return params, x0, state

    def _prepare_data(self, params, x, state):
        use_put_replicated = False
        reshape_all = True
        if self.config.evaluator != "co_eval":
            if self.parallel:
                assert self.config.batch_size % jax.local_device_count() == 0
                mini_batch = self.config.batch_size // jax.local_device_count()
                bshape = (jax.local_device_count(),)
                x_shape = bshape + (mini_batch,) + self.config_model.shape
                use_put_replicated = True
                if self.sample_idx:
                    self.sample_idx = jnp.array(
                        [self.sample_idx] * (jax.local_device_count() // self.config.num_models)
                    )
            else:
                reshape_all = False
                bshape = ()
                x_shape = (self.config.batch_size,) + self.config_model.shape
        else:
            if self.parallel:
                if self.config.num_models >= jax.local_device_count():
                    assert self.config.num_models % jax.local_device_count() == 0
                    num_models_per_device = self.config.num_models // jax.local_device_count()
                    bshape = (jax.local_device_count(), num_models_per_device)
                    x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
                else:
                    assert self.config.batch_size % jax.local_device_count() == 0
                    batch_size_per_device = self.config.batch_size // jax.local_device_count()
                    use_put_replicated = True
                    bshape = (jax.local_device_count(), self.config.num_models)
                    x_shape = bshape + (batch_size_per_device,) + self.config_model.shape
                    if self.sample_idx:
                        self.sample_idx = jnp.array(
                            [self.sample_idx] * (jax.local_device_count() // self.config.num_models)
                        )
            else:
                bshape = (self.config.num_models,)
                x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
        fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
        if reshape_all:
            if not use_put_replicated:
                state = jax.tree_util.tree_map(fn_breshape, state)
                params = jax.tree_util.tree_map(fn_breshape, params)
            else:
                params = jax.device_put_replicated(params, jax.local_devices())
                state = jax.device_put_replicated(state, jax.local_devices())
        x = jnp.reshape(x, x_shape)

        print("x shape: ", x.shape)
        print("state shape: ", state["steps"].shape)
        return params, x, state, bshape

    def _compile_sampler_step(self, step_fn):
        if not self.parallel:
            compiled_step = jax.jit(step_fn)
        else:
            compiled_step = jax.pmap(step_fn)
        return compiled_step

    def _compile_evaluator(self, obj_fn):
        if not self.parallel:
            compiled_obj_fn = jax.jit(obj_fn)
        else:
            compiled_obj_fn = jax.pmap(obj_fn)
        return compiled_obj_fn

    def _compile_fns(self, sampler, model, evaluator):
        if self.config.evaluator == "co_eval":
            step_fn = jax.vmap(functools.partial(sampler.step, model=model))
            obj_fn = self._vmap_evaluator(evaluator, model)
        else:
            step_fn = functools.partial(sampler.step, model=model)
            obj_fn = evaluator.evaluate
        compiled_obj_only_fn = jax.vmap(functools.partial(model.objective))
        compiled_penalty_fn = jax.vmap(functools.partial(model.penalty))
        compiled_mask_fn = jax.jit(jax.vmap(self.mask_per_instance, in_axes=(0, 0)))
        if self.config.pt == "deo":
            _pt_swap_fn = swap_samples_deo
        elif self.config.pt == "reversible":
            _pt_swap_fn = swap_samples_reversible
        else:
            _pt_swap_fn = swap_samples_seo
        compiled_pt_temp_fn = jax.jit(
            jax.vmap(
                functools.partial(_pt_swap_fn),
                in_axes=(0, 0, 0, None, None),
            )
        )
        compiled_pt_pen_fn = jax.jit(
            jax.vmap(
                functools.partial(
                    penalty_pt.swap_samples_deo if self.config.pt == "deo" else penalty_pt.swap_samples_seo
                ),
                in_axes=(0, 0, 0, 0, 0, None,None)
            )
        )
        
        get_hop = jax.jit(self._get_hop)
        compiled_step = self._compile_sampler_step(step_fn)
        compiled_step_burnin = compiled_step
        compiled_step_mixing = compiled_step
        compiled_obj_fn = self._compile_evaluator(obj_fn)
        model_frwrd = jax.jit(model.forward)
        return (
            compiled_step_burnin,
            compiled_step_mixing,
            get_hop,
            compiled_obj_fn,
            model_frwrd,
            compiled_mask_fn,
            compiled_obj_only_fn,
            compiled_penalty_fn,
            compiled_pt_temp_fn,
            compiled_pt_pen_fn,
        )

    def _get_hop(self, x, new_x):
        return jnp.sum(abs(x - new_x)) / self.config.batch_size / self.config.num_models

    def _compute_chain(
        self,
        compiled_fns,
        state,
        params,
        rng,
        x,
        saver,
        evaluator,
        bshape,
        model,
    ):
        raise NotImplementedError

    def vmap_evaluator(self, evaluator, model):
        raise NotImplementedError

    def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
        rnd = jax.random.PRNGKey(rnd_key)
        params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
        if params is None:
            print("Params is NONE")
            return False
        params, x, state, breshape = self._prepare_data(params, x, state)
        compiled_fns = self._compile_fns(sampler, model, evaluator)
        return [
            compiled_fns,
            state,
            params,
            rnd,
            x,
            saver,
            evaluator,
            breshape,
            model,
        ]

    def _get_chains_and_evaluations(self, model, sampler, evaluator, saver, rnd_key=0):
        """Sets up the model and the samlping alg and gets the chain of samples."""
        preprocessed_info = self.preprocess(model, sampler, evaluator, saver, rnd_key=1)
        if not preprocessed_info:
            return False
        self._compute_chain(*preprocessed_info)
        return True

    def get_results(self, model, sampler, evaluator, saver):
        self._get_chains_and_evaluations(model, sampler, evaluator, saver)


class CO_Experiment(Experiment):
    """Class used to run annealing schedule for CO problems."""

    def get_results(self, model, sampler, evaluator, saver):
        self.all_chain = []
        self.all_best_ratio = []
        self.all_current_time = []
        self.all_best_samples = []
        while True:
            if not self._get_chains_and_evaluations(model, sampler, evaluator, saver):
                break

    def _initialize_model_and_sampler(self, rnd, model, sampler):
        data_list, x0, state = super()._initialize_model_and_sampler(rnd, model, sampler)
        if data_list is None:
            return None, x0, state
        sample_idx, params, reference_obj = zip(*data_list)
        params = flax.core.frozen_dict.unfreeze(utils.tree_stack(params))
        self.ref_obj = jnp.array(reference_obj)
        if self.config_model.name in ["mis", "ilp"]:
            self.ref_obj = jnp.ones_like(self.ref_obj)
        self.sample_idx = jnp.array(sample_idx)
        return params, x0, state

    def _vmap_evaluator(self, evaluator, model):
        obj_fn = jax.vmap(functools.partial(evaluator.evaluate, model=model))
        return obj_fn

    def _build_temperature_schedule(self, config):
        """Temperature schedule."""

        if config.t_schedule == "constant":
            schedule = lambda step: step * 0 + jnp.array(config.init_temperature)
        elif config.t_schedule == "linear":
            assert config.final_temperature is not None  # cannot be None for linear schedule
            schedule = optax.linear_schedule(
                config.init_temperature, config.final_temperature, config.chain_length
            )
        elif config.t_schedule == "exp_decay":
            schedule = optax.exponential_decay(
                config.init_temperature,
                config.chain_length,
                config.decay_rate,
                end_value=0.0,
            )
        elif config.t_schedule == "pt":
            schedule = (
                lambda step: step * 0
                + jnp.geomspace(self.config.t_min, self.config.t_max, num=self.config.batch_size)[
                    None, :
                ]
            )
        elif config.t_schedule == "pt_exp_decay":
            pt_base = jnp.geomspace(
                self.config.t_min, self.config.t_max, num=self.config.batch_size
            )[None, :]
            decay_schedule = optax.exponential_decay(
                1.0,
                config.chain_length,
                config.decay_rate,
                end_value=0.0,
            )
            schedule = lambda step: pt_base * decay_schedule(step)
        elif config.t_schedule == "pen_pt":
            schedule = lambda step: step * 0 + jnp.array(config.init_temperature)
        elif config.t_schedule == "pen_pt_exp_decay":
            schedule = optax.exponential_decay(
                config.init_temperature,
                config.chain_length,
                config.decay_rate,
                end_value=0.0,
            )
        else:
            raise ValueError("Unknown schedule %s" % config.t_schedule)
        return schedule

    # === ADDED by Claude: pass cont_bounds + correct obj_sign (was hardcoded 1.0; item is minimize so obj_sign=-1) ===
    _lp_success_count = 0
    _lp_total_count = 0
    # === ADDED by Claude (verification): counters + log-throttling for linprog vs SCIP cross-check ===
    _lp_verify_count = 0
    _lp_verify_match = 0
    _lp_verify_max_logs = 5  # only print first N detailed mismatches to avoid spam
    # ---------------------------------------------------------------------
    # MODE 1: classic sub-LP — fix x_int from MCMC sample, optimise y_cont.
    # ---------------------------------------------------------------------
    def solve_sub_lp(self, x_, const_m, rhs, lhs, obj_coeffs,
                     int_indices, cont_indices, cont_bounds, obj_sign=1.0):
        """Sub-LP: fix integer vars at MCMC values, optimise only continuous slice.

        max  obj_sign * c_cont^T y_cont
        s.t. lhs - A_int x_int <= A_cont y_cont <= rhs - A_int x_int
             y_cont in cont_bounds

        Returns x_ with cont_indices overwritten by the LP optimum, fallback to
        x_ unchanged on infeasibility.
        """
        x_int = x_[int_indices]
        A_cont = const_m[:, cont_indices]
        A_int  = const_m[:, int_indices]
        int_contribution = A_int @ x_int

        adj_rhs = rhs - int_contribution
        adj_lhs = lhs - int_contribution

        A_ub_rows, b_ub_rows = [], []
        finite_ub = np.isfinite(adj_rhs)
        if finite_ub.any():
            A_ub_rows.append(A_cont[finite_ub])
            b_ub_rows.append(adj_rhs[finite_ub])
        finite_lb = np.isfinite(adj_lhs)
        if finite_lb.any():
            A_ub_rows.append(-A_cont[finite_lb])
            b_ub_rows.append(-adj_lhs[finite_lb])
        A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
        b_ub = np.concatenate(b_ub_rows) if b_ub_rows else None

        c_cont = -obj_sign * obj_coeffs[cont_indices]
        bounds = cont_bounds  # per-cont-var (lb, ub); None for unbounded sides

        result = linprog(c_cont, A_ub=A_ub, b_ub=b_ub, bounds=bounds, method="highs")

        CO_Experiment._lp_total_count += 1
        if result.success:
            CO_Experiment._lp_success_count += 1
            x_result = x_.copy().astype(np.float64)
            x_result[cont_indices] = result.x
            return x_result
        return x_   # fallback: keep current values if infeasible

    def solve_sub_lp_batch(self, x_batch, const_m, rhs, lhs, obj_coeffs,
                           int_indices, cont_indices, cont_bounds, obj_sign=1.0):
        results = [
            self.solve_sub_lp(x_, const_m, rhs, lhs, obj_coeffs,
                              int_indices, cont_indices, cont_bounds, obj_sign)
            for x_ in x_batch
        ]
        return np.stack(results)

    # ---------------------------------------------------------------------
    # MODE 2: full MILP — solve the actual MILP via SCIP (integer constraints
    # kept). Returns a true integer-feasible solution (no rounding needed).
    # ---------------------------------------------------------------------
    def solve_full_milp(self, x_, const_m, rhs, lhs, obj_coeffs,
                        int_indices, cont_indices, var_bounds, var_types,
                        obj_sign=1.0, time_limit=10.0, warm_start=None):
        """Solve the full MILP with SCIP (no relaxation).

        max  obj_sign * c^T x
        s.t. lhs <= A x <= rhs
             x_i in [lb_i, ub_i]
             x_i integer / binary for the corresponding indices (var_types)

        - var_types: per-var int (0 = BINARY, 1 = INTEGER, 3 = CONTINUOUS).
        - time_limit (seconds): hard cap; MILP can take long, so this matters.
        - warm_start (optional): initial solution to seed SCIP. If passed, must
          be a length-n_vars array; we feed it as a candidate via
          createPartialSol/setSolVal.

        SCIP minimises by default, so we set objective to -obj_sign * c (i.e.
        minimise this) — equivalent to maximising obj_sign * c^T x.

        Returns the best integer-feasible solution found within the time limit
        (or x_ unchanged on failure).
        """
        n_vars = len(obj_coeffs)
        m = Model("FullMILP")
        m.hideOutput()
        m.setParam("limits/time", float(time_limit))

        # Add variables with proper types and bounds.
        xs = []
        for k in range(n_vars):
            lb_k, ub_k = var_bounds[k] if k < len(var_bounds) else (None, None)
            lb_v = -1e20 if lb_k is None else float(lb_k)
            ub_v =  1e20 if ub_k is None else float(ub_k)
            t = int(var_types[k]) if k < len(var_types) else 3
            if t == 0:    vtype = "B"   # binary
            elif t == 1:  vtype = "I"   # general integer
            else:         vtype = "C"   # continuous
            xs.append(m.addVar(name=f"x_{k}", vtype=vtype, lb=lb_v, ub=ub_v))

        # Constraints: lhs[j] <= sum_k A[j,k] * x_k <= rhs[j].
        n_rows = const_m.shape[0]
        for j in range(n_rows):
            row = const_m[j]
            expr = quicksum(float(row[k]) * xs[k] for k in range(n_vars) if row[k] != 0.0)
            ub_j = float(rhs[j]) if np.isfinite(rhs[j]) else None
            lb_j = float(lhs[j]) if np.isfinite(lhs[j]) else None
            if ub_j is not None: m.addCons(expr <= ub_j)
            if lb_j is not None: m.addCons(expr >= lb_j)

        # SCIP minimises this, equivalent to maximising obj_sign * c^T x.
        m.setObjective(quicksum(float(-obj_sign * obj_coeffs[k]) * xs[k]
                                for k in range(n_vars)),
                       "minimize")

        # Optional warm start (use the MCMC sample as a candidate solution).
        if warm_start is not None:
            try:
                sol = m.createPartialSol()
                for k in range(n_vars):
                    m.setSolVal(sol, xs[k], float(warm_start[k]))
                m.addSol(sol, free=True)
            except Exception:
                pass  # warm-start best-effort only

        m.optimize()

        CO_Experiment._lp_total_count += 1
        if m.getNSols() > 0:
            CO_Experiment._lp_success_count += 1
            sol = m.getBestSol()
            return np.array([m.getSolVal(sol, v) for v in xs], dtype=np.float64)
        return x_   # fallback: keep current values if SCIP found nothing

    def solve_full_milp_batch(self, x_batch, const_m, rhs, lhs, obj_coeffs,
                              int_indices, cont_indices, var_bounds, var_types,
                              obj_sign=1.0, time_limit=10.0):
        """Batch wrapper.

        IMPORTANT: the MILP itself does not depend on the per-sample x values
        (they are only used as warm-start hints), so solving it once per batch
        and broadcasting is the natural choice. We do exactly that here:
        SCIP is called ONCE (warm-started from x_batch[0]) and the result is
        copied to every batch position. This avoids `batch_size` redundant
        MILP solves per LP step and keeps the chain from idling.

        WARNING: broadcasting makes all replicas identical immediately after
        each MILP step. Parallel-tempering / batch diversity is therefore lost
        on those steps. If you want diverse warm-starts per replica, change
        this to loop over x_batch (slow: time_limit * batch_size per step).
        """
        x_one = self.solve_full_milp(
            x_batch[0], const_m, rhs, lhs, obj_coeffs,
            int_indices, cont_indices, var_bounds, var_types,
            obj_sign=obj_sign, time_limit=time_limit, warm_start=x_batch[0],
        )
        return np.broadcast_to(x_one[None, :], x_batch.shape).copy()
    # === END ADDED ===

    # ---------------------------------------------------------------------
    # LP-guided proposals (non-breaking, opt-in via config "lp_guided_mask").
    # Idea: solve the FULL LP relaxation once; integer positions that LP is
    # confident about (close to 0 or 1) are FROZEN; only LP-fractional integer
    # positions remain in the MCMC's flippable set. This restricts the
    # 1050-dim search to the 30~200-dim "uncertain" subspace LP could not
    # decide on its own — closer to LNS / RINS heuristics for MILP.
    # ---------------------------------------------------------------------
    def _compute_lp_guided_mask(self, const_m, rhs, lhs, obj_coeffs,
                                int_indices, var_bounds, obj_sign,
                                threshold=0.05, soft=False):
        """Solve LP relaxation, return a mask vector of shape (n_vars,).

        Output convention (matches existing `params['mask']`):
        - 0.0  -> position is frozen (continuous, or LP-confident integer)
        - 1.0  -> position is fully flippable
        - in (0,1) only when soft=True (proposal-weight mode)

        On LP failure: returns all-ones for integer positions (no restriction)
        so the chain is never WORSE than the un-guided baseline.
        """
        # Build linprog inequalities from row-wise lhs <= A x <= rhs (same as
        # solve_full_milp / solve_sub_ilp_relaxation, but kept LOCAL to this
        # helper so we don't perturb anything else).
        A_ub_rows, b_ub_rows = [], []
        finite_ub = np.isfinite(rhs)
        if finite_ub.any():
            A_ub_rows.append(const_m[finite_ub])
            b_ub_rows.append(rhs[finite_ub])
        finite_lb = np.isfinite(lhs)
        if finite_lb.any():
            A_ub_rows.append(-const_m[finite_lb])
            b_ub_rows.append(-lhs[finite_lb])
        A_ub = np.vstack(A_ub_rows) if A_ub_rows else None
        b_ub = np.concatenate(b_ub_rows) if b_ub_rows else None

        c = -obj_sign * obj_coeffs                         # linprog minimises
        result = linprog(c, A_ub=A_ub, b_ub=b_ub,
                         bounds=var_bounds, method="highs")

        n_vars = len(obj_coeffs)
        mask = np.zeros(n_vars, dtype=np.float32)          # cont stays 0

        if not result.success:
            # Fallback: open all integer positions (preserve baseline behaviour).
            int_arr = np.asarray(int_indices, dtype=np.int64)
            mask[int_arr] = 1.0
            return mask, None

        x_lp = np.asarray(result.x, dtype=np.float64)

        int_arr = np.asarray(int_indices, dtype=np.int64)
        # frac in [0, 0.5]: 0 = LP picked exactly 0 or 1, 0.5 = max uncertain.
        frac = np.minimum(x_lp[int_arr], 1.0 - x_lp[int_arr])

        if soft:
            # Soft weights: proportional to fractionality (∈ [0, 1]).
            mask[int_arr] = np.clip(frac / 0.5, 0.0, 1.0).astype(np.float32)
        else:
            # Hard mask: keep ONLY the integer positions LP is uncertain on.
            keep = frac > threshold
            mask[int_arr[keep]] = 1.0

        return mask, x_lp

    def _compute_chain(
        self,
        compiled_fns,
        state,
        params,
        rng,
        x,
        saver,
        evaluator,
        bshape,
        model,
    ):
        """Generates the chain of samples."""

        (
            chain,
            acc_ratios,
            hops,
            running_time,
            best_ratio,
            init_temperature,
            t_schedule,
            sample_mask,
            best_samples,
        ) = self._initialize_chain_vars(bshape)

        stp_burnin, stp_mixing, get_hop, obj_fn, _, mask_fn, obj_only_fn, penalty_fn, pt_temp_fn, pt_pen_fn = (
            compiled_fns
        )
        fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])

        wandb_name = f"{self.config_model.graph_type}_{self.config_model.max_num_nodes}_{self.sampler_name}_{self.config_model.formulation}_lambda{self.config_model.penalty}_bsz{self.config.batch_size}_{self.config.t_schedule}"
        if self.config.t_schedule == "constant":
            wandb_name += f"_init{self.config.init_temperature}"
        elif self.config.t_schedule == "exp_decay":
            wandb_name += f"_init{self.config.init_temperature}_decay{self.config.decay_rate}"
        elif self.config.t_schedule == "pt":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_tmin{self.config.t_min}_tmax{self.config.t_max}"
        elif self.config.t_schedule == "pt_exp_decay":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_tmin{self.config.t_min}_tmax{self.config.t_max}_decay{self.config.decay_rate}"
        elif self.config.t_schedule == "pen_pt":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_lmin{self.config.l_min}_lmax{self.config.l_max}"
        elif self.config.t_schedule == "pen_pt_exp_decay":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_lmin{self.config.l_min}_lmax{self.config.l_max}_decay{self.config.decay_rate}"
        else:
            raise ValueError(f"Unknown t_schedule: {self.config.t_schedule}")
        wandb.init(name=wandb_name)

        # reheated mechanism
        best_eval_val = jnp.ones(self.config.num_models) * -jnp.inf
        value_chain = jnp.zeros((100, self.config.num_models, self.config.batch_size))
        if self.config.reweight == "reheat":
            shape = (self.config.num_models, self.config.batch_size)
            fake_step = jnp.ones(shape, dtype=jnp.int32)
            max_specific_heat = jnp.zeros(shape, dtype=jnp.float32)
            reheat_step = jnp.zeros(shape, dtype=jnp.int32)
            print_specific_heat = False
            skip_step = 200
            wandering_length = 100
            threshold = 0.5
            reheat_time = jnp.zeros((self.config.num_models, self.config.batch_size))
            trapped_num = jnp.zeros(shape, dtype=jnp.int32)
            trapped_threshold_length = jnp.ones(shape, dtype=jnp.int32) * wandering_length
            old_value = jnp.zeros(shape, dtype=jnp.float32)
            temp_shape = bshape + (self.config.batch_size,)
            init_temperature = jnp.ones(temp_shape, dtype=jnp.float32)
            params["temperature"] = t_schedule(0) * init_temperature  # (100,32)

        pt_step = 0
        num_log = 0
        mean_accept = 0
        pairs = []
        acceptance_ratio = jnp.ones((self.config.num_models, self.config.batch_size)) * -jnp.nan
        elapsed_time = 0

        if 'var_types' in params:
            cont_indices = np.where(params['var_types'] == 3)[1]
            int_indices  = np.where(params['var_types'] == 0)[1]

        const_m    = np.array(params['constraint_matrix'][0])
        rhs        = np.array(params['constraint_rhs'][0])
        lhs        = np.array(params['constraint_lhs'][0])
        obj_coeffs = np.array(params['obj_coeffs'].flatten())
        # === ADDED by Claude: per-variable (lb, ub) for LP; convert +/-inf to None (linprog convention) ===
        # Build BOTH cont_bounds (for sub-LP mode) AND var_bounds (for full-LP-
        # relaxation mode) so the call site can switch between them at runtime.
        if 'var_types' in params and 'var_lbs' in params and 'var_ubs' in params:
            var_lbs_np = np.array(params['var_lbs'][0])
            var_ubs_np = np.array(params['var_ubs'][0])
            n_vars_total = var_lbs_np.shape[0]
            # Bounds for ALL vars (used by solve_sub_ilp_relaxation)
            var_bounds = [
                (None if not np.isfinite(var_lbs_np[i]) else float(var_lbs_np[i]),
                 None if not np.isfinite(var_ubs_np[i]) else float(var_ubs_np[i]))
                for i in range(n_vars_total)
            ]
            # Bounds restricted to continuous slice (used by solve_sub_lp)
            cont_bounds = [
                (None if not np.isfinite(var_lbs_np[i]) else float(var_lbs_np[i]),
                 None if not np.isfinite(var_ubs_np[i]) else float(var_ubs_np[i]))
                for i in cont_indices
            ]
        else:
            var_bounds  = [(None, None)] * len(obj_coeffs)
            cont_bounds = [(None, None)] * (len(cont_indices) if 'var_types' in params else 0)
        # obj_sign from model (e.g., -1.0 for item/sc/mvc minimise problems). Previously hardcoded to 1.0 below -> LP direction was wrong.
        lp_obj_sign = float(getattr(model, "obj_sign", -1.0))
        lp_every = int(self.config.get("lp_every", 50))
        # lp_mode: "sub"  = classic sub-LP (fix int, optimise cont, linprog)
        #          "full" = full MILP solve via SCIP (integer constraints kept)
        lp_mode = str(self.config.get("lp_mode", "sub")).lower()
        # Time limit for SCIP per call in "full" mode (seconds). Tune with care:
        # this fires every lp_every steps so total cost = chain_steps/lp_every * milp_time_limit.
        milp_time_limit = float(self.config.get("milp_time_limit", 30.0))
        # var_types as plain numpy (needed by SCIP MILP path to set vtype per var).
        var_types_np = (
            np.array(params['var_types'][0])
            if 'var_types' in params else np.full(len(obj_coeffs), 3, dtype=np.int32)
        )
        print(f"[LP schedule] lp_every={lp_every}, lp_obj_sign={lp_obj_sign}, "
              f"lp_mode={lp_mode}, n_cont={len(cont_indices) if 'var_types' in params else 0}, "
              f"milp_time_limit={milp_time_limit if lp_mode == 'full' else 'n/a'}")
        # === END ADDED ===

        # === ADDED by Claude: ILS — Iterated Local Search (opt-in, default off) ===
        # When bks_obj has not improved for `ils_stagnation_window` steps, randomly
        # flip `ils_n_flips` integer positions PER REPLICA to escape the local
        # vertex. Sub-LP / MCMC will then re-stabilise from the new state.
        ils_enabled = bool(self.config.get("ils_enabled", False))
        ils_window = int(self.config.get("ils_stagnation_window", 2000))
        ils_n_flips = int(self.config.get("ils_n_flips", 50))
        ils_last_best = float("-inf")
        ils_last_improve_step = 0
        ils_perturb_count = 0
        if ils_enabled:
            print(f"[ILS] enabled: stagnation_window={ils_window} steps, "
                  f"n_flips={ils_n_flips} positions per replica")
        # === END ADDED ===

        # === ADDED by Claude: LP-guided mask (opt-in, default off; non-breaking) ===
        # Solve the LP relaxation ONCE before the chain starts. Integer
        # positions LP is confident on (frac < threshold) are frozen by the
        # mask; only LP-fractional integers remain flippable.
        # Disable by passing --lp_guided_mask=False (default), which skips this
        # entire block and leaves params['mask'] handling unchanged.
        lp_guided = bool(self.config.get("lp_guided_mask", False))
        lp_guided_thresh = float(self.config.get("lp_guided_threshold", 0.05))
        lp_guided_soft = bool(self.config.get("lp_guided_soft", False))
        lp_mask_jnp = None  # None disables mask multiplication in the loop
        if lp_guided and 'var_types' in params:
            lp_mask_np, x_lp_dbg = self._compute_lp_guided_mask(
                const_m, rhs, lhs, obj_coeffs,
                int_indices, var_bounds, lp_obj_sign,
                threshold=lp_guided_thresh, soft=lp_guided_soft,
            )
            n_total_int = len(int_indices)
            n_active = int(np.sum(lp_mask_np[np.asarray(int_indices)] > 0))
            print(f"[LP-guided] threshold={lp_guided_thresh}, soft={lp_guided_soft}, "
                  f"active int positions = {n_active}/{n_total_int} "
                  f"({100.0*n_active/max(n_total_int,1):.1f}%)")
            lp_mask_jnp = jnp.asarray(lp_mask_np)
        # === END ADDED ===
        # if lp_mode == "full":
        # x_opt = self.solve_full_milp_batch(
        #     np.array(x.squeeze(axis=0)), const_m, rhs, lhs, obj_coeffs,
        #     int_indices, cont_indices, var_bounds, var_types_np,
        #     obj_sign=lp_obj_sign, time_limit=milp_time_limit,
        # )
        # else:  # default: classic sub-LP
        #     x_opt = self.solve_sub_lp_batch(
        #         np.array(x.squeeze(axis=0)), const_m, rhs, lhs, obj_coeffs,
        #         int_indices, cont_indices, cont_bounds, obj_sign=lp_obj_sign,
        #     )
        # if x_opt is not None:
        #     x = jnp.expand_dims(jnp.array(x_opt), 0)

        for step in tqdm.tqdm(range(1, self.config.chain_length * 2 + 1), dynamic_ncols=True):
            start = time.time()

            if self.config.reweight == "reheat":
                cur_temp = t_schedule(fake_step)
                params["temperature"] = jnp.array(cur_temp).reshape(params["temperature"].shape)
            else:
                cur_temp = t_schedule(step)
                params["temperature"] = init_temperature[:, None] * cur_temp

            rng = jax.random.fold_in(rng, step)
            step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))

            if self.config.reweight == "mask":
                params["mask"] = mask_fn(x, params)
                # mask 갱신 후에도 continuous 변수 인덱스는 항상 0 유지
            if 'var_types' in params:
                params['mask'] = jnp.where(params['var_types'] == 3, 0, params['mask'])
            # === ADDED by Claude: apply LP-guided mask (multiplicative; no-op if None) ===
            # Hard mode: zeros out LP-confident integer positions.
            # Soft mode: scales proposal weights by LP fractionality.
            # NOTE: multiplication preserves the existing constraint that
            # continuous positions stay 0 (LP mask is also 0 there).
            if lp_mask_jnp is not None:
                params['mask'] = params['mask'] * lp_mask_jnp
            # === END ADDED ===
            # solve sub-ILP / sub-LP
            # === ADDED by Claude: gate LP by lp_every; lp_every=1 reproduces original 1:1 MCMC:LP schedule ===
            do_lp_this_step = (lp_every <= 1) or ((step - 1) % lp_every == 0)
            if jnp.sum(params['var_types'] == 3) > 0 and do_lp_this_step:
                # === ADDED by Claude: lp_mode switch (sub-LP vs full MILP) ===
                # "sub"  -> classic sub-LP (fix x_int from MCMC, optimise y_cont via linprog)
                # "full" -> full MILP solve via SCIP (integer constraints kept)
                if lp_mode == "full":
                    x_opt = self.solve_full_milp_batch(
                        np.array(x.squeeze(axis=0)), const_m, rhs, lhs, obj_coeffs,
                        int_indices, cont_indices, var_bounds, var_types_np,
                        obj_sign=lp_obj_sign, time_limit=milp_time_limit,
                    )
                else:  # default: classic sub-LP
                    x_opt = self.solve_sub_lp_batch(
                        np.array(x.squeeze(axis=0)), const_m, rhs, lhs, obj_coeffs,
                        int_indices, cont_indices, cont_bounds, obj_sign=lp_obj_sign,
                    )
                if x_opt is not None:
                    x = jnp.expand_dims(jnp.array(x_opt), 0)

                # === END ADDED ===
            # transition
            new_x, state, acc = stp_mixing(
                rng=step_rng,
                x=x,
                model_param=params,
                state=state,
                x_mask=params["mask"],
            )
            new_ll = state["log_prob"] * params["temperature"]
            is_valid = state["is_valid"]

            # parallel tempering
            if (self.config.t_schedule in ["pen_pt", "pen_pt_exp_decay"] and step % self.config.pt_interval == 0):
                new_x, acceptance_ratio, indices_a, indices_b = pt_pen_fn(
                    new_x, new_ll, params['obj_coeffs'], params["temperature"] , jnp.geomspace(self.config.l_min, self.config.l_max, num=self.config.batch_size)[None,:], rng, pt_step
                )
                mean_accept = jnp.mean(acceptance_ratio)
                pairs = [f"{a}-{b}" for a, b in zip(indices_a[0], indices_b[0])]
                pt_step += 1
            elif(
                self.config.t_schedule in ["pt", "pt_exp_decay"]
                and self.config.pt in ["deo", "seo", "reversible"]
                and step % self.config.pt_interval == 0
            ):
                new_x, acceptance_ratio, indices_a, indices_b = pt_temp_fn(
                    new_x, new_ll, params["temperature"], rng, pt_step
                )
                mean_accept = jnp.mean(acceptance_ratio)
                # FIXME: remove [0] later
                pairs = [f"{a}-{b}" for a, b in zip(indices_a[0], indices_b[0])]
                pt_step += 1


            eval_val = jnp.where(is_valid, new_ll, -jnp.inf)
            best_idx = jnp.argmax(eval_val, axis=-1, keepdims=True)
            batch_best_val = jnp.take_along_axis(eval_val, best_idx, axis=-1).squeeze(-1)
            batch_best_x = jnp.take_along_axis(new_x, best_idx[..., None], axis=-2).squeeze(-2)
            is_better = batch_best_val > best_eval_val
            best_eval_val = jnp.where(is_better, batch_best_val, best_eval_val)
            best_samples = jnp.where(is_better[..., None], batch_best_x, best_samples)
            ratio = batch_best_val.reshape(-1) / self.ref_obj  # FIXME: Do we need this?
            best_ratio = jnp.maximum(ratio, best_ratio)  # FIXME: Do we need this?

            new_x.block_until_ready()  # wait for new_x to be ready before calculating the time
            step_time = time.time() - start
            running_time += step_time  # FIXME: Do we need this?
            elapsed_time += step_time

            if self.config_model.mode == "test":
                log_cond = elapsed_time / (
                    self.config_model.max_runtime / self.config.log_every_steps
                ) >= (num_log + 1)
            elif self.config_model.mode == "test_step":
                log_cond = step % (self.config_model.step_limit / self.config.log_every_steps) == 0
            else:  # self.config_model.mode == "val" or "test_step"
                log_cond = step % self.config.log_every_steps == 0

            if self.config_model.mode == "test_step":
                terminate_cond = step >= self.config_model.step_limit
            elif self.config_model.mode == "test":
                terminate_cond = (
                    elapsed_time > self.config_model.max_runtime
                    and num_log > self.config_model.max_runtime // self.config.log_every_steps
                )

            if log_cond or terminate_cond:
                num_log += 1
                penalty_val = penalty_fn(params, new_x)

                for _i, _j in enumerate(self.sample_idx):
                    mean_ll = jnp.mean(new_ll[_i], axis=-1).item()
                    best_ll = jnp.max(new_ll[_i], axis=-1).item()
                    mean_obj = jnp.mean(eval_val[_i], axis=-1).item()
                    best_obj = jnp.max(eval_val[_i], axis=-1).item()
                    mean_penalty = jnp.mean(penalty_val[_i], axis=-1).item()
                    bks_obj = best_eval_val[_i].item()

                    if _i == 0:
                        if len(cur_temp.shape) > 1:  # FIXME: why does cur_temp have length 2?
                            print(
                                f"instance{_j}, temp_min: {cur_temp[0, 0]:.4f}, temp_max: {cur_temp[0, -1]:.4f}",
                                end=" ",
                            )
                        else:
                            print(f"instance{_j}, temp: {cur_temp}", end=" ")
                        print(
                            f"bks_obj: {bks_obj:.4f}, best_ll: {best_ll:.4f}, mean_ll: {mean_ll:.4f}, best_obj: {best_obj:.4f}, mean_obj: {mean_obj:.4f}, mean_penalty: {mean_penalty:.4f}"
                        )
                        if "pt" in self.config.t_schedule:
                            print(
                                f"pt_step: {pt_step}, mean_accept: {mean_accept:.4f}, first_pair_accept: {acceptance_ratio[0][0]:.4f}, last_pair_accept: {acceptance_ratio[0][-1]:.4f}"
                            )

                    wandb.log(
                        {
                            f"instance{_j}/bks_obj": bks_obj,
                            f"instance{_j}/best_ll": best_ll,
                            f"instance{_j}/mean_ll": mean_ll,
                            f"instance{_j}/best_obj": best_obj,
                            f"instance{_j}/mean_obj": mean_obj,
                            f"instance{_j}/mean_penalty": mean_penalty,
                            f"instance{_j}/mean_accept": mean_accept,
                            f"instance{_j}/elapsed_time": elapsed_time,
                            f"instance{_j}/step": step,
                            **{
                                f"instance{_j}/swap_accept_{pairs[k]}": acceptance_ratio[0][k]
                                for k in range(len(pairs))
                            },
                        }
                    )

                if terminate_cond:
                    if self.config_model.mode == "test":
                        print(
                            f"time limit {self.config_model.max_runtime}s reached, early stopping (step: {step})"
                        )
                    elif self.config_model.mode == "test_step":
                        print(
                            f"step limit {self.config_model.step_limit} reached, early stopping (elapsed_time: {elapsed_time:.2f}s)"
                        )
                    # === ADDED by Claude: report LP success rate ===
                    print(
                        f"[LP stats] success={CO_Experiment._lp_success_count}/"
                        f"{CO_Experiment._lp_total_count} "
                        f"({100.0*CO_Experiment._lp_success_count/max(CO_Experiment._lp_total_count,1):.1f}% feasible)"
                    )
                    # === END ADDED ===
                    # === ADDED by Claude (verification): final linprog vs SCIP tally ===
                    if os.environ.get("LP_VERIFY", "0") == "1":
                        vc = CO_Experiment._lp_verify_count
                        vm = CO_Experiment._lp_verify_match
                        print(
                            f"[LP-VERIFY stats] match={vm}/{vc} "
                            f"({100.0*vm/max(vc,1):.1f}% within tol; "
                            f"set LP_VERIFY_TOL to adjust, default 1e-4)"
                        )
                    # === END ADDED ===
                    break

                sample_mask = sample_mask.reshape(best_ratio.shape)
                br = np.array(best_ratio[sample_mask])
                br = jax.device_put(br, jax.devices("cpu")[0])
                chain.append(br)

                if self.config.save_samples or self.config_model.name == "normcut":
                    step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
                    rnew_x = jnp.reshape(
                        new_x,
                        (self.config.num_models, self.config.batch_size) + self.config_model.shape,
                    )
                    chosen_samples = jnp.take_along_axis(
                        rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
                    )
                    chosen_samples = jnp.squeeze(chosen_samples, -2)
                    is_better = ratio > best_ratio
                    best_samples = jnp.where(
                        jnp.expand_dims(is_better, -1), chosen_samples, best_samples
                    )

            if self.config.get_additional_metrics:
                # avg over all models
                acc = jnp.mean(acc)
                acc_ratios.append(acc)
                # hop avg over batch size and num models
                hops.append(get_hop(x, new_x))

            if self.config.reweight == "reheat":
                value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
                append_temp = cur_temp.reshape(self.config.num_models, self.config.batch_size)
                value_diff = jnp.abs(eval_val - old_value)
                trapped_num = jnp.where(
                    jnp.abs(value_diff) < threshold,
                    trapped_num + jnp.ones_like(trapped_num),
                    jnp.zeros_like(trapped_num),
                )
                old_value = eval_val
                reheat_time_array = jnp.where(
                    trapped_num >= trapped_threshold_length,
                    jnp.ones_like(trapped_num),
                    jnp.zeros_like(trapped_num),
                )
                reheat_time = reheat_time + reheat_time_array
                if step >= skip_step:
                    specific_heat = jnp.var(value_chain, axis=0) / (append_temp**2)
                    specific_heat = jnp.where(
                        reheat_time == jnp.zeros_like(reheat_time),
                        specific_heat,
                        jnp.zeros_like(reheat_time),
                    )
                    if print_specific_heat:
                        print("specific_heat", jnp.mean(specific_heat))
                    max_specific_heat = jnp.maximum(specific_heat, max_specific_heat)
                    reheat_step = jnp.where(
                        specific_heat >= max_specific_heat, fake_step, reheat_step
                    )
                    fake_step = jnp.where(
                        trapped_num >= trapped_threshold_length,
                        reheat_step - jnp.ones_like(reheat_step),
                        fake_step,
                    )

            # if self.config.reweight == 'reheat': # burn out mechanism
            #   # we don't calculate specific heat for the mixing phase, since we don't update the critical temperature anymore in case it becomes too small
            #   value_diff = jnp.abs(eval_val - old_value)
            #   trapped_num = jnp.where(jnp.abs(value_diff) < threshold, trapped_num + jnp.ones_like(trapped_num),
            #                           jnp.zeros_like(trapped_num))
            #   old_value = eval_val
            #   reheat_time_array = jnp.where(trapped_num >= trapped_threshold_length, jnp.ones_like(trapped_num),
            #                                 jnp.zeros_like(trapped_num))
            #   reheat_time = reheat_time + reheat_time_array
            #   fake_step = jnp.where(trapped_num >= trapped_threshold_length, reheat_step - jnp.ones_like(reheat_step), fake_step)

            x = new_x
            # === ADDED by Claude: ILS perturbation trigger ===
            # Track best_eval_val improvement; if stagnant beyond window, kick.
            if ils_enabled:
                try:
                    cur_best = float(jnp.max(best_eval_val))
                except Exception:
                    cur_best = ils_last_best
                if cur_best > ils_last_best + 1e-6:
                    ils_last_best = cur_best
                    ils_last_improve_step = step
                elif (step - ils_last_improve_step) > ils_window and len(int_indices) > 0:
                    # Stagnant — perturb each replica independently.
                    x_np = np.array(x.squeeze(axis=0)).copy()      # (batch_size, n_vars)
                    bsz = x_np.shape[0]
                    rng_pert = np.random.default_rng(step)
                    int_idx_np = np.asarray(int_indices, dtype=np.int64)
                    n_perturb = int(min(ils_n_flips, len(int_idx_np)))
                    for r in range(bsz):
                        flip_pos = rng_pert.choice(int_idx_np, size=n_perturb, replace=False)
                        x_np[r, flip_pos] = 1.0 - x_np[r, flip_pos]
                    x = jnp.expand_dims(jnp.array(x_np), 0)
                    ils_last_improve_step = step                  # avoid immediate re-trigger
                    ils_perturb_count += 1
                    if ils_perturb_count <= 20:                   # cap log spam
                        print(f"[ILS] step={step}: perturbed {n_perturb} int positions "
                              f"per replica (stagnant {ils_window} steps; total {ils_perturb_count})")
            # === END ADDED ===
            if self.config.reweight == "reheat":
                fake_step = fake_step + jnp.ones_like(fake_step)

        if not (self.config.save_samples or self.config_model.name == "normcut"):
            best_samples = []

        # saver.save_co_resuts(
        #     chain, best_ratio[sample_mask], running_time, best_samples
        # )
        # saver.save_results(acc_ratios, hops, None, running_time)

        self.all_chain.append(chain)
        self.all_best_ratio.append(best_ratio[sample_mask])
        self.all_current_time.append([running_time])
        self.all_best_samples.append(best_samples)

        all_chain = np.concatenate(self.all_chain, axis=1)
        all_best_ratio = np.concatenate(self.all_best_ratio, axis=0)
        all_current_time = np.concatenate(self.all_current_time, axis=0)
        all_best_samples = np.concatenate(self.all_best_samples, axis=0)

        saver.save_co_results(all_chain, all_best_ratio, all_current_time, all_best_samples)
        saver.save_results(acc_ratios, hops, None, running_time)

    def _initialize_chain_vars(self, bshape):
        t_schedule = self._build_temperature_schedule(self.config)
        sample_mask = self.sample_idx >= 0
        chain = []
        acc_ratios = []
        hops = []
        running_time = 0
        best_ratio = jnp.ones(self.config.num_models, dtype=jnp.float32) * -float("inf")
        init_temperature = jnp.ones(bshape, dtype=jnp.float32)
        dim = math.prod(self.config_model.shape)
        best_samples = jnp.zeros([self.config.num_models, dim])
        return (
            chain,
            acc_ratios,
            hops,
            running_time,
            best_ratio,
            init_temperature,
            t_schedule,
            sample_mask,
            best_samples,
        )

    def mask_per_instance(self, x, params):
        cur_selected_vars = x == 1
        cur_x = x

        constraint_matrix = params["constraint_matrix"]
        constraint_rhs = params["constraint_rhs"]
        constraint_lhs = params["constraint_lhs"]

        cur_constraint_matrix = constraint_matrix
        cur_rhs = constraint_rhs
        cur_lhs = constraint_lhs
        cur_constraint_values = jnp.dot(
            cur_selected_vars.astype(jnp.float32), cur_constraint_matrix.T
        )
        sign = (1 - 2 * cur_x).astype(jnp.float32)

        constraint_changes = jnp.einsum("bv,vc->bvc", sign, cur_constraint_matrix.T)
        updated_values = cur_constraint_values[:, None, :] + constraint_changes

        violates = (updated_values < cur_lhs[None, None, :]) | (
            updated_values > cur_rhs[None, None, :]
        )
        feasible_mask = ~violates.any(axis=2)
        all_masks = jnp.stack(feasible_mask)

        return all_masks[None, ...]
