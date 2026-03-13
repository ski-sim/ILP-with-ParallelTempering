"""Block Locally Balanced Proposal Sampler Class."""

from itertools import product
from discs.common import math_util as math
from discs.common import utils
from discs.samplers import locallybalanced
import jax
from jax import random
import jax.numpy as jnp
import ml_collections
import numpy as np
from scipy.optimize import linprog as scipy_linprog

from jax.scipy import special


def _solve_lp_for_continuous(x_np, A_np, c_np, ub_np, lb_np,
                              cont_idx_np, int_idx_np, negate_np):
    """Solve LP for continuous variables given fixed integer/binary variables.

    For each batch element, fixes integer/binary variable values and solves
    an LP sub-problem to find optimal continuous variable values.

    Args:
        x_np: current solution, shape (batch_size, N)
        A_np: constraint matrix, shape (M, N)
        c_np: objective coefficients, shape (N,)
        ub_np: constraint upper bounds (rhs), shape (M,)
        lb_np: constraint lower bounds (lhs), shape (M,)
        cont_idx_np: continuous variable indices (padded with -1), shape (total_size,)
        int_idx_np: integer/binary variable indices (padded with -1), shape (total_size,)
        negate_np: LP objective sign scalar (1.0 for sc/mvc, -1.0 otherwise)

    Returns:
        new_x: updated solution, shape (batch_size, N), float32
        is_feasible: feasibility flags, shape (batch_size,), float32 (1.0=feasible, 0.0=infeasible)
    """
    x_arr = np.asarray(x_np, dtype=np.float64)
    A_arr = np.asarray(A_np, dtype=np.float64)
    c_arr = np.asarray(c_np, dtype=np.float64)
    ub_arr = np.asarray(ub_np, dtype=np.float64)
    lb_arr = np.asarray(lb_np, dtype=np.float64)
    cont_idx = np.asarray(cont_idx_np, dtype=np.int64)
    int_idx = np.asarray(int_idx_np, dtype=np.int64)
    negate = float(np.asarray(negate_np))

    batch_size, N = x_arr.shape
    new_x = np.copy(x_arr)
    is_feasible = np.zeros(batch_size, dtype=np.float32)

    # Filter out padding indices (-1)
    valid_cont = cont_idx[cont_idx >= 0]
    valid_int = int_idx[int_idx >= 0]

    if len(valid_cont) == 0:
        return new_x.astype(np.float32), is_feasible

    A_cont = A_arr[:, valid_cont]
    c_cont = c_arr[valid_cont]
    A_int = A_arr[:, valid_int] if len(valid_int) > 0 else None

    for b in range(batch_size):
        # Compute A_int @ x_int for the fixed integer/binary variables
        if A_int is not None and len(valid_int) > 0:
            Ax_int = A_int @ x_arr[b, valid_int]
        else:
            Ax_int = np.zeros(A_arr.shape[0])

        # Adjusted bounds: lb - Ax_int <= A_cont @ x_cont <= ub - Ax_int
        adj_ub = ub_arr - Ax_int
        adj_lb = lb_arr - Ax_int

        # Build inequality constraints for linprog: A_ub @ x <= b_ub
        fin_ub = np.isfinite(adj_ub)
        fin_lb = np.isfinite(adj_lb)
        parts_A, parts_b = [], []

        if np.any(fin_ub):
            parts_A.append(A_cont[fin_ub])
            parts_b.append(adj_ub[fin_ub])
        if np.any(fin_lb):
            parts_A.append(-A_cont[fin_lb])
            parts_b.append(-adj_lb[fin_lb])

        A_ub = np.vstack(parts_A) if parts_A else None
        b_ub = np.concatenate(parts_b) if parts_b else None

        # negate * c_cont: linprog minimizes, so negate=-1.0 means maximize c_cont@x
        bounds = [(0, None)] * len(valid_cont)

        try:
            res = scipy_linprog(negate * c_cont, A_ub=A_ub, b_ub=b_ub,
                                bounds=bounds, method='highs')
            if res.success:
                new_x[b, valid_cont] = res.x
                is_feasible[b] = 1.0
        except Exception:
            pass

    return new_x.astype(np.float32), is_feasible

class BlockLBPSampler(locallybalanced.LocallyBalancedSampler):
    """Block Locally Balanced Proposal Sampler.

    Combines block enumeration (from Block Gibbs) with locally balanced
    proposals and Metropolis-Hastings correction (from LBP).

    For a selected block of variables, enumerates all possible configurations,
    computes proposal probabilities via a balancing function g on likelihood
    ratios, and applies MH accept/reject.
    """

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__(config)
        self.config = config
        self.num_flips = config.sampler.get("num_flips", 1)
        self.batch_rows = jnp.expand_dims(jnp.arange(config.experiment.batch_size), axis=1)
        self.block_size = config.sampler.block_size
        if self.block_size != 1:
            self.categories_iter = jnp.array(
                list(product(range(self.num_categories), repeat=self.block_size)) # all combinations of block_size length
            )
        else:
            self.categories_iter = jnp.arange(self.num_categories).reshape([-1,1])
        self.chunk_size = self.config.get("chunk_size", 5000)
        self._category_powers = self.num_categories ** jnp.arange(self.block_size - 1, -1, -1)

    def make_init_state(self, rng):
        """Init sampler state."""
        state = super().make_init_state(rng)
        state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
        state['block_type'] = jnp.zeros(shape=(), dtype=jnp.int32)  # 0=integer, 1=continuous
        return state

    def step(self, model, rng, x, model_param, state, x_mask=None):
        var_types = jnp.array(model_param['var_types'])
        # 인덱스를 추출하는 대신 마스크를 생성
        int_mask = (var_types == 0)
        cont_mask = (var_types == 3)
        num_int = jnp.sum(int_mask)
        num_cont = jnp.sum(cont_mask)
        total_size = var_types.shape[0]
        int_indices = jnp.where(var_types == 0, size=total_size, fill_value=-1)[0]
        cont_indices = jnp.where(var_types == 3, size=total_size, fill_value=-1)[0]
        all_ordered_indices = jnp.argsort(var_types, stable=True)

        block_type = state['block_type']  # persisted in state: 0=integer, 1=continuous
        start_pos = jnp.where(block_type == 0, 0, num_int)
        current_block_size = jnp.where(block_type == 0, num_int, num_cont)
        indices_to_flip = jax.lax.dynamic_slice_in_dim(
            all_ordered_indices, start_pos, 1050
        )
        condition = (block_type == 0)

        _ = x_mask
        rng_new_sample, rng_acceptance = jax.random.split(rng)

        start_index = state['index']

        def true_fun(idx):
            """Integer/binary block: LBP proposal with MH correction."""
            ll_x, y, trajectory, num_calls_forward, is_valid_x = self.proposal(
                model, rng_new_sample, x, model_param, state, int_indices
            )
            ll_x2y = trajectory["ll_x2y"]
            ll_y, ll_y2x, num_calls_backward, is_valid_y = self.ll_y2x(
                model, x, model_param, trajectory, y, int_indices
            )
            log_acc = ll_y + ll_y2x - ll_x - ll_x2y
            new_x, new_state = self.select_sample(
                rng_acceptance,
                num_calls_forward + num_calls_backward,
                log_acc,
                x,
                y,
                ll_x,
                ll_y,
                is_valid_x,
                is_valid_y,
                state,
            )
            acc = jnp.mean(jnp.clip(jnp.exp(log_acc), a_max=1))
            dim = math.prod(self.sample_shape)
            new_state['index'] = (new_state['index'] + self.block_size) % dim
            new_state['block_type'] = jnp.array(1, dtype=jnp.int32)  # next: continuous block
            return new_x, new_state, acc

        def false_fun(idx):
            """Continuous block: solve LP to find optimal continuous variables."""
            A = model_param["constraint_matrix"]  # [M, N]
            c = model_param["obj_coeffs"]          # [N]
            ub = model_param["constraint_rhs"]     # [M]
            lb = model_param["constraint_lhs"]     # [M]
            temp = model_param["temperature"]      # [batch] or [1]

            # LP objective sign:
            #   For sc/mvc: maximize -c@x → linprog minimizes c@x → negate=1.0
            #   Otherwise:  maximize  c@x → linprog minimizes -c@x → negate=-1.0
            negate = jnp.array(
                1.0 if self.config.model.graph_type in ["sc", "mvc"] else -1.0
            )

            result_shapes = (
                jax.ShapeDtypeStruct(x.shape, x.dtype),
                jax.ShapeDtypeStruct((x.shape[0],), jnp.float32),
            )

            new_x_lp, is_feasible_f = jax.pure_callback(
                _solve_lp_for_continuous,
                result_shapes,
                x, A, c, ub, lb, cont_indices, int_indices, negate
            )

            is_feasible = is_feasible_f > 0.5

            # If LP infeasible, keep original x
            new_x = jnp.where(is_feasible[:, None], new_x_lp, x)

            # Compute log_prob and is_valid for the resulting sample
            batch_size = x.shape[0]
            obj_val = new_x @ c  # [batch]
            if self.config.model.graph_type in ["sc", "mvc"]:
                obj_val = -obj_val

            Ax = new_x @ A.T  # [batch, M]
            v = jnp.maximum(0, jnp.maximum(Ax - ub, lb - Ax))  # [batch, M]
            if self.config.model.formulation in ["obj", "lagrangian"]:
                penalty_val = jnp.zeros(batch_size)
                is_valid_new = jnp.sum(v, axis=-1) <= 1e-3
            elif self.config.model.formulation == "max_linear":
                penalty_val = self.config.model.penalty * jnp.sum(v, axis=-1)
                is_valid_new = penalty_val <= 1e-3
            else:  # max_linear_square
                penalty_val = self.config.model.penalty * jnp.sum(jnp.square(v), axis=-1)
                is_valid_new = penalty_val <= 1e-3

            log_prob_new = (obj_val - penalty_val) / temp

            new_state = utils.copy_pytree(state)
            new_state['index'] = jnp.zeros((), dtype=jnp.int32)
            new_state['log_prob'] = log_prob_new
            new_state['is_valid'] = is_valid_new
            new_state['num_ll_calls'] = state['num_ll_calls']
            new_state['steps'] = state['steps'] + 1
            new_state['block_type'] = jnp.array(0, dtype=jnp.int32)  # next: integer block

            acc = jnp.mean(is_feasible.astype(jnp.float32))

            return new_x, new_state, acc

        # jax.lax.cond: block_type==0 → integer LBP, block_type==1 → continuous LP
        new_x, new_state, acc = jax.lax.cond(condition, true_fun, false_fun, indices_to_flip)

        return new_x, new_state, acc

    def neighborhood_fn(self, params, x, indices_to_flip):
        A = params["constraint_matrix"]  # [M, N]
        c = params["obj_coeffs"]  # [N]
        ub = params["constraint_rhs"]  # [M]
        lb = params["constraint_lhs"]  # [M]
        temp = params["temperature"]  # [batch] or [1]

        candidates_size = len(self.categories_iter)
        batch_size, N = x.shape
        M = A.shape[0]
        m_chunk = min(self.chunk_size, M)

        Ax = x @ A.T  # [batch, M]
        obj_x = x @ c  # [batch]
        if self.config.model.graph_type in ["sc", "mvc"]:
            obj_x = -obj_x

        v_curr = jnp.maximum(0, jnp.maximum(Ax - ub, lb - Ax))  # [batch, M]
        if self.config.model.formulation == "obj" or self.config.model.formulation == "lagrangian":
            penalty_x = jnp.zeros(batch_size)
            is_valid_x = jnp.sum(v_curr, axis=-1) <= 1e-3  # [batch]  # 1e-3 is a small threshold
        else:
            if self.config.model.formulation == "max_linear":
                penalty_x = self.config.model.penalty * jnp.sum(v_curr, axis=-1)
            elif self.config.model.formulation == "max_linear_square":
                penalty_x = self.config.model.penalty * jnp.sum(jnp.square(v_curr), axis=-1)
            is_valid_x = penalty_x <= 1e-3  # [batch]  # 1e-3 is a small threshold

        ll_x = (obj_x - penalty_x) / temp  # [batch]

        c_sub = c[indices_to_flip]
        delta_x = 1 - 2 * x[:,indices_to_flip]
        delta_obj = c_sub[None, :] * delta_x
        if self.config.model.graph_type in ["sc", "mvc"]:
            delta_obj = -delta_obj

        if self.config.model.formulation == "obj" or self.config.model.formulation == "lagrangian":
            penalty_new = jnp.zeros((batch_size, N))
        else:
            # Scan over M (constraints)
            excess_upper = Ax - ub[None, :]  # [batch, M]
            excess_lower = lb[None, :] - Ax  # [batch, M]
            pad_m = (-M) % m_chunk
            if pad_m > 0:
                A_padded = jnp.pad(A, ((0, pad_m), (0, 0)))
                Ax_padded = jnp.pad(Ax, ((0, 0), (0, pad_m)))
                excess_upper = jnp.pad(excess_upper, ((0, 0), (0, pad_m)), constant_values=-jnp.inf)
                excess_lower = jnp.pad(excess_lower, ((0, 0), (0, pad_m)), constant_values=-jnp.inf)
            else:
                A_padded = A

            num_m_chunks = (M + pad_m) // m_chunk
            A_scan = A_padded.reshape(num_m_chunks, m_chunk, N)  # [num_m_chunks, m_chunk, N]
            eu_scan = excess_upper.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
            el_scan = excess_lower.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
            # [num_m_chunks, batch, m_chunk] each
            def scan_body(penalty_acc, inputs):
                A_chunk, eu_chunk, el_chunk = inputs
                # A_chunk: [m_chunk, N], eu/el_chunk: [batch, m_chunk]

                A_sub = A_chunk[:, indices_to_flip]  # [m_chunk, block_size]
                shift = A_sub[None, :, :] * delta_x[:, None, :]  # [batch, m_chunk, N]

                # penalty_current = x[:, indices_to_flip] @ A_sub.T  # [batch, m_chunk]
                # penalty_new = self.categories_iter @ A_sub.T  # [candidates, m_chunk]

                # shift = penalty_new.transpose(1,0)[None,:,:] - penalty_current[:,:,None] # [batch, m_chunk]
                v_new = jnp.maximum(
                        0, jnp.maximum(eu_chunk[:, :, None] + shift, el_chunk[:, :, None] - shift)
                    )

                if self.config.model.formulation == "max_linear":
                    return penalty_acc + jnp.sum(v_new, axis=1), None
                else:  # formulation == "max_linear_square"
                    return penalty_acc + jnp.sum(jnp.square(v_new), axis=1), None
            penalty_new, _ = jax.lax.scan(
                scan_body,  jnp.zeros((batch_size, len(indices_to_flip))),(A_scan, eu_scan, el_scan)
            )
            penalty_new = self.config.model.penalty * penalty_new
        
        ll_new = (obj_x[:, None] + delta_obj - penalty_new) / temp[:, None]
        logratios = ll_new - ll_x[:, None]

        return ll_x, logratios, 1, self.neighborhood_fn, is_valid_x

    def get_local_dist(self, model, x, model_param, indices_to_flip):
        # Lazy initialization: create neighborhood_fn only once
        ll_x, logratio, num_calls, _, is_valid_x = self.neighborhood_fn(model_param, x, indices_to_flip)

        logits = self.apply_weight_function_logscale(logratio)
        if self.num_categories != 2:
            logits = logits * (1 - x) + x * -1e9
        log_prob = jax.nn.log_softmax(logits, -1)
        return ll_x, log_prob, num_calls, is_valid_x

    def proposal(self, model, rng, x, model_param, state, indices_to_flip):
        ll_x, log_prob, num_calls, is_valid_x = self.get_local_dist(model, x, model_param, indices_to_flip)

        if self.num_categories > 2:
            log_prob_all = jnp.reshape(log_prob, [log_prob.shape[0], -1, self.num_categories])
            log_prob = special.logsumexp(log_prob_all, axis=-1)
            log_prob_all = jax.nn.log_softmax(log_prob_all, axis=-1)
            x = jnp.reshape(x, log_prob_all.shape)

        selected_idx, ll_selected = math.multinomial(
            rng, log_prob, self.num_flips, replacement=False, return_ll=True, is_nsample_const=True
        )
        if self.num_categories > 2:
            val_logprob = log_prob_all[self.batch_rows, selected_idx]
            rng, _ = jax.random.split(rng)
            new_val = jax.random.categorical(rng, val_logprob)
            new_val = jax.nn.one_hot(new_val, self.num_categories)
        else:
            new_val = 1 - x[self.batch_rows, selected_idx + indices_to_flip[0]]
        y = x.at[self.batch_rows, selected_idx + indices_to_flip[0]].set(new_val)

        trajectory = {
            "ll_x2y": jnp.sum(ll_selected, axis=-1),
            "selected_idx": selected_idx,
        }
        return ll_x,  y, trajectory, num_calls, is_valid_x

    def ll_y2x(self, model, x, model_param, forward_trajectory, y, indices_to_flip):
        ll_y, log_prob, num_calls, is_valid_y = self.get_local_dist(model, y, model_param, indices_to_flip)

        if self.num_categories > 2:
            log_prob_all = jnp.reshape(log_prob, [log_prob.shape[0], -1, self.num_categories])
            log_prob = special.logsumexp(log_prob_all, axis=-1)
            log_prob_all = jax.nn.log_softmax(log_prob_all, axis=-1)
        backwd_ll = jnp.take_along_axis(log_prob, forward_trajectory["selected_idx"], -1) # (batch,1)
        
        ll_y2x_traj = math.noreplacement_sampling_renormalize(backwd_ll)
        if self.num_categories > 2:
            val_logprob = log_prob_all[
                jnp.expand_dims(jnp.arange(x.shape[0]), axis=1), forward_trajectory["selected_idx"]
            ]
            x = jnp.reshape(x, [x.shape[0], -1, self.num_categories])
            orig_val = x[
                jnp.expand_dims(jnp.arange(x.shape[0]), axis=1), forward_trajectory["selected_idx"]
            ]
            ll_val = jnp.sum(orig_val * val_logprob, axis=-1)

        ll_y2x = jnp.sum(ll_y2x_traj, axis=-1)
        if self.num_categories > 2:
            ll_y2x = ll_y2x + jnp.sum(ll_val, axis=-1)

        return ll_y, ll_y2x, num_calls, is_valid_y

    def select_sample(
        self,
        rng,
        num_calls,
        log_acc,
        current_sample,
        new_sample,
        current_ll,
        new_ll,
        is_valid_x,
        is_valid_y,
        sampler_state,
    ):
        y, new_state = super().select_sample(
            rng,
            log_acc,
            current_sample,
            new_sample,
            current_ll,
            new_ll,
            is_valid_x,
            is_valid_y,
            sampler_state,
        )
        new_state["num_ll_calls"] += num_calls
        return y, new_state
def build_sampler(config):
  return BlockLBPSampler(config)
