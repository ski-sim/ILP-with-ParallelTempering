"""Block Locally Balanced Proposal Sampler Class."""

from itertools import product
from PT_ILP.common import math_util as math
from PT_ILP.common import utils
from PT_ILP.samplers import locallybalanced
import jax
from jax import random
import jax.numpy as jnp
import ml_collections

from jax.scipy import special

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
        return state

    def step(self, model, rng, x, model_param, state, x_mask=None):
        _ = x_mask
        rng_new_sample, rng_acceptance = jax.random.split(rng)

        start_index = state['index'] # 추가
        indices_to_flip = jnp.arange(self.block_size) + start_index # 추가

        ll_x, ll_y, ll_y2x, y, trajectory, num_calls_forward, is_valid_x = self.proposal(
                model, rng_new_sample, x, model_param, state, indices_to_flip
            )
        ll_x2y = trajectory["ll_x2y"]

        num_calls_backward = num_calls_forward 
        is_valid_y = is_valid_x


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
        if self.config.model.instance_name in ["sc", "mvc"]:
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
        obj_new = self.categories_iter @ c_sub  # [candidates]
        obj_current = x[:, indices_to_flip] @ c_sub  # [batch]
        delta_obj = obj_new[None, :] - obj_current[:, None]  # [batch, candidates]

        if self.config.model.instance_name in ["sc", "mvc"]:
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
            Ax_scan = Ax_padded.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
            eu_scan = excess_upper.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
            el_scan = excess_lower.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
            # [num_m_chunks, batch, m_chunk] each
            def scan_body(penalty_acc, inputs):
                A_chunk, Ax_chunk, eu_chunk, el_chunk = inputs
                # A_chunk: [m_chunk, N], eu/el_chunk: [batch, m_chunk]
                # self.categories_iter: [candidates, block_size]

                A_sub = A_chunk[:, indices_to_flip]  # [m_chunk, block_size]
                penalty_current = x[:, indices_to_flip] @ A_sub.T  # [batch, m_chunk]
                penalty_new = self.categories_iter @ A_sub.T  # [candidates, m_chunk]

                shift = penalty_new.transpose(1,0)[None,:,:] - penalty_current[:,:,None] # [batch, m_chunk]
                v_new = jnp.maximum(
                        0, jnp.maximum(eu_chunk[:, :, None] + shift, el_chunk[:, :, None] - shift)
                    )

                if self.config.model.formulation == "max_linear":
                    return penalty_acc + jnp.sum(v_new, axis=1), None
                else:  # formulation == "max_linear_square"
                    return penalty_acc + jnp.sum(jnp.square(v_new), axis=1), None
            penalty_new, _ = jax.lax.scan(
                scan_body,  jnp.zeros((batch_size, candidates_size)),(A_scan, Ax_scan, eu_scan, el_scan)
            )
            penalty_new = self.config.model.penalty * penalty_new
        
        ll_new = (obj_x[:, None] + delta_obj - penalty_new) / temp[:, None]
        logratios = ll_new - ll_x[:, None]

        return ll_x, logratios, 1, self.neighborhood_fn, is_valid_x, ll_new

    def get_local_dist(self, model, x, model_param, indices_to_flip):
        # Lazy initialization: create neighborhood_fn only once
        ll_x, logratio, num_calls, _, is_valid_x, ll_new = self.neighborhood_fn(model_param, x, indices_to_flip)

        logits = self.apply_weight_function_logscale(logratio)
        if self.num_categories != 2:
            logits = logits * (1 - x) + x * -1e9
        log_prob = jax.nn.log_softmax(logits, -1)
        return ll_x, log_prob, num_calls, is_valid_x, ll_new

    def proposal(self, model, rng, x, model_param, state, indices_to_flip):
        ll_x, log_prob, num_calls, is_valid_x, ll_new = self.get_local_dist(model, x, model_param, indices_to_flip)

        if self.num_categories > 2:
            log_prob_all = jnp.reshape(log_prob, [log_prob.shape[0], -1, self.num_categories])
            log_prob = special.logsumexp(log_prob_all, axis=-1)
            log_prob_all = jax.nn.log_softmax(log_prob_all, axis=-1)
            x = jnp.reshape(x, log_prob_all.shape)

        selected_idx, ll_selected = math.multinomial(
            rng, log_prob, self.num_flips, replacement=False, return_ll=True, is_nsample_const=True
        )

        # if self.num_categories > 2:
        #     val_logprob = log_prob_all[self.batch_rows, selected_idx]
        #     rng, _ = jax.random.split(rng)
        #     new_val = jax.random.categorical(rng, val_logprob)
        #     new_val = jax.nn.one_hot(new_val, self.num_categories)
        # else:
        #     new_val = 1 - x[self.batch_rows, selected_idx]

        y = x.at[self.batch_rows, indices_to_flip].set( self.categories_iter[selected_idx].squeeze(axis=1))
        
        
        ll_y = jnp.take_along_axis(ll_new, selected_idx, axis=-1)[:, 0] # calculate ll_y

        x_flipped = x[:, indices_to_flip]
        matched_indices = jnp.sum(x_flipped * self._category_powers, axis=-1).astype(jnp.int32)
        backward_indices = matched_indices[:, None]
        backwd_ll = jnp.take_along_axis(log_prob,backward_indices, -1)
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


        trajectory = {
            "ll_x2y": jnp.sum(ll_selected, axis=-1),
            "selected_idx": selected_idx,
        }
        return ll_x, ll_y, ll_y2x, y, trajectory, num_calls, is_valid_x

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
