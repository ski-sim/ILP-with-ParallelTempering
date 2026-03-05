"""Block Locally Balanced Proposal Sampler Class."""

from itertools import product
from discs.common import math_util as math
from discs.common import utils
from discs.samplers import locallybalanced
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
    self.chunk_size = self.config.get("chunk_size", 1000)
  def make_init_state(self, rng):
    """Init sampler state."""
    state = super().make_init_state(rng)
    state['index'] = jnp.zeros(shape=(), dtype=jnp.int32)
    return state

  def update_sampler_state(self, sampler_state):
    sampler_state = super().update_sampler_state(sampler_state)
    dim = math.prod(self.sample_shape)
    sampler_state['index'] = (sampler_state['index'] + self.block_size) % dim
    sampler_state['num_ll_calls'] += self.num_categories**self.block_size
    return sampler_state

  def step(self, model, rng, x, model_param, state, x_mask=None):
    _ = x_mask
    rng_new_sample, rng_acceptance = jax.random.split(rng)

    start_index = state['index']
    indices_to_flip = jnp.arange(self.block_size) + start_index

    ll_x, y, trajectory, num_calls_forward, is_valid_x = self.proposal(
            model, rng_new_sample, x, model_param, state, indices_to_flip
        )
    ll_x2y = trajectory["ll_x2y"]
    ll_y, ll_y2x, num_calls_backward, is_valid_y = self.ll_y2x(
        model, x, model_param, trajectory, y
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
    return new_x, new_state, acc






    batch_size = x.shape[0]
    batch_arange = jnp.arange(batch_size)
    
    def _get_ll_at_block(model, indices_to_flip, x, model_param):
        """Compute log-likelihoods for all block configurations.

        Returns:
          ll_x: shape (batch_size,) — current log-likelihood
          ll_all: shape (num_configs, batch_size)
        """
        def fn_ll(_, i):
            y_flatten = x.reshape(x.shape[0], -1)
            y_flatten = y_flatten.at[:, indices_to_flip].set(
                self.categories_iter[i]
            )
            y = y_flatten.reshape((-1,) + self.sample_shape)
            ll, _ = model.forward(model_param, y)  # unpack (ll, isvalid)
            return None, ll

        _, ll_all = jax.lax.scan(
            fn_ll, None, jnp.arange(0, len(self.categories_iter))
        )
        ll_x, _ = model.forward(model_param, x)
        return ll_x, ll_all

    ll_x, ll_all = _get_ll_at_block(
        model, indices_to_flip, x, model_param
    )  # ll_all: (num_configs, batch_size), ll_x: (batch_size,)

    # --- Forward proposal: q(y|x) ∝ g(π(y)/π(x)) ---
    logratio_forward = ll_all - ll_x[None, :]  # (num_configs, batch_size)
    logits_forward = self.apply_weight_function_logscale(logratio_forward)
    log_prob_forward = jax.nn.log_softmax(logits_forward, axis=0)
    
    selected_idx = jax.random.categorical(
        rng_proposal, logits_forward.T, axis=-1
    )  # (batch_size,)


    ll_selected = log_prob_forward[selected_idx, batch_arange]
    y = x.at[:, indices_to_flip].set(self.categories_iter[selected_idx])
    ll_x2y = jnp.sum(ll_selected, axis=-1)


    # Proposed log-likelihood
    ll_y = ll_all[selected_idx, batch_arange]  # (batch_size,)

    # --- Backward proposal: q(x|y) ∝ g(π(x)/π(y)) ---
    logratio_backward = ll_all - ll_y[None, :]  # (num_configs, batch_size)
    logits_backward = self.apply_weight_function_logscale(logratio_backward)
    log_prob_backward = jax.nn.log_softmax(logits_backward, axis=0)
    

    ll_selected = log_prob_backward[selected_idx, batch_arange]
    y = x.at[:, indices_to_flip].set(self.categories_iter[selected_idx])
    ll_y2x = jnp.sum(ll_selected, axis=-1)
    # # --- MH acceptance ratio ---
    log_acc = ll_y - ll_x + ll_x2y - ll_y2x
    y, acc = math.mh_step(rng_accept, log_acc, x, y)
    if self.num_categories == 2:
        y = y.astype(jnp.int32)
    else:
        y = jnp.argmax(y, axis=-1)
        
    acc = jnp.mean(jnp.clip(jnp.exp(log_acc), a_max=1))
    new_state = self.update_sampler_state(state)
    new_state["log_prob"] = jnp.where(acc, ll_y, ll_x)
    new_state["is_valid"] = False
    return y, new_state, acc
  
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

      obj_new = jnp.sum(c[None, indices_to_flip] * self.categories_iter, axis=-1)
      obj_current = jnp.sum(c[None, indices_to_flip] * x[:, indices_to_flip], axis=-1)
      delta_obj = obj_new[None,:] - obj_current[:,None] # (batch_size,candidates))
      
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
          Ax_scan = Ax_padded.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
          eu_scan = excess_upper.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
          el_scan = excess_lower.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
          # [num_m_chunks, batch, m_chunk] each
          def scan_body(penalty_acc, inputs):
            A_chunk, Ax_chunk, eu_chunk, el_chunk = inputs
            # A_chunk: [m_chunk, N], eu/el_chunk: [batch, m_chunk]
            # self.categories_iter: [candidates, block_size]

            penalty_current = jnp.sum(A_chunk[None, :, indices_to_flip] * x[:, None,indices_to_flip], axis=-1)

            shift = jnp.sum(A_chunk[None, :, indices_to_flip] * self.categories_iter[:,None,:], axis=-1) # [candidates, m_chunk]
            
            penalty_new = Ax_chunk[:,:,None] - penalty_current[:,:,None] + shift.transpose(1,0)[None,:,:]
            v_new = jnp.maximum(
                    0, jnp.maximum(eu_chunk[:, :, None] + penalty_new, el_chunk[:, :, None] - penalty_new)
                )

            if self.config.model.formulation == "max_linear":
                return penalty_acc + jnp.sum(v_new, axis=1), None
            else:  # formulation == "max_linear_square"
                return penalty_acc + jnp.sum(jnp.square(v_new), axis=1), None
          penalty_new, _ = jax.lax.scan(
              scan_body,  jnp.zeros((batch_size, candidates_size)),(A_scan, Ax_scan, eu_scan, el_scan)
          )
          penalty_new = self.config.model.penalty * penalty_new
      import pdb;     pdb.set_trace()
      ll_new = (obj_x[:, None] + delta_obj - penalty_new) / temp[:, None]
      logratios = ll_new - ll_x[:, None]

      return ll_x, logratios, 1, self.get_neighbor_fn, is_valid_x
 
  def get_local_dist(self, model, x, model_param, indices_to_flip):
        # Lazy initialization: create neighborhood_fn only once
        if not hasattr(self, "neighborhood_fn"):
            self.neighborhood_fn = model.logratio_in_neighborhood
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
          new_val = 1 - x[self.batch_rows, selected_idx]
      y = x.at[self.batch_rows, selected_idx].set(new_val)

      trajectory = {
          "ll_x2y": jnp.sum(ll_selected, axis=-1),
          "selected_idx": selected_idx,
      }
      return ll_x, y, trajectory, num_calls, is_valid_x
  


def build_sampler(config):
  return BlockLBPSampler(config)
