"""Path auxiliary sampler."""

from discs.common import math_util as math
from discs.samplers import locallybalanced
import jax
import jax.numpy as jnp
from jax.scipy import special
import ml_collections
import pdb

class PathAuxiliarySampler(locallybalanced.LocallyBalancedSampler):
  """Path auxiliary sampler."""

  def step(self, model, rng, x, model_param, state, x_mask=None):
    if self.num_categories != 2:
      x = jax.nn.one_hot(x, self.num_categories, dtype=jnp.float32)
      if x_mask is not None:
        x_mask = jnp.expand_dims(x_mask, axis=-1)
    rng_new_sample, rng_acceptance = jax.random.split(rng)
    if self.reweight == 'mask':

      reweight = self.mask_per_instance(x, model_param)
    else:
      reweight = jnp.ones_like(x, dtype=jnp.bool_)
    ll_x, y, trajectory, num_calls_forward = self.proposal(
        model, rng_new_sample, x, model_param, state, x_mask, reweight)
    ll_x2y = trajectory['ll_x2y']
    if self.reweight == 'mask':
      reweight = self.mask_per_instance(y, model_param)
    ll_y, ll_y2x, num_calls_backward = self.ll_y2x(
        model, x, model_param, trajectory, y, x_mask, reweight)

    log_acc = ll_y + ll_y2x - ll_x - ll_x2y
    new_x, new_state = self.select_sample(
        rng_acceptance, num_calls_forward + num_calls_backward,
        log_acc, x, y, state)

    acc = jnp.mean(jnp.clip(jnp.exp(log_acc), a_max=1))
    return new_x, new_state, acc

  def mask_per_instance(self, x, params):
    cur_selected_vars = (x == 1)
    cur_x = x
    if 'constraint_matrix' not in params:
        return jnp.ones_like(x, dtype=jnp.bool_)
    # 제약조건 정보를 가져올 때부터 float32로 타입을 지정합니다.
    constraint_matrix = params['constraint_matrix'].astype(jnp.float32)
    constraint_rhs = params['constraint_rhs'].astype(jnp.float32)
    constraint_lhs = params['constraint_lhs'].astype(jnp.float32)

    cur_constraint_matrix = constraint_matrix
    cur_rhs = constraint_rhs
    cur_lhs = constraint_lhs
    cur_constraint_values = jnp.dot(cur_selected_vars.astype(jnp.float32), cur_constraint_matrix.T)
    sign = (1 - 2 * cur_x).astype(jnp.float32)

    final_feasible_mask = jnp.zeros_like(cur_selected_vars, dtype=jnp.bool_)
    constraint_changes = jnp.einsum('bv,vc->bvc', sign, cur_constraint_matrix.T)
    updated_values = cur_constraint_values[:, None, :] + constraint_changes
        
    # 제약조건 위반 여부 확인
    violates = (updated_values < cur_lhs[None, None, :]) | (updated_values > cur_rhs[None, None, :]) # 위반 -> True
    feasible_mask = ~violates.any(axis=2) # 하나라도 위반하면 False, 만족하면 True
    final_feasible_mask = final_feasible_mask.at[:, :].set(feasible_mask)
    all_masks = jnp.stack(final_feasible_mask)

    return all_masks[None, ...]

  def select_sample(self, rng, num_calls, log_acc,
                    current_sample, new_sample, sampler_state):
    y, new_state = super().select_sample(
        rng, log_acc, current_sample, new_sample, sampler_state)
    new_state['num_ll_calls'] += num_calls
    return y, new_state


class PAFSNoReplacement(PathAuxiliarySampler):
  """Path auxiliary sampler with no replacement proposal sampling."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.reweight = config.experiment.get('reweight', 'None')
    self.adaptive = config.sampler.get('adaptive', False)
    if self.adaptive:
      self.target_acceptance_rate = config.sampler.target_acceptance_rate
    self.num_flips = config.sampler.get('num_flips', 1)
    self.approx_with_grad = config.sampler.get('approx_with_grad', True)

  def select_sample(self, rng, num_calls, log_acc,
                    current_sample, new_sample, sampler_state):
    y, new_state = super().select_sample(
        rng, num_calls, log_acc, current_sample, new_sample, sampler_state)
    if self.adaptive:
      acc = jnp.mean(jnp.exp(jnp.clip(log_acc, a_max=0.0)))
      r = sampler_state['radius'] + acc - self.target_acceptance_rate
      new_state['radius'] = jnp.clip(
          r, a_min=1, a_max=math.prod(self.sample_shape))
    return y, new_state

  def make_init_state(self, rng):
    state = super().make_init_state(rng)
    state['radius'] = jnp.ones(shape=(), dtype=jnp.float32) * self.num_flips
    return state

  def get_local_dist(self, model, x, model_param, x_mask, reweight):
    if self.approx_with_grad:
      ll_x, grad_x = model.get_value_and_grad(model_param, x)
      if self.num_categories != 2:
        logratio = grad_x - jnp.sum(grad_x * x, axis=-1, keepdims=True)
      else:
        logratio = (1 - 2 * x) * grad_x
      num_calls = 2
    else:
      ll_x, logratio, num_calls, _ = model.logratio_in_neighborhood(
          model_param, x)
      assert logratio.shape == x.shape
    logits = self.apply_weight_function_logscale(logratio)
    if x_mask is not None:
      logits = logits * x_mask + -1e9 * (1 - x_mask)
    if self.num_categories != 2:
      logits = logits * (1 - x) + x * -1e9
    # masked_logits = logits == -1e9 
    # reweighted_logits = logits * reweight + -1e9 * (1 - reweight)
    # logits = jnp.where(masked_logits, logits, reweighted_logits)
    log_prob = jax.nn.log_softmax(jnp.reshape(logits, (x.shape[0], -1)), -1)
    return ll_x, log_prob, num_calls

  def proposal(self, model, rng, x, model_param, state, x_mask, reweight):
    ll_x, log_prob, num_calls = self.get_local_dist(
        model, x, model_param, x_mask, reweight)
    if self.adaptive:
      num_samples = jnp.clip(jnp.round(state['radius']).astype(jnp.int32),
                             a_min=1, a_max=math.prod(self.sample_shape))
    else:
      num_samples = self.num_flips
    x_shape = x.shape
    x = jnp.reshape(x, (x.shape[0], -1))
    if self.num_categories > 2:
      log_prob_all = jnp.reshape(log_prob,
                                 [log_prob.shape[0], -1, self.num_categories])
      log_prob = special.logsumexp(log_prob_all, axis=-1)
      log_prob_all = jax.nn.log_softmax(log_prob_all, axis=-1)
      x = jnp.reshape(x, log_prob_all.shape)
    selected_idx, ll_selected = math.multinomial(
        rng, log_prob, num_samples, replacement=False, return_ll=True,
        is_nsample_const=not self.adaptive, need_ordering_info=True)
    if self.adaptive:
      mask = selected_idx['selected_mask']
      if self.num_categories > 2:
        rng, _ = jax.random.split(rng)
        new_val = jax.random.categorical(rng, log_prob_all)
        new_val = jax.nn.one_hot(new_val, self.num_categories)
        ll_val = jnp.sum(new_val * log_prob_all, axis=-1) * mask
        mask = jnp.expand_dims(mask, axis=-1)
        ll_selected = ll_selected + ll_val
      else:
        new_val = 1 - x
      y = x * (1 - mask) + mask * new_val
    else:
      rows = jnp.expand_dims(jnp.arange(x.shape[0]), axis=1)
      if self.num_categories > 2:
        val_logprob = log_prob_all[rows, selected_idx]
        rng, _ = jax.random.split(rng)
        new_val = jax.random.categorical(rng, val_logprob)
        new_val = jax.nn.one_hot(new_val, self.num_categories)
      else:
        new_val = 1 - x[rows, selected_idx]
      y = x.at[rows, selected_idx].set(new_val)
    y = jnp.reshape(y, x_shape)
    trajectory = {
        'll_x2y': jnp.sum(ll_selected, axis=-1),
        'selected_idx': selected_idx,
    }
    return ll_x, y, trajectory, num_calls

  def ll_y2x(self, model, x, model_param, forward_trajectory, y, x_mask, reweight):
    ll_y, log_prob, num_calls = self.get_local_dist(
        model, y, model_param, x_mask, reweight)
    if self.num_categories > 2:
      log_prob_all = jnp.reshape(log_prob,
                                 [log_prob.shape[0], -1, self.num_categories])
      log_prob = special.logsumexp(log_prob_all, axis=-1)
      log_prob_all = jax.nn.log_softmax(log_prob_all, axis=-1)
    if self.adaptive:
      selected_mask = forward_trajectory['selected_idx']['selected_mask']
      order_info = forward_trajectory['selected_idx']['perturbed_ll']
      if self.num_categories > 2:
        x = jnp.reshape(x, [x.shape[0], -1, self.num_categories])
        ll_val = jnp.sum(x * log_prob_all, axis=-1) * selected_mask
      backwd_idx = jnp.argsort(order_info)
      log_prob = jnp.where(selected_mask, log_prob, -1e18)
      backwd_ll = jnp.take_along_axis(log_prob, backwd_idx, -1)
      backwd_mask = jnp.take_along_axis(selected_mask, backwd_idx, -1)
      ll_backwd = math.noreplacement_sampling_renormalize(backwd_ll)
      ll_y2x = jnp.sum(jnp.where(backwd_mask, ll_backwd, 0.0), axis=-1)
    else:
      reverse_idx_traj = jnp.flip(forward_trajectory['selected_idx'], axis=-1)
      backwd_ll = jnp.take_along_axis(log_prob, reverse_idx_traj, -1)
      ll_y2x_traj = math.noreplacement_sampling_renormalize(backwd_ll)
      if self.num_categories > 2:
        val_logprob = log_prob_all[
            jnp.expand_dims(jnp.arange(x.shape[0]), axis=1), reverse_idx_traj]
        x = jnp.reshape(x, [x.shape[0], -1, self.num_categories])
        orig_val = x[jnp.expand_dims(jnp.arange(x.shape[0]), axis=1),
                     reverse_idx_traj]
        ll_val = jnp.sum(orig_val * val_logprob, axis=-1)
      ll_y2x = jnp.sum(ll_y2x_traj, axis=-1)
    if self.num_categories > 2:
      ll_y2x = ll_y2x + jnp.sum(ll_val, axis=-1)
    return ll_y, ll_y2x, num_calls


def build_sampler(config):
  if config.sampler.use_fast_path:
    return PAFSNoReplacement(config)
  else:
    raise NotImplementedError
