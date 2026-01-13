"""Max Independent Set model."""

from discs.models import comb_ebm
import jax.numpy as jnp
import ml_collections
import jax

class MIS(comb_ebm.BinaryNodeCombEBM):
  """Max Independent Set model."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.config = config.model
    self.max_num_nodes = self.config.max_num_nodes
    self.penalty_coeff = self.config.get('penalty', 2.0)
    self.formulation = self.config.get('formulation', 'quadratic')

  def make_init_params(self, rng):
    try:
      data_list = next(self.datagen)
    except:
      return None
    return data_list

  def penalty(self, params, x):
    x = x * params['mask'] # 유효한 노드인지 아닌지. 
    edge_from = params['edge_from']
    edge_to = params['edge_to']
    edge_mask = params['edge_mask'] # edge index가 유효한지 아닌지 판단단

    gather2src = x[:, edge_from]
    gather2dst = x[:, edge_to]
    if self.formulation == 'quadratic':
      violation = gather2src * gather2dst * edge_mask # 1 * 1 이면 violation
    elif self.formulation == 'linear':
      violation = (gather2src + gather2dst - 1) * edge_mask
    elif self.formulation == 'max_linear':
      violation = jnp.maximum(0, gather2src + gather2dst - 1) * edge_mask
    elif self.formulation == 'indicator':
      constraint_satisfied = (gather2src + gather2dst - 1) * edge_mask <= 0
      violation = ~constraint_satisfied
    elif self.formulation == 'objective':
      violation = jnp.zeros((x.shape[0], edge_mask.shape[0]), dtype=jnp.int32)
    penalty = self.penalty_coeff * jnp.sum(violation, axis=1)
    return penalty

  def objective(self, params, x):
    _ = params
    x = x * params['mask']
    return jnp.sum(x, axis=1)

  def logratio_in_neighborhood(self, params, x):
    edge_from = params['bidir_edge_from']
    edge_to = params['bidir_edge_to']

    gather2dst = x[:, edge_to]
    diff_penalty = self.penalty_coeff * gather2dst
    diff_arr = jnp.zeros(x.shape, dtype=diff_penalty.dtype)
    diff_arr = diff_arr.at[jnp.expand_dims(jnp.arange(x.shape[0]), axis=1),
                           edge_from].add(diff_penalty)

    sign = (1 - x * 2) * params['mask']
    logratio = (sign - sign * diff_arr) / params['temperature']
    logratio = logratio * params['mask'] + -1e9 * (1 - params['mask'])
    return logratio, 1, self.get_neighbor_fn


def build_model(config):
  return MIS(config)
