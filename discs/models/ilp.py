"""Max Independent Set model."""

from discs.models import comb_ebm
import jax.numpy as jnp
import ml_collections


class ILP(comb_ebm.BinaryNodeCombEBM):
  """Max Independent Set model."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.config = config.model
    self.max_num_nodes = self.config.max_num_nodes
    self.penalty_coeff = self.config.get('penalty', 2.0)
    self.formulation = self.config.get('formulation', 'max_linear')
    self.chunk_size = self.config.get('chunk_size', 2000)

  def make_init_params(self, rng):
    try:
      data_list = next(self.datagen)
    except:
      return None
    return data_list

  def penalty(self, params, x):
    batch_size = x.shape[0]
    num_vars = x.shape[-1]
    num_constraints = params['constraint_matrix'].shape[0]
    if num_vars > self.chunk_size:
      Ax = jnp.zeros((num_constraints, batch_size), dtype=jnp.float32)
      for i in range(0, num_vars, self.chunk_size):
        x_chunk = x[..., i:i+self.chunk_size]
        Ax_chunk = jnp.dot(params['constraint_matrix'][:, i:i+self.chunk_size], x_chunk.T)
        Ax += Ax_chunk
    else:
      Ax = jnp.dot(params['constraint_matrix'], x.T)
    ub = params['constraint_rhs'][:, None]
    lb = params['constraint_lhs'][:, None]

    if self.formulation == 'max_linear':
      violation = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) 
    elif self.formulation == 'max_linear_square':
      violation = jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) ) 
    elif self.formulation == 'indicator':
      constraint_satisfied = (Ax <= ub) & (Ax >= lb)
      violation = ~constraint_satisfied

    penalty = self.penalty_coeff * jnp.sum(violation, axis=0)
    return penalty

  def objective(self, params, x):
    if self.config.graph_type == 'ca':
      return jnp.dot(x, params['obj_coeffs'])
    elif self.config.graph_type == 'sc':
      return -jnp.dot(x, params['obj_coeffs'])

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
  return ILP(config)
