"""Max Independent Set model."""

from discs.models import comb_ebm
import jax.numpy as jnp
import ml_collections
import jax

class ILP(comb_ebm.BinaryNodeCombEBM):
  """Max Independent Set model."""

  def __init__(self, config: ml_collections.ConfigDict):
    super().__init__(config)
    self.config = config.model
    self.max_num_nodes = self.config.max_num_nodes
    self.penalty_coeff = self.config.get('penalty', 2.0)
    self.formulation = self.config.get('formulation', 'max_linear')
    self.proposal_type = self.config.get('proposal_type', 'obj')
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
    if self.formulation == 'obj':
      return 0.0
    elif self.formulation == 'obj_lagrangian':
      violation = Ax - ub +  lb - Ax
    elif self.formulation == 'max_linear':
      violation = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) 
    elif self.formulation == 'max_linear_square':
      violation = jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) ) 
    elif self.formulation == 'max_linear_cubic':
      violation = jnp.power(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax), 3) 
    elif self.formulation == 'augmented_lagrangian':
      violation = (jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax)) + jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) ) 
    elif self.formulation == 'indicator':
      constraint_satisfied = (Ax <= ub) & (Ax >= lb)
      violation = ~constraint_satisfied
      # violation = (~constraint_satisfied).astype(jnp.float32)

    penalty = self.penalty_coeff * jnp.sum(violation, axis=0)
    return penalty

  def penalty2(self, params, x):
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
    if self.proposal_type == 'obj':
      return 0.0
    elif self.proposal_type == 'uniform':
      return 0.0
    elif self.proposal_type == 'obj_lagrangian':
      violation = Ax - ub +  lb - Ax
    elif self.proposal_type == 'max_linear':
      violation = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) 
    elif self.proposal_type == 'max_linear_square':
      violation = jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) ) 
    elif self.proposal_type == 'penalty_linear':
      violation = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) 
    elif self.proposal_type == 'penalty_square':
      violation = jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax) ) 

    penalty = self.penalty_coeff * jnp.sum(violation, axis=0)
    return penalty

  def objective(self, params, x):
    if self.config.graph_type == 'ca':
      return jnp.dot(x, params['obj_coeffs'])
    elif self.config.graph_type == 'sc':
      return -jnp.dot(x, params['obj_coeffs'])
    elif self.config.graph_type == 'mis':
      return jnp.dot(x, params['obj_coeffs'])

  def logratio_in_neighborhood(self, params, x):
    # x shape: [batch, N]
    A = params['constraint_matrix']  # [M, N]
    c = params['obj_coeffs']         # [N]
    ub = params['constraint_rhs']    # [M]
    lb = params['constraint_lhs']    # [M]
    temp = params.get('temperature', 1.0)

    Ax = jnp.dot(A, x.T)  # current Ax [M, batch]
    obj_x = self.objective(params, x)  # current c^Tx [batch]
    v_curr = jnp.maximum(0, Ax - ub[:, None]) + jnp.maximum(0, lb[:, None] - Ax)  # [M, batch]
    penalty_x = self.penalty_coeff * jnp.sum(jnp.square(v_curr), axis=0)  # current penalty [batch]
    ll_x = (obj_x - penalty_x) / temp  # current -energy [batch]

    # Deltas for flipping each bit j
    delta_x = 1 - 2 * x  # [batch, N]

    # Change in objective: c_j * delta_x_j
    delta_obj = c[None, :] * delta_x  # [batch, N]
    if self.config.graph_type == 'sc':
        delta_obj = -delta_obj

    # Change in Penalty
    Ax_new = Ax[:, :, None] + A[:, None, :] * delta_x[None, :, :]
    v_new = jnp.maximum(0, Ax_new - ub[:, None, None]) + jnp.maximum(0, lb[:, None, None] - Ax_new)
    penalty_x_new = self.penalty_coeff * jnp.sum(jnp.square(v_new), axis=0)

    # Calculate Log-Ratios
    ll_x_new = (obj_x[:, None] + delta_obj - penalty_x_new) / temp
    logratio = ll_x_new - ll_x[:, None]

    return ll_x, logratio, 1, self.get_neighbor_fn
def build_model(config):
  return ILP(config)
