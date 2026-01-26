"""Config for sc dataset."""

from discs.common import utils
from ml_collections import config_dict


def get_model_config(cfg_str):
  """Get config for sc benchmark graphs."""
  extra_cfg = utils.parse_cfg_str(cfg_str)
  rand_type = extra_cfg['r']
  num_nodes = 1000
  num_constraints = 5000
  num_instances = 100

  model_config = dict(
      num_models=100,
      max_num_nodes=num_nodes,
      max_num_constraints=num_constraints,
      num_instances=num_instances,
      num_categories=2,
      shape=(0,),
      rand_type=rand_type,
      formulation='max_linear_square', # indicator
      # proposal_type='obj_lagrangian', # grad, ll
  )
  return config_dict.ConfigDict(model_config)
