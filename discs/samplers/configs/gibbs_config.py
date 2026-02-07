"""Config for Gibbs sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      name='gibbs',
      num_flips=1,
  )
  return config_dict.ConfigDict(sampler_config)
