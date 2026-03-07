"""Config for Block LBP sampler."""
from ml_collections import config_dict


def get_config():
  sampler_config = dict(
      block_size=2,
      balancing_fn_type='SQRT',
      num_flips=1,
      adaptive=False,
      name='blockgbp',
  )
  return config_dict.ConfigDict(sampler_config)
