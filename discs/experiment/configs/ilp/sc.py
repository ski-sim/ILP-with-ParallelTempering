"""Experiment config for ertest dataset."""
from ml_collections import config_dict


def get_config():
  """Get config for er benchmark graphs."""
  exp_config = dict(
      experiment=dict(
          batch_size=32,
          t_schedule='exp_decay',
          chain_length=100000,
          log_every_steps=100,
          save_every_steps=100,
          init_temperature=5,
          decay_rate=0.01,
          final_temperature=0.0001,
          save_root='',
          pt = 'deo',
          pt_interval = 1000,
          t_min = 0.1,
          t_max = 5.0,
          reweight= 'None',
      )
  )
  return config_dict.ConfigDict(exp_config)
# 38117
# 52008