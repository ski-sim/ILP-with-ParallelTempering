"""Config file for graph models for mis problem."""
from ml_collections import config_dict

def get_config():
  model_config = config_dict.ConfigDict(
      dict(
          name='mis',
          graph_type='mistest',
          cfg_str='r-1500',
          data_root='./sco/',
          penalty=1000,
      )
  )
  model_config['save_dir_name'] = model_config['name']
  return model_config
