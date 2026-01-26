from ml_collections import config_dict
# config=./discs/run_configs/co/ilp/sc_sweep.py ./discs/run_xmanager.sh

def get_config():
  """Get config."""

  config = config_dict.ConfigDict(
      dict(
          model='ilp',
          sampler='path_auxiliary',
          graph_type='ca',
          sweep=[
              
              {
                  'sampler_config.name': [
                      'path_auxiliary',
                  ],
                  'config.experiment.chain_length': [2000],
                  'config.experiment.t_min': [0.01,0.02,0.05],
                  'config.experiment.t_min': [5],
                  
              },
              
          ],
      )
  )
  return config

