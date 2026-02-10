"""Config file for graph models for ilp problem."""
from ml_collections import config_dict

def get_config():
    model_config = config_dict.ConfigDict(
        dict(
            name='ilp',
            graph_type='sc',
            cfg_str='r-800',
            data_root='./sco/',
            penalty=10,
        )
    )
    model_config['save_dir_name'] = model_config['name']
    return model_config
