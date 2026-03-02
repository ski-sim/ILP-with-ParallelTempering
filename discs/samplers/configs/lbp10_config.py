"""Config for Path Auxiliary sampler."""

from ml_collections import config_dict


def get_config():
    sampler_config = dict(
        name="lbp",
        num_flips=10,
        adaptive=False,
        balancing_fn_type="SQRT",
    )
    return config_dict.ConfigDict(sampler_config)
