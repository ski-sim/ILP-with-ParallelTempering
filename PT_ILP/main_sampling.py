"""Main script for sampling based experiments."""

import importlib
import os
from absl import app
from absl import flags
from PT_ILP.common import configs as common_configs
from PT_ILP.common import utils
import PT_ILP.common.experiment_saver as saver_mod
from PT_ILP.sampling import MCMCExperiment
from ml_collections import config_flags

FLAGS = flags.FLAGS

# Config file flags
_SAMPLER_CONFIG = config_flags.DEFINE_config_file("sampler_config")

# Model flags
flags.DEFINE_string("model", "ilp", "model")
flags.DEFINE_string("instance_name", "mis", "instance_name")
flags.DEFINE_string("data_root", "./instances/", "data_root")
flags.DEFINE_float("penalty_weight", 10, "penalty_weight")
flags.DEFINE_string("save_dir_name", "ilp", "save_dir_name")
flags.DEFINE_integer("num_models", 1, "num_models")
flags.DEFINE_integer("max_num_vars", 1500, "max_num_vars")
flags.DEFINE_integer("max_num_cons", 10000, "max_num_cons")
flags.DEFINE_integer("num_instances", 20, "num_instances")
flags.DEFINE_integer("num_categories", 2, "num_categories")
flags.DEFINE_string("formulation", "max_linear", "formulation")

# Experiment flags
flags.DEFINE_integer("batch_size", 15, "batch_size")
flags.DEFINE_string("t_schedule", "exp_decay", "t_schedule")
flags.DEFINE_integer("chain_length", 100000, "chain_length")
flags.DEFINE_integer("log_every_steps", 100, "log_every_steps")
flags.DEFINE_float("decay_rate", 0.5, "decay_rate")
flags.DEFINE_string("save_root", "./PT_ILP/results", "save_root")
flags.DEFINE_string("pt", "deo", "pt")
flags.DEFINE_integer("pt_interval", 200, "pt_interval")
flags.DEFINE_float("init_temperature", 5, "init_temperature")
flags.DEFINE_float("final_temperature", 0.0, "final_temperature")
flags.DEFINE_float("t_min", 0.1, "t_min")
flags.DEFINE_float("t_max", 0.2, "t_max")
flags.DEFINE_float("l_min", 5.0, "l_min")
flags.DEFINE_float("l_max", 10.0, "l_max")
flags.DEFINE_boolean("reheat", False, "reheat")
flags.DEFINE_string("mode", "steplimit", "termination mode: steplimit or runtimelimit")
flags.DEFINE_float("max_runtime", 200.0, "max_runtime")
flags.DEFINE_integer("step_limit", 200000, "step_limit")

def get_main_config(FLAGS):
    """Merge experiment, model and sampler config."""
    config = common_configs.get_config(FLAGS)
    config.sampler.update(_SAMPLER_CONFIG.value)
    save_folder = config.model.get("save_dir_name", config.model.name)
    config.experiment.save_root = os.path.join(config.experiment.save_root, save_folder)
    return config


def main(_):
    # set the config
    config = get_main_config(FLAGS)
    utils.setup_logging(config)

    # model
    from PT_ILP.ilp import build_model
    model = build_model(config)

    # sampler
    sampler_mod = importlib.import_module("PT_ILP.samplers.%s" % config.sampler.name)
    sampler = sampler_mod.build_sampler(config)

    # experiment
    experiment = MCMCExperiment(config)

    # saver
    saver = saver_mod.build_saver(config)

    # chain generation
    experiment.run(model, sampler, saver)


if __name__ == "__main__":
    app.run(main)
