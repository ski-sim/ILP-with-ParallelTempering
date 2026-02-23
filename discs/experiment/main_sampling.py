"""Main script for sampling based experiments."""

import importlib
from absl import app
from absl import flags
from discs.common import configs as common_configs
from discs.common import utils
import discs.common.experiment_saver as saver_mod
from ml_collections import config_flags

FLAGS = flags.FLAGS

# Config file flags
_SAMPLER_CONFIG = config_flags.DEFINE_config_file("sampler_config")
_RUN_LOCAL = flags.DEFINE_boolean("run_local", False, "if runnng local")

# Model flags
flags.DEFINE_string("model", "ilp", "model")
flags.DEFINE_string("graph_type", "mis", "graph_type")
flags.DEFINE_string("cfg_str", "r-800", "cfg_str")
flags.DEFINE_string("data_root", "./sco/", "data_root")
flags.DEFINE_float("penalty_weight", 10, "penalty_weight")
flags.DEFINE_string("save_dir_name", "ilp", "save_dir_name")
flags.DEFINE_integer("num_models", 1, "num_models")
flags.DEFINE_integer("max_num_nodes", 1500, "max_num_nodes")
flags.DEFINE_integer("max_num_constraints", 10000, "max_num_constraints")
flags.DEFINE_integer("num_instances", 20, "num_instances")
flags.DEFINE_integer("num_categories", 2, "num_categories")
flags.DEFINE_string("rand_type", "r-800", "rand_type")
flags.DEFINE_string("formulation", "max_linear_square", "formulation")

# Experiment flags
flags.DEFINE_integer("batch_size", 32, "batch_size")
flags.DEFINE_string("t_schedule", "exp_decay", "t_schedule")
flags.DEFINE_integer("chain_length", 100000, "chain_length")
flags.DEFINE_integer("log_every_steps", 100, "log_every_steps")
flags.DEFINE_integer("save_every_steps", 100, "save_every_steps")
flags.DEFINE_float("decay_rate", 0.01, "decay_rate")
flags.DEFINE_string("save_root", "./discs/results", "save_root")
flags.DEFINE_string("pt", "deo", "pt")
flags.DEFINE_integer("pt_interval", 1000, "pt_interval")
flags.DEFINE_float("init_temperature", 5, "init_temperature")
flags.DEFINE_float("final_temperature", 0.0, "final_temperature")
flags.DEFINE_float("t_min", 0.1, "t_min")
flags.DEFINE_float("t_max", 5.0, "t_max")
flags.DEFINE_string("reweight", "None", "reweight")
flags.DEFINE_string("mode", "val", "mode")
flags.DEFINE_float("max_runtime", 200.0, "max_runtime")
flags.DEFINE_integer("step_limit", 50000, "step_limit")

def update_save_dir(config):
    if _RUN_LOCAL.value:
        save_folder = config.model.get("save_dir_name", config.model.name)
        save_root = "./discs/results/" + save_folder
        config.experiment.save_root = save_root


def get_main_config(FLAGS):
    """Merge experiment, model and sampler config."""
    config = common_configs.get_config(FLAGS)
    config.sampler.update(_SAMPLER_CONFIG.value)
    return config


def main(_):
    config = get_main_config(FLAGS)
    update_save_dir(config)
    utils.setup_logging(config)

    # model
    model_mod = importlib.import_module("discs.models.%s" % config.model.name)
    model = model_mod.build_model(config)

    # sampler
    sampler_mod = importlib.import_module("discs.samplers.%s" % config.sampler.name)
    sampler = sampler_mod.build_sampler(config)

    # experiment
    experiment_mod = getattr(
        importlib.import_module("discs.experiment.sampling"),
        f"{config.experiment.name}",
    )
    experiment = experiment_mod(config)

    # evaluator
    evaluator_mod = importlib.import_module("discs.evaluators.%s" % config.experiment.evaluator)
    evaluator = evaluator_mod.build_evaluator(config)

    # saver
    saver = saver_mod.build_saver(config)

    # chain generation
    experiment.get_results(model, sampler, evaluator, saver)


if __name__ == "__main__":
    # import jax

    # with jax.disable_jit():
    app.run(main)
