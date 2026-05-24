"""Utilities."""

import os
from absl import logging
from clu import metric_writers
from clu.metric_writers.summary_writer import SummaryWriter
from PT_ILP import ilp_loader
import jax
import jax.numpy as jnp


def copy_pytree(pytree):
    return jax.tree_util.tree_map(jnp.array, pytree)


def tree_stack(trees):
    """https://gist.github.com/willwhitney/dd89cac6a5b771ccff18b06b33372c75 ."""
    leaves_list = []
    treedef_list = []
    for tree in trees:
        leaves, treedef = jax.tree_util.tree_flatten(tree)
        leaves_list.append(leaves)
        treedef_list.append(treedef)

    grouped_leaves = zip(*leaves_list)
    result_leaves = [jnp.stack(l) for l in grouped_leaves]
    return treedef_list[0].unflatten(result_leaves)


def setup_logging(config):
    """Setup logging and writer."""
    if jax.process_index() == 0:
        logging.info(config)
        logging.info("process count: %d", jax.process_count())
        logging.info("device count: %d", jax.device_count())
        logging.info("device/host: %d", jax.local_device_count())
    logdir = os.path.join(config.experiment.save_root, "logs")
    writer = metric_writers.create_default_writer(logdir, just_logging=jax.process_index() > 0)
    fig_folder = os.path.join(logdir, "figures")
    if jax.process_index() == 0:
        if not os.path.exists(logdir):
            os.makedirs(logdir)
        if not os.path.exists(fig_folder):
            os.makedirs(fig_folder)
        if config.experiment.use_tqdm:
            writer = SummaryWriter(logdir)
        else:
            writer = metric_writers.create_default_writer(logdir)
    else:
        writer = None
    config.experiment.fig_folder = fig_folder
    with open(os.path.join(config.experiment.save_root, "config.yaml"), "w") as f:
        f.write(config.to_yaml())
    return writer


def update_ilp_cfg(config, ilp):
    config.model.max_num_vars = ilp.max_num_vars
    config.model.max_num_cons = ilp.max_num_cons
    config.model.shape = (ilp.max_num_vars,)


def get_datagen(config):
    test_graphs = ilp_loader.get_instances(config)
    update_ilp_cfg(config, test_graphs)
    datagen = test_graphs.get_iterator("test", config.model.num_models)
    return datagen
