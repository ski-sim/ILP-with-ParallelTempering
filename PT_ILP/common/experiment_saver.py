"""Saver Class."""

import os
import pickle
import jax.numpy as jnp
import matplotlib.pyplot as plt
import ml_collections
import numpy as np


class Saver:
    """Class used to plot and save the results of the experiments."""

    def __init__(self, config: ml_collections.ConfigDict):
        self.config = config
        self.save_dir = config.experiment.save_root
        if not os.path.isdir(self.save_dir):
            os.makedirs(self.save_dir)

    def _dump_dict(self, params, key_name: str):
        path = os.path.join(self.save_dir, f"{key_name}.pkl")
        if not isinstance(params, dict):
            params = np.array(params)
            params_dict = {}
            params_dict[f"{key_name}"] = params
        else:
            params_dict = params
        with open(path, "wb") as file:
            pickle.dump(params_dict, file, protocol=pickle.HIGHEST_PROTOCOL)

    def save_ilp_results(self, trajectory, best_ratio, elapsed_time, best_samples, acc_ratios):
        results = {}
        results["trajectory"] = np.array(trajectory)
        results["best_ratio"] = np.array(best_ratio)
        results["elapsed_time"] = elapsed_time
        results["best_ratio_mean"] = np.mean(np.array(best_ratio))
        results["acc_ratios"] = np.array(acc_ratios)
        if len(best_samples) != 0:
            results["best_samples"] = np.array(best_samples)

        sampler_name = self.config.sampler.name
        if "lbp" in sampler_name or "path_auxiliary" in sampler_name:
            sampler_name = (
                sampler_name
                + f"_nflip{self.config.sampler.num_flips}"
                + ("_adaptive" if self.config.sampler.adaptive else "")
            )
        self._dump_dict(
            results,
            f"results_{self.config.model.name}_{self.config.model.instance_name}_{self.config.model.max_num_vars}_{sampler_name}_{self.config.model.formulation}_{self.config.experiment.final_temperature}_{self.config.experiment.init_temperature}_{self.config.model.penalty}_{self.config.experiment.l_min}_{self.config.experiment.l_max}_{self.config.experiment.t_schedule}_{self.config.experiment.decay_rate}_{self.config.experiment.pt}_{self.config.experiment.pt_interval}_{self.config.experiment.t_min}_{self.config.experiment.t_max}_{self.config.experiment.reheat}_{self.config.experiment.batch_size}",
        )


def build_saver(config):
    return Saver(config)
