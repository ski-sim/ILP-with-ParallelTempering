"""Main class that runs sampler on the model to generate chains."""

import functools
import time

import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
import wandb

from PT_ILP.common import math_util as math
from PT_ILP.common import utils
import PT_ILP.common.penalty_pt as penalty_pt
from PT_ILP.common.temperature_pt import swap_samples_deo, swap_samples_seo, swap_samples_reversible


class ReheatController:
    """Reheat state and per-step update logic."""

    SKIP_STEP = 200
    WANDERING_LENGTH = 100
    THRESHOLD = 0.5
    PRINT_SPECIFIC_HEAT = False

    def __init__(self, num_models, batch_size, bshape):
        shape = (num_models, batch_size)
        temp_shape = bshape + (batch_size,)
        self.num_models = num_models
        self.batch_size = batch_size

        self.fake_step = jnp.ones(shape, dtype=jnp.int32)
        self.max_specific_heat = jnp.zeros(shape, dtype=jnp.float32)
        self.reheat_step = jnp.zeros(shape, dtype=jnp.int32)
        self.reheat_time = jnp.zeros(shape)
        self.trapped_num = jnp.zeros(shape, dtype=jnp.int32)
        self.trapped_threshold_length = jnp.ones(shape, dtype=jnp.int32) * self.WANDERING_LENGTH
        self.old_value = jnp.zeros(shape, dtype=jnp.float32)
        self.value_chain = jnp.zeros((100, num_models, batch_size))
        self.init_temperature = jnp.ones(temp_shape, dtype=jnp.float32)

    def initial_params_temperature(self, t_schedule):
        return t_schedule(0) * self.init_temperature

    def temperature(self, t_schedule, params_temperature_shape):
        cur_temp = t_schedule(self.fake_step)
        params_temperature = jnp.array(cur_temp).reshape(params_temperature_shape)
        return cur_temp, params_temperature

    def update(self, eval_val, cur_temp, step):
        self.value_chain = self.value_chain.at[(step - 1) % 100].set(eval_val)
        append_temp = cur_temp.reshape(self.num_models, self.batch_size)

        value_diff = jnp.abs(eval_val - self.old_value)
        is_trapped_step = value_diff < self.THRESHOLD
        self.trapped_num = jnp.where(is_trapped_step, self.trapped_num + 1, 0)
        self.old_value = eval_val

        is_trapped_full = self.trapped_num >= self.trapped_threshold_length
        self.reheat_time = self.reheat_time + is_trapped_full.astype(self.reheat_time.dtype)

        if step >= self.SKIP_STEP:
            specific_heat = jnp.var(self.value_chain, axis=0) / (append_temp ** 2)
            specific_heat = jnp.where(self.reheat_time == 0, specific_heat, 0)
            if self.PRINT_SPECIFIC_HEAT:
                print("specific_heat", jnp.mean(specific_heat))
            self.max_specific_heat = jnp.maximum(specific_heat, self.max_specific_heat)
            self.reheat_step = jnp.where(
                specific_heat >= self.max_specific_heat, self.fake_step, self.reheat_step
            )
            self.fake_step = jnp.where(
                is_trapped_full, self.reheat_step - 1, self.fake_step
            )

    def tick(self):
        self.fake_step = self.fake_step + 1


class MCMCExperiment:
    """Experiment class that generates chains of samples."""

    def __init__(self, config):
        self.config = config.experiment
        self.config_model = config.model
        self.sampler_name = config.sampler.name
        if "lbp" in self.sampler_name or "path_auxiliary" in self.sampler_name:
            self.sampler_name = (
                self.sampler_name
                + f"_nflip{config.sampler.num_flips}"
                + ("_adaptive" if config.sampler.adaptive else "")
            )
        self.parallel = False
        self.sample_idx = None
        if jax.local_device_count() != 1 and self.config.run_parallel:
            self.parallel = True
        
        self.batches = []

    def run(self, model, sampler, saver):
        while True:
            if not self.run_batch(model, sampler, saver):
                break

    def run_batch(self, model, sampler, saver, rnd_key=0):
        """Sets up the model and the samlping alg and gets the chain of samples."""
        preprocessed_info = self.preprocess(model, sampler, saver, rnd_key=0)
        if not preprocessed_info:
            return False
        self.run_mcmc_loop(*preprocessed_info)
        return True

    def preprocess(self, model, sampler, saver, rnd_key=0):
        rnd = jax.random.PRNGKey(rnd_key)
        params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
        if params is None:
            print("Params is NONE")
            return False
        params, x, state, breshape = self._prepare_data(params, x, state)
        compiled_fns = self._compile_fns(sampler, model)
        return [
            compiled_fns,
            state,
            params,
            rnd,
            x,
            saver,
            breshape,
            model,
        ]

    def _initialize_model_and_sampler(self, rnd, model, sampler):
        """Load the next ILP instance, draw initial samples, init sampler state."""
        rng_param, rng_x0, rng_state = jax.random.split(rnd, num=3)

        # Initial samples + per-model sampler state (vmapped over num_models)
        num_samples = self.config.batch_size * self.config.num_models
        x0 = model.get_init_samples(rng_x0, num_samples)
        sampler_init = jax.vmap(sampler.make_init_state)
        state = sampler_init(jax.random.split(rng_state, self.config.num_models))

        # Pull next ILP instance; None signals datagen exhausted
        data_list = model.make_init_params(jax.random.split(rng_param, self.config.num_models))
        if data_list is None:
            return None, x0, state

        # Unpack per-model (idx, params) tuples and stack params trees
        sample_idx, params = zip(*data_list)
        params = flax.core.frozen_dict.unfreeze(utils.tree_stack(params))
        self.sample_idx = jnp.array(sample_idx)

        return params, x0, state

    def _prepare_data(self, params, x, state):
        use_put_replicated = False
        reshape_all = True
        if self.parallel:
            if self.config.num_models >= jax.local_device_count():
                assert self.config.num_models % jax.local_device_count() == 0
                num_models_per_device = self.config.num_models // jax.local_device_count()
                bshape = (jax.local_device_count(), num_models_per_device)
                x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
            else:
                assert self.config.batch_size % jax.local_device_count() == 0
                batch_size_per_device = self.config.batch_size // jax.local_device_count()
                use_put_replicated = True
                bshape = (jax.local_device_count(), self.config.num_models)
                x_shape = bshape + (batch_size_per_device,) + self.config_model.shape
                if self.sample_idx:
                    self.sample_idx = jnp.array(
                        [self.sample_idx] * (jax.local_device_count() // self.config.num_models)
                    )
        else:
            bshape = (self.config.num_models,)
            x_shape = bshape + (self.config.batch_size,) + self.config_model.shape
        fn_breshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])
        if reshape_all:
            if not use_put_replicated:
                state = jax.tree_util.tree_map(fn_breshape, state)
                params = jax.tree_util.tree_map(fn_breshape, params)
            else:
                params = jax.device_put_replicated(params, jax.local_devices())
                state = jax.device_put_replicated(state, jax.local_devices())
        x = jnp.reshape(x, x_shape)

        print("x shape: ", x.shape)
        print("state shape: ", state["steps"].shape)
        return params, x, state, bshape

    def _compile_fns(self, sampler, model):
        step_fn = jax.vmap(functools.partial(sampler.step, model=model))
        compiled_step = self._compile_sampler_step(step_fn)
        compiled_penalty_fn = jax.vmap(functools.partial(model.penalty))

        # Pick PT swap functions based on pt mode (fallback to seo for unknown modes)
        temp_swap_fns = {
            "deo": swap_samples_deo,
            "seo": swap_samples_seo,
            "reversible": swap_samples_reversible,
        }
        pen_swap_fns = {
            "deo": penalty_pt.swap_samples_deo,
            "seo": penalty_pt.swap_samples_seo,
            "reversible": penalty_pt.swap_samples_reversible,
        }
        temp_swap_fn = temp_swap_fns.get(self.config.pt, swap_samples_seo)
        pen_swap_fn = pen_swap_fns.get(self.config.pt, penalty_pt.swap_samples_seo)

        compiled_pt_temp_fn = jax.jit(jax.vmap(
            temp_swap_fn,
            in_axes=(0, 0, 0, None, None),
            out_axes=(0, 0, None, None),
        ))
        compiled_pt_pen_fn = jax.jit(jax.vmap(
            pen_swap_fn,
            in_axes=(0, 0, 0, 0, 0, None, None),
            out_axes=(0, 0, None, None),
        ))

        return (
            compiled_step,
            compiled_penalty_fn,
            compiled_pt_temp_fn,
            compiled_pt_pen_fn,
        )

    def _compile_sampler_step(self, step_fn):
        if not self.parallel:
            compiled_step = jax.jit(step_fn)
        else:
            compiled_step = jax.pmap(step_fn)
        return compiled_step

    # ---------- MCMC loop ----------

    def run_mcmc_loop(
        self,
        compiled_fns,
        state,
        params,
        rng,
        x,
        saver,
        bshape,
        model,
    ):
        """Generates the chain of samples."""

        wandb_name = f"{self.config_model.instance_name}_{self.config_model.max_num_vars}_{self.sampler_name}_{self.config_model.formulation}_lambda{self.config_model.penalty}_bsz{self.config.batch_size}_{self.config.t_schedule}"
        if self.config.t_schedule == "constant":
            wandb_name += f"_init{self.config.init_temperature}"
        elif self.config.t_schedule == "exp_decay":
            wandb_name += f"_init{self.config.init_temperature}_decay{self.config.decay_rate}"
        elif self.config.t_schedule == "pt":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_tmin{self.config.t_min}_tmax{self.config.t_max}"
        elif self.config.t_schedule == "pt_exp_decay":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_tmin{self.config.t_min}_tmax{self.config.t_max}_decay{self.config.decay_rate}"
        elif self.config.t_schedule == "pen_pt":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_lmin{self.config.l_min}_lmax{self.config.l_max}"
        elif self.config.t_schedule == "pen_pt_exp_decay":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_lmin{self.config.l_min}_lmax{self.config.l_max}_decay{self.config.decay_rate}"
        else:
            raise ValueError(f"Unknown t_schedule: {self.config.t_schedule}")
        wandb.init(name=wandb_name)

        (
            chain,
            acc_ratios,
            init_temperature,
            t_schedule,
            sample_mask,
            best_samples,
        ) = self._initialize_chain_vars(bshape)

        step_fn, penalty_fn, pt_temp_fn, pt_pen_fn = compiled_fns
        fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])

        # reheated mechanism
        best_eval_val = jnp.ones(self.config.num_models) * -jnp.inf
        reheat = None
        if self.config.reheat:
            reheat = ReheatController(
                self.config.num_models, self.config.batch_size, bshape
            )
            params["temperature"] = reheat.initial_params_temperature(t_schedule)

        pt_step = 0
        num_log = 0
        mean_accept = 0
        pairs = []
        acceptance_ratio = jnp.ones((self.config.num_models, self.config.batch_size)) * -jnp.nan
        elapsed_time = 0

        for step in tqdm.tqdm(range(1, self.config.chain_length * 2 + 1), dynamic_ncols=True):
            start_time = time.time()

            if reheat is not None:
                cur_temp, params["temperature"] = reheat.temperature(
                    t_schedule, params["temperature"].shape
                )
            else:
                cur_temp = t_schedule(step)
                params["temperature"] = init_temperature[:, None] * cur_temp

            rng = jax.random.fold_in(rng, step)
            step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))

            # transition
            new_x, state, acc = step_fn(
                rng=step_rng,
                x=x,
                model_param=params,
                state=state,
                x_mask=params["mask"],
            )
            new_ll = state["log_prob"] * params["temperature"]
            is_valid = state["is_valid"]

            # parallel tempering
            if step % self.config.pt_interval == 0:
                swap_result = None
                if self.config.t_schedule in ["pen_pt", "pen_pt_exp_decay"]:
                    lambda_ladder = jnp.geomspace(
                        self.config.l_min, self.config.l_max, num=self.config.batch_size
                    )[None, :]
                    swap_result = pt_pen_fn(
                        new_x, new_ll, model.obj_sign * params['obj_coeffs'],
                        params["temperature"], lambda_ladder, rng, pt_step,
                    )
                elif (
                    self.config.t_schedule in ["pt", "pt_exp_decay"]
                    and self.config.pt in ["deo", "seo", "reversible"]
                ):
                    swap_result = pt_temp_fn(
                        new_x, new_ll, params["temperature"], rng, pt_step,
                    )

                if swap_result is not None:
                    new_x, acceptance_ratio, indices_a, indices_b = swap_result
                    mean_accept = jnp.mean(acceptance_ratio)
                    pairs = [f"{a}-{b}" for a, b in zip(indices_a, indices_b)]
                    pt_step += 1


            eval_val = jnp.where(is_valid, new_ll, -jnp.inf)
            best_idx = jnp.argmax(eval_val, axis=-1, keepdims=True)
            batch_best_val = jnp.take_along_axis(eval_val, best_idx, axis=-1).squeeze(-1)
            batch_best_x = jnp.take_along_axis(new_x, best_idx[..., None], axis=-2).squeeze(-2)
            is_better = batch_best_val > best_eval_val
            best_eval_val = jnp.where(is_better, batch_best_val, best_eval_val)
            best_samples = jnp.where(is_better[..., None], batch_best_x, best_samples)

            new_x.block_until_ready()  # wait for new_x to be ready before calculating the time
            step_time = time.time() - start_time
            elapsed_time += step_time

            if self.config_model.mode == "runtimelimit":
                log_cond = elapsed_time / (
                    self.config_model.max_runtime / self.config.log_every_steps
                ) >= (num_log + 1)
                terminate_cond = (
                    elapsed_time > self.config_model.max_runtime
                    and num_log > self.config_model.max_runtime // self.config.log_every_steps
                )
            elif self.config_model.mode == "steplimit":
                log_cond = step % (self.config_model.step_limit / self.config.log_every_steps) == 0
                terminate_cond = step >= self.config_model.step_limit
            else:
                raise ValueError(f"Unknown mode: {self.config_model.mode}")

            if log_cond or terminate_cond:
                num_log += 1
                penalty_val = penalty_fn(params, new_x)

                for _i, _j in enumerate(self.sample_idx):
                    mean_ll = jnp.mean(new_ll[_i], axis=-1).item()
                    best_ll = jnp.max(new_ll[_i], axis=-1).item()
                    mean_obj = jnp.mean(eval_val[_i], axis=-1).item()
                    best_obj = jnp.max(eval_val[_i], axis=-1).item()
                    mean_penalty = jnp.mean(penalty_val[_i], axis=-1).item()
                    bks_obj = best_eval_val[_i].item()

                    if _i == 0:
                        if len(cur_temp.shape) > 1:  # FIXME: why does cur_temp have length 2?
                            print(
                                f"instance{_j}, temp_min: {cur_temp[0, 0]:.4f}, temp_max: {cur_temp[0, -1]:.4f}",
                                end=" ",
                            )
                        else:
                            print(f"instance{_j}, temp: {cur_temp}", end=" ")
                        print(
                            f"bks_obj: {bks_obj:.4f}, best_ll: {best_ll:.4f}, mean_ll: {mean_ll:.4f}, best_obj: {best_obj:.4f}, mean_obj: {mean_obj:.4f}, mean_penalty: {mean_penalty:.4f}"
                        )
                        if "pt" in self.config.t_schedule:
                            print(
                                f"pt_step: {pt_step}, mean_accept: {mean_accept:.4f}, first_pair_accept: {acceptance_ratio[0][0]:.4f}, last_pair_accept: {acceptance_ratio[0][-1]:.4f}"
                            )

                    wandb.log(
                        {
                            f"instance{_j}/bks_obj": bks_obj,
                            f"instance{_j}/best_ll": best_ll,
                            f"instance{_j}/mean_ll": mean_ll,
                            f"instance{_j}/best_obj": best_obj,
                            f"instance{_j}/mean_obj": mean_obj,
                            f"instance{_j}/mean_penalty": mean_penalty,
                            f"instance{_j}/mean_accept": mean_accept,
                            f"instance{_j}/elapsed_time": elapsed_time,
                            f"instance{_j}/step": step,
                            **{
                                f"instance{_j}/swap_accept_{pair}": acceptance_ratio[0][k]
                                for k, pair in enumerate(pairs)
                            },
                        }
                    )

                if terminate_cond:
                    if self.config_model.mode == "runtimelimit":
                        print(
                            f"time limit {self.config_model.max_runtime}s reached, early stopping (step: {step})"
                        )
                    elif self.config_model.mode == "steplimit":
                        print(
                            f"step limit {self.config_model.step_limit} reached, early stopping (elapsed_time: {elapsed_time:.2f}s)"
                        )
                    break

                sample_mask = sample_mask.reshape(best_eval_val.shape)
                bev = np.array(best_eval_val[sample_mask])
                bev = jax.device_put(bev, jax.devices("cpu")[0])
                chain.append(bev)

            acc_ratios.append(jnp.mean(acc))

            if reheat is not None:
                reheat.update(eval_val, cur_temp, step)

            x = new_x
            if reheat is not None:
                reheat.tick()

        self.batches.append({
            "trajectory": chain,
            "best_obj": best_eval_val[sample_mask],
            "elapsed_time": [elapsed_time],
            "best_samples": best_samples,
            "acc_ratios": np.asarray(acc_ratios),
        })
        concat_axes = {
            "trajectory": 1,
            "best_obj": 0,
            "elapsed_time": 0,
            "best_samples": 0,
            "acc_ratios": 0,
        }
        saver.save_ilp_results(**{
            k: np.concatenate([b[k] for b in self.batches], axis=ax)
            for k, ax in concat_axes.items()
        })

    # ---------- MCMC loop helpers ----------

    def _initialize_chain_vars(self, bshape):
        t_schedule = self._build_temperature_schedule(self.config)
        sample_mask = self.sample_idx >= 0
        chain = []
        acc_ratios = []
        init_temperature = jnp.ones(bshape, dtype=jnp.float32)
        dim = math.prod(self.config_model.shape)
        best_samples = jnp.zeros([self.config.num_models, dim])
        return (
            chain,
            acc_ratios,
            init_temperature,
            t_schedule,
            sample_mask,
            best_samples,
        )

    def _build_temperature_schedule(self, config):
        """Temperature schedule."""

        if config.t_schedule == "constant":
            schedule = lambda step: step * 0 + jnp.array(config.init_temperature)
        elif config.t_schedule == "linear":
            assert config.final_temperature is not None  # cannot be None for linear schedule
            schedule = optax.linear_schedule(
                config.init_temperature, config.final_temperature, config.chain_length
            )
        elif config.t_schedule == "exp_decay":
            schedule = optax.exponential_decay(
                config.init_temperature,
                config.chain_length,
                config.decay_rate,
                end_value=0.0,
            )
        elif config.t_schedule == "pt":
            schedule = (
                lambda step: step * 0
                + jnp.geomspace(self.config.t_min, self.config.t_max, num=self.config.batch_size)[
                    None, :
                ]
            )
        elif config.t_schedule == "pt_exp_decay":
            pt_base = jnp.geomspace(
                self.config.t_min, self.config.t_max, num=self.config.batch_size
            )[None, :]
            decay_schedule = optax.exponential_decay(
                1.0,
                config.chain_length,
                config.decay_rate,
                end_value=0.0,
            )
            schedule = lambda step: pt_base * decay_schedule(step)
        elif config.t_schedule == "pen_pt":
            schedule = lambda step: step * 0 + jnp.array(config.init_temperature)
        elif config.t_schedule == "pen_pt_exp_decay":
            schedule = optax.exponential_decay(
                config.init_temperature,
                config.chain_length,
                config.decay_rate,
                end_value=0.0,
            )
        else:
            raise ValueError("Unknown schedule %s" % config.t_schedule)
        return schedule

        

