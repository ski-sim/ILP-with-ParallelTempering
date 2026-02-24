"""Main class that runs sampler on the model to generate chains."""

import functools
import time
from discs.common import math_util as math
from discs.common import utils
import flax
import jax
import jax.numpy as jnp
import numpy as np
import optax
import tqdm
from discs.common.parallel_tempering import swap_samples_deo, swap_samples_seo
import wandb
import os


class Experiment:
    """Experiment class that generates chains of samples."""

    def __init__(self, config):
        self.config = config.experiment
        self.config_model = config.model
        self.sampler_name = config.sampler.name
        self.parallel = False
        self.sample_idx = None
        self.num_saved_samples = config.get("nun_saved_samples", 4)
        if jax.local_device_count() != 1 and self.config.run_parallel:
            self.parallel = True

    def _initialize_model_and_sampler(self, rnd, model, sampler):
        """Initializes model params, sampler state and gets the initial samples."""

        if self.config.evaluator == "co_eval":
            sampler_init_state_fn = jax.vmap(sampler.make_init_state)
        else:
            sampler_init_state_fn = sampler.make_init_state
        model_init_params_fn = model.make_init_params
        rng_param, rng_x0, rng_state = jax.random.split(rnd, num=3)
        # params of the model
        params = model_init_params_fn(jax.random.split(rng_param, self.config.num_models))
        # initial samples
        num_samples = self.config.batch_size * self.config.num_models
        x0 = model.get_init_samples(rng_x0, num_samples)
        # initial state of sampler
        state = sampler_init_state_fn(jax.random.split(rng_state, self.config.num_models))
        return params, x0, state

    def _prepare_data(self, params, x, state):
        use_put_replicated = False
        reshape_all = True
        if self.config.evaluator != "co_eval":
            if self.parallel:
                assert self.config.batch_size % jax.local_device_count() == 0
                mini_batch = self.config.batch_size // jax.local_device_count()
                bshape = (jax.local_device_count(),)
                x_shape = bshape + (mini_batch,) + self.config_model.shape
                use_put_replicated = True
                if self.sample_idx:
                    self.sample_idx = jnp.array(
                        [self.sample_idx] * (jax.local_device_count() // self.config.num_models)
                    )
            else:
                reshape_all = False
                bshape = ()
                x_shape = (self.config.batch_size,) + self.config_model.shape
        else:
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
                state = jax.tree_map(fn_breshape, state)
                params = jax.tree_map(fn_breshape, params)
            else:
                params = jax.device_put_replicated(params, jax.local_devices())
                state = jax.device_put_replicated(state, jax.local_devices())
        x = jnp.reshape(x, x_shape)

        print("x shape: ", x.shape)
        print("state shape: ", state["steps"].shape)
        return params, x, state, bshape

    def _compile_sampler_step(self, step_fn):
        if not self.parallel:
            compiled_step = jax.jit(step_fn)
        else:
            compiled_step = jax.pmap(step_fn)
        return compiled_step

    def _compile_evaluator(self, obj_fn):
        if not self.parallel:
            compiled_obj_fn = jax.jit(obj_fn)
        else:
            compiled_obj_fn = jax.pmap(obj_fn)
        return compiled_obj_fn

    def _compile_fns(self, sampler, model, evaluator):
        if self.config.evaluator == "co_eval":
            step_fn = jax.vmap(functools.partial(sampler.step, model=model))
            obj_fn = self._vmap_evaluator(evaluator, model)
        else:
            step_fn = functools.partial(sampler.step, model=model)
            obj_fn = evaluator.evaluate
        compiled_obj_only_fn = jax.vmap(functools.partial(model.objective))
        compiled_penalty_fn = jax.vmap(functools.partial(model.penalty))
        compiled_mask_fn = jax.jit(jax.vmap(self.mask_per_instance, in_axes=(0, 0)))
        compiled_pt_fn = jax.jit(
            jax.vmap(
                functools.partial(
                    swap_samples_deo if self.config.pt == "deo" else swap_samples_seo
                ),
                in_axes=(0, 0, 0, None, None),
            )
        )
        get_hop = jax.jit(self._get_hop)
        compiled_step = self._compile_sampler_step(step_fn)
        compiled_step_burnin = compiled_step
        compiled_step_mixing = compiled_step
        compiled_obj_fn = self._compile_evaluator(obj_fn)
        model_frwrd = jax.jit(model.forward)
        return (
            compiled_step_burnin,
            compiled_step_mixing,
            get_hop,
            compiled_obj_fn,
            model_frwrd,
            compiled_mask_fn,
            compiled_obj_only_fn,
            compiled_penalty_fn,
            compiled_pt_fn,
        )

    def _get_hop(self, x, new_x):
        return jnp.sum(abs(x - new_x)) / self.config.batch_size / self.config.num_models

    def _compute_chain(
        self,
        compiled_fns,
        state,
        params,
        rng,
        x,
        saver,
        evaluator,
        bshape,
        model,
    ):
        raise NotImplementedError

    def vmap_evaluator(self, evaluator, model):
        raise NotImplementedError

    def preprocess(self, model, sampler, evaluator, saver, rnd_key=0):
        rnd = jax.random.PRNGKey(rnd_key)
        params, x, state = self._initialize_model_and_sampler(rnd, model, sampler)
        if params is None:
            print("Params is NONE")
            return False
        params, x, state, breshape = self._prepare_data(params, x, state)
        compiled_fns = self._compile_fns(sampler, model, evaluator)
        return [
            compiled_fns,
            state,
            params,
            rnd,
            x,
            saver,
            evaluator,
            breshape,
            model,
        ]

    def _get_chains_and_evaluations(self, model, sampler, evaluator, saver, rnd_key=0):
        """Sets up the model and the samlping alg and gets the chain of samples."""
        preprocessed_info = self.preprocess(model, sampler, evaluator, saver, rnd_key=0)
        if not preprocessed_info:
            return False
        self._compute_chain(*preprocessed_info)
        return True

    def get_results(self, model, sampler, evaluator, saver):
        self._get_chains_and_evaluations(model, sampler, evaluator, saver)


class CO_Experiment(Experiment):
    """Class used to run annealing schedule for CO problems."""

    def get_results(self, model, sampler, evaluator, saver):
        self.all_chain = []
        self.all_best_ratio = []
        self.all_current_time = []
        self.all_best_samples = []
        while True:
            if not self._get_chains_and_evaluations(model, sampler, evaluator, saver):
                break

    def _initialize_model_and_sampler(self, rnd, model, sampler):
        data_list, x0, state = super()._initialize_model_and_sampler(rnd, model, sampler)
        if data_list is None:
            return None, x0, state
        sample_idx, params, reference_obj = zip(*data_list)
        params = flax.core.frozen_dict.unfreeze(utils.tree_stack(params))
        self.ref_obj = jnp.array(reference_obj)
        if self.config_model.name in ["mis", "ilp"]:
            self.ref_obj = jnp.ones_like(self.ref_obj)
        self.sample_idx = jnp.array(sample_idx)
        return params, x0, state

    def _vmap_evaluator(self, evaluator, model):
        obj_fn = jax.vmap(functools.partial(evaluator.evaluate, model=model))
        return obj_fn

    def _build_temperature_schedule(self, config):
        """Temperature schedule."""

        if config.t_schedule == "constant":
            schedule = lambda step: step * 0 + config.init_temperature
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
                end_value=config.final_temperature,  # can be None
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
        else:
            raise ValueError("Unknown schedule %s" % config.t_schedule)
        return schedule

    def _compute_chain(
        self,
        compiled_fns,
        state,
        params,
        rng,
        x,
        saver,
        evaluator,
        bshape,
        model,
    ):
        """Generates the chain of samples."""

        (
            chain,
            acc_ratios,
            hops,
            running_time,
            best_ratio,
            init_temperature,
            t_schedule,
            sample_mask,
            best_samples,
        ) = self._initialize_chain_vars(bshape)

        stp_burnin, stp_mixing, get_hop, obj_fn, _, mask_fn, obj_only_fn, penalty_fn, pt_fn = (
            compiled_fns
        )
        fn_reshape = lambda x: jnp.reshape(x, bshape + x.shape[1:])

        wandb_name = f"{self.config_model.graph_type}_{self.config_model.max_num_nodes}_{self.sampler_name}_{self.config_model.formulation}_lambda{self.config_model.penalty}_bsz{self.config.batch_size}_{self.config.t_schedule}_decay{self.config.decay_rate}"
        if self.config.t_schedule == "exp_decay":
            wandb_name += f"_init{self.config.init_temperature}"
        elif self.config.t_schedule == "pt_exp_decay":
            wandb_name += f"_{self.config.pt}_int{self.config.pt_interval}_tmin{self.config.t_min}_tmax{self.config.t_max}"
        else:
            raise ValueError(f"Unknown t_schedule: {self.config.t_schedule}")
        wandb.init(name=wandb_name)

        # reheated mechanism
        best_eval_val = jnp.ones(self.config.num_models) * -jnp.inf
        value_chain = jnp.zeros((100, self.config.num_models, self.config.batch_size))
        if self.config.reweight == "reheat":
            shape = (self.config.num_models, self.config.batch_size)
            fake_step = jnp.ones(shape, dtype=jnp.int32)
            max_specific_heat = jnp.zeros(shape, dtype=jnp.float32)
            reheat_step = jnp.zeros(shape, dtype=jnp.int32)
            print_specific_heat = False
            skip_step = 200
            wandering_length = 100
            threshold = 0.5
            reheat_time = jnp.zeros((self.config.num_models, self.config.batch_size))
            trapped_num = jnp.zeros(shape, dtype=jnp.int32)
            trapped_threshold_length = jnp.ones(shape, dtype=jnp.int32) * wandering_length
            old_value = jnp.zeros(shape, dtype=jnp.float32)
            temp_shape = bshape + (self.config.batch_size,)
            init_temperature = jnp.ones(temp_shape, dtype=jnp.float32)
            params["temperature"] = t_schedule(0) * init_temperature  # (100,32)

        pt_step = 0
        num_log = 0
        mean_accept = 0
        pairs = []
        acceptance_ratio = jnp.ones((self.config.num_models, self.config.batch_size)) * -jnp.nan
        elapsed_time = 0
        for step in tqdm.tqdm(range(1, self.config.chain_length * 2 + 1), dynamic_ncols=True):
            start = time.time()

            if self.config.reweight == "reheat":
                cur_temp = t_schedule(fake_step)
                params["temperature"] = jnp.array(cur_temp).reshape(params["temperature"].shape)
            else:
                cur_temp = t_schedule(step)
                params["temperature"] = init_temperature[:, None] * cur_temp

            rng = jax.random.fold_in(rng, step)
            step_rng = fn_reshape(jax.random.split(rng, math.prod(bshape)))

            if self.config.reweight == "mask":
                params["mask"] = mask_fn(x, params)

            new_x, state, acc = stp_mixing(
                rng=step_rng,
                x=x,
                model_param=params,
                state=state,
                x_mask=params["mask"],
            )
            new_ll = state["log_prob"] * params["temperature"]
            is_valid = state["is_valid"]

            # parallel tempering
            if (
                "pt" in self.config.t_schedule
                and self.config.pt in ["deo", "seo"]
                and step % self.config.pt_interval == 0
            ):
                new_x, acceptance_ratio, indices_a, indices_b = pt_fn(
                    new_x, new_ll, params["temperature"], rng, step
                )
                mean_accept = jnp.mean(acceptance_ratio)
                # FIXME: remove [0] later
                pairs = [f"{a}-{b}" for a, b in zip(indices_a[0], indices_b[0])]
                pt_step += 1

            eval_val = jnp.where(is_valid, new_ll, -jnp.inf)
            best_idx = jnp.argmax(eval_val, axis=-1, keepdims=True)
            batch_best_val = jnp.take_along_axis(eval_val, best_idx, axis=-1).squeeze(-1)
            batch_best_x = jnp.take_along_axis(new_x, best_idx[..., None], axis=-2).squeeze(-2)
            is_better = batch_best_val > best_eval_val
            best_eval_val = jnp.where(is_better, batch_best_val, best_eval_val)
            best_samples = jnp.where(is_better[..., None], batch_best_x, best_samples)
            ratio = batch_best_val.reshape(-1) / self.ref_obj  # FIXME: Do we need this?
            best_ratio = jnp.maximum(ratio, best_ratio)  # FIXME: Do we need this?

            new_x.block_until_ready()  # wait for new_x to be ready before calculating the time
            step_time = time.time() - start
            running_time += step_time  # FIXME: Do we need this?
            elapsed_time += step_time

            if self.config_model.mode == "test":
                log_cond = elapsed_time / (
                    self.config_model.max_runtime / self.config.log_every_steps
                ) >= (num_log + 1)
            elif self.config_model.mode == "test_step":
                log_cond = step % (self.config_model.step_limit / self.config.log_every_steps) == 0
            else:  # self.config_model.mode == "val" or "test_step"
                log_cond = step % self.config.log_every_steps == 0

            if self.config_model.mode == "test_step":
                terminate_cond = step >= self.config_model.step_limit
            elif self.config_model.mode == "test":
                terminate_cond = (
                    elapsed_time > self.config_model.max_runtime
                    and num_log > self.config_model.max_runtime // self.config.log_every_steps
                )

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
                                f"instance{_j}/swap_accept_{pairs[k]}": acceptance_ratio[0][k]
                                for k in range(len(pairs))
                            },
                        }
                    )

                if terminate_cond:
                    if self.config_model.mode == "test":
                        print(
                            f"time limit {self.config_model.max_runtime}s reached, early stopping (step: {step})"
                        )
                    elif self.config_model.mode == "test_step":
                        print(
                            f"step limit {self.config_model.step_limit} reached, early stopping (elapsed_time: {elapsed_time:.2f}s)"
                        )
                    break

                sample_mask = sample_mask.reshape(best_ratio.shape)
                br = np.array(best_ratio[sample_mask])
                br = jax.device_put(br, jax.devices("cpu")[0])
                chain.append(br)

                if self.config.save_samples or self.config_model.name == "normcut":
                    step_chosen = jnp.argmax(eval_val, axis=-1, keepdims=True)
                    rnew_x = jnp.reshape(
                        new_x,
                        (self.config.num_models, self.config.batch_size) + self.config_model.shape,
                    )
                    chosen_samples = jnp.take_along_axis(
                        rnew_x, jnp.expand_dims(step_chosen, -1), axis=-2
                    )
                    chosen_samples = jnp.squeeze(chosen_samples, -2)
                    is_better = ratio > best_ratio
                    best_samples = jnp.where(
                        jnp.expand_dims(is_better, -1), chosen_samples, best_samples
                    )

            if self.config.get_additional_metrics:
                # avg over all models
                acc = jnp.mean(acc)
                acc_ratios.append(acc)
                # hop avg over batch size and num models
                hops.append(get_hop(x, new_x))

            if self.config.reweight == "reheat":
                value_chain = value_chain.at[(step - 1) % 100].set(eval_val)
                append_temp = cur_temp.reshape(self.config.num_models, self.config.batch_size)
                value_diff = jnp.abs(eval_val - old_value)
                trapped_num = jnp.where(
                    jnp.abs(value_diff) < threshold,
                    trapped_num + jnp.ones_like(trapped_num),
                    jnp.zeros_like(trapped_num),
                )
                old_value = eval_val
                reheat_time_array = jnp.where(
                    trapped_num >= trapped_threshold_length,
                    jnp.ones_like(trapped_num),
                    jnp.zeros_like(trapped_num),
                )
                reheat_time = reheat_time + reheat_time_array
                if step >= skip_step:
                    specific_heat = jnp.var(value_chain, axis=0) / (append_temp**2)
                    specific_heat = jnp.where(
                        reheat_time == jnp.zeros_like(reheat_time),
                        specific_heat,
                        jnp.zeros_like(reheat_time),
                    )
                    if print_specific_heat:
                        print("specific_heat", jnp.mean(specific_heat))
                    max_specific_heat = jnp.maximum(specific_heat, max_specific_heat)
                    reheat_step = jnp.where(
                        specific_heat >= max_specific_heat, fake_step, reheat_step
                    )
                    fake_step = jnp.where(
                        trapped_num >= trapped_threshold_length,
                        reheat_step - jnp.ones_like(reheat_step),
                        fake_step,
                    )

            # if self.config.reweight == 'reheat': # burn out mechanism
            #   # we don't calculate specific heat for the mixing phase, since we don't update the critical temperature anymore in case it becomes too small
            #   value_diff = jnp.abs(eval_val - old_value)
            #   trapped_num = jnp.where(jnp.abs(value_diff) < threshold, trapped_num + jnp.ones_like(trapped_num),
            #                           jnp.zeros_like(trapped_num))
            #   old_value = eval_val
            #   reheat_time_array = jnp.where(trapped_num >= trapped_threshold_length, jnp.ones_like(trapped_num),
            #                                 jnp.zeros_like(trapped_num))
            #   reheat_time = reheat_time + reheat_time_array
            #   fake_step = jnp.where(trapped_num >= trapped_threshold_length, reheat_step - jnp.ones_like(reheat_step), fake_step)

            x = new_x
            if self.config.reweight == "reheat":
                fake_step = fake_step + jnp.ones_like(fake_step)

        if not (self.config.save_samples or self.config_model.name == "normcut"):
            best_samples = []

        # saver.save_co_resuts(
        #     chain, best_ratio[sample_mask], running_time, best_samples
        # )
        # saver.save_results(acc_ratios, hops, None, running_time)

        self.all_chain.append(chain)
        self.all_best_ratio.append(best_ratio[sample_mask])
        self.all_current_time.append([running_time])
        self.all_best_samples.append(best_samples)

        all_chain = np.concatenate(self.all_chain, axis=1)
        all_best_ratio = np.concatenate(self.all_best_ratio, axis=0)
        all_current_time = np.concatenate(self.all_current_time, axis=0)
        all_best_samples = np.concatenate(self.all_best_samples, axis=0)

        saver.save_co_results(all_chain, all_best_ratio, all_current_time, all_best_samples)
        saver.save_results(acc_ratios, hops, None, running_time)

    def _initialize_chain_vars(self, bshape):
        t_schedule = self._build_temperature_schedule(self.config)
        sample_mask = self.sample_idx >= 0
        chain = []
        acc_ratios = []
        hops = []
        running_time = 0
        best_ratio = jnp.ones(self.config.num_models, dtype=jnp.float32) * -float("inf")
        init_temperature = jnp.ones(bshape, dtype=jnp.float32)
        dim = math.prod(self.config_model.shape)
        best_samples = jnp.zeros([self.config.num_models, dim])
        return (
            chain,
            acc_ratios,
            hops,
            running_time,
            best_ratio,
            init_temperature,
            t_schedule,
            sample_mask,
            best_samples,
        )

    def mask_per_instance(self, x, params):
        cur_selected_vars = x == 1
        cur_x = x

        constraint_matrix = params["constraint_matrix"]
        constraint_rhs = params["constraint_rhs"]
        constraint_lhs = params["constraint_lhs"]

        cur_constraint_matrix = constraint_matrix
        cur_rhs = constraint_rhs
        cur_lhs = constraint_lhs
        cur_constraint_values = jnp.dot(
            cur_selected_vars.astype(jnp.float32), cur_constraint_matrix.T
        )
        sign = (1 - 2 * cur_x).astype(jnp.float32)

        constraint_changes = jnp.einsum("bv,vc->bvc", sign, cur_constraint_matrix.T)
        updated_values = cur_constraint_values[:, None, :] + constraint_changes

        violates = (updated_values < cur_lhs[None, None, :]) | (
            updated_values > cur_rhs[None, None, :]
        )
        feasible_mask = ~violates.any(axis=2)
        all_masks = jnp.stack(feasible_mask)

        return all_masks[None, ...]
