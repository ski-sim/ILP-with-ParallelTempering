"""Hamming Ball Sampler Class."""

import copy
from PT_ILP.common import math_util as math
from PT_ILP.samplers import abstractsampler
import jax
from jax import random
import jax.numpy as jnp
import ml_collections


class HammingBallSampler(abstractsampler.AbstractSampler):
    """Hamming Ball Sampler with hamming dist of 1."""

    def __init__(self, config: ml_collections.ConfigDict):
        self.sample_shape = config.model.shape
        self.num_categories = config.model.num_categories
        self.block_size = config.sampler.block_size
        self.hamming = 1
        self.hamming_logit = [1.0]
        if self.num_categories == 2:
            num_samples_per_hamming = [
                math.comb(self.block_size, j + 1) for j in range(self.hamming)
            ]
        else:
            num_samples_per_hamming = [
                math.comb(self.block_size, j + 1) * (self.num_categories - 1) ** (j + 1)
                for j in range(self.hamming)
            ]
        self.hamming_logit = jnp.array(self.hamming_logit + num_samples_per_hamming)
        self.choose_index_vmapped = jax.vmap(self.choose_index, in_axes=[0, None])

    def choose_index(self, rng, arr):
        res = jax.random.choice(rng, arr, shape=(1,), replace=False)
        return res

    def update_sampler_state(self, sampler_state):
        sampler_state = super().update_sampler_state(sampler_state)
        dim = math.prod(self.sample_shape)
        sampler_state["index"] = (sampler_state["index"] + self.block_size) % dim
        sampler_state["num_ll_calls"] += (self.num_categories - 1) * (self.block_size) + 1
        return sampler_state

    def make_init_state(self, rng):
        state = super().make_init_state(rng)
        state["index"] = jnp.zeros(shape=(), dtype=jnp.int32)
        return state

    def compute_u(self, rng, x, block):
        rng_spl, rng_int = random.split(rng)
        flipping_value = random.randint(
            rng_int,
            shape=[x.shape[0]],
            minval=1,
            maxval=self.num_categories,
        )
        rng_v = jax.random.split(rng_spl, x.shape[0])
        flip_index = self.choose_index_vmapped(rng_v, block)
        flip_index = jnp.reshape(flip_index, [-1])
        b_idx = jnp.arange(x.shape[0])
        u = x.at[b_idx, flip_index].set(
            (x[b_idx, flip_index] + flipping_value) % self.num_categories
        )
        return u

    def step(self, model, rng, x, model_param, state, x_mask=None):
        rng1, rng2, rng3 = jax.random.split(rng, 3)
        state_init = copy.deepcopy(state)
        _ = x_mask
        x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        rad = jax.random.categorical(rng1, self.hamming_logit)
        start_index = state["index"]
        block = start_index + jnp.arange(self.block_size)
        u = jnp.where(rad, self.compute_u(rng2, x, block), x)
        u = jnp.reshape(u, x_shape)

        def generate_new_samples(indices_to_flip, x):
            x_flatten = x.reshape(1, -1)
            y_flatten = jnp.repeat(x_flatten, (self.num_categories - 1) * self.block_size, axis=0)
            indices_to_flip = jnp.repeat(indices_to_flip, self.num_categories - 1, axis=0)
            categories_iter = jnp.tile(jnp.arange(1, self.num_categories), self.block_size)
            b_idx = jnp.arange(y_flatten.shape[0])
            y_flatten = y_flatten.at[b_idx, indices_to_flip].set(
                (y_flatten[b_idx, indices_to_flip] + categories_iter) % self.num_categories
            )
            y = y_flatten.reshape((y_flatten.shape[0],) + self.sample_shape)
            return y

        def per_sample_step(rng_key, curr_sample):
            y = generate_new_samples(block, curr_sample)
            y_all = jnp.concatenate([curr_sample[None], y], axis=0)
            loglikelihood, is_valid = model.forward(model_param, y_all)
            selected_index = random.categorical(rng_key, loglikelihood)
            selected_sample = y_all[selected_index]
            log_prob = loglikelihood[selected_index]
            is_valid_sel = is_valid[selected_index]
            return selected_sample, log_prob, is_valid_sel

        batch_size = u.shape[0]
        rng_keys = jax.random.split(rng3, batch_size)
        new_x, log_probs, is_valids = jax.vmap(per_sample_step)(rng_keys, u)
        new_state = self.update_sampler_state(state_init)
        new_state["log_prob"] = log_probs
        new_state["is_valid"] = is_valids
        return new_x, new_state, 1


def build_sampler(config):
    return HammingBallSampler(config)
