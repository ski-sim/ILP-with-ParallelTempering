"""Path auxiliary sampler."""

from discs.common import math_util as math
from discs.samplers import locallybalanced
import jax
import jax.numpy as jnp
from jax.scipy import special
import ml_collections


class LBPforILP(locallybalanced.LocallyBalancedSampler):
    """Path auxiliary sampler with no replacement proposal sampling."""

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__(config)
        self.num_flips = config.sampler.get("num_flips", 1)
        self.adaptive = config.sampler.get("adaptive", False)  # TODO
        self.batch_rows = jnp.expand_dims(jnp.arange(config.experiment.batch_size), axis=1)

    def step(self, model, rng, x, model_param, state, x_mask=None):
        rng_new_sample, rng_acceptance = jax.random.split(rng)
        ll_x, y, trajectory, num_calls_forward, is_valid_x = self.proposal(
            model, rng_new_sample, x, model_param, state
        )
        ll_x2y = trajectory["ll_x2y"]
        ll_y, ll_y2x, num_calls_backward, is_valid_y = self.ll_y2x(
            model, x, model_param, trajectory, y
        )

        log_acc = ll_y + ll_y2x - ll_x - ll_x2y
        new_x, new_state = self.select_sample(
            rng_acceptance,
            num_calls_forward + num_calls_backward,
            log_acc,
            x,
            y,
            ll_x,
            ll_y,
            is_valid_x,
            is_valid_y,
            state,
        )

        acc = jnp.mean(jnp.clip(jnp.exp(log_acc), a_max=1))
        return new_x, new_state, acc

    def select_sample(
        self,
        rng,
        num_calls,
        log_acc,
        current_sample,
        new_sample,
        current_ll,
        new_ll,
        is_valid_x,
        is_valid_y,
        sampler_state,
    ):
        y, new_state = super().select_sample(
            rng,
            log_acc,
            current_sample,
            new_sample,
            current_ll,
            new_ll,
            is_valid_x,
            is_valid_y,
            sampler_state,
        )
        new_state["num_ll_calls"] += num_calls
        return y, new_state

    def make_init_state(self, rng):
        state = super().make_init_state(rng)
        return state

    def get_local_dist(self, model, x, model_param):
        # Lazy initialization: create neighborhood_fn only once
        if not hasattr(self, "neighborhood_fn"):
            self.neighborhood_fn = model.logratio_in_neighborhood
        ll_x, logratio, num_calls, _, is_valid_x = self.neighborhood_fn(model_param, x)

        logits = self.apply_weight_function_logscale(logratio)
        if self.num_categories != 2:
            logits = logits * (1 - x) + x * -1e9
        log_prob = jax.nn.log_softmax(logits, -1)
        return ll_x, log_prob, num_calls, is_valid_x

    def proposal(self, model, rng, x, model_param, state):
        ll_x, log_prob, num_calls, is_valid_x = self.get_local_dist(model, x, model_param)

        if self.num_categories > 2:
            log_prob_all = jnp.reshape(log_prob, [log_prob.shape[0], -1, self.num_categories])
            log_prob = special.logsumexp(log_prob_all, axis=-1)
            log_prob_all = jax.nn.log_softmax(log_prob_all, axis=-1)
            x = jnp.reshape(x, log_prob_all.shape)

        selected_idx, ll_selected = math.multinomial(
            rng, log_prob, self.num_flips, replacement=False, return_ll=True, is_nsample_const=True
        )

        if self.num_categories > 2:
            val_logprob = log_prob_all[self.batch_rows, selected_idx]
            rng, _ = jax.random.split(rng)
            new_val = jax.random.categorical(rng, val_logprob)
            new_val = jax.nn.one_hot(new_val, self.num_categories)
        else:
            new_val = 1 - x[self.batch_rows, selected_idx]
        y = x.at[self.batch_rows, selected_idx].set(new_val)

        trajectory = {
            "ll_x2y": jnp.sum(ll_selected, axis=-1),
            "selected_idx": selected_idx,
        }
        return ll_x, y, trajectory, num_calls, is_valid_x

    def ll_y2x(self, model, x, model_param, forward_trajectory, y):
        ll_y, log_prob, num_calls, is_valid_y = self.get_local_dist(model, y, model_param)

        if self.num_categories > 2:
            log_prob_all = jnp.reshape(log_prob, [log_prob.shape[0], -1, self.num_categories])
            log_prob = special.logsumexp(log_prob_all, axis=-1)
            log_prob_all = jax.nn.log_softmax(log_prob_all, axis=-1)

        backwd_ll = jnp.take_along_axis(log_prob, forward_trajectory["selected_idx"], -1)
        ll_y2x_traj = math.noreplacement_sampling_renormalize(backwd_ll)
        if self.num_categories > 2:
            val_logprob = log_prob_all[
                jnp.expand_dims(jnp.arange(x.shape[0]), axis=1), forward_trajectory["selected_idx"]
            ]
            x = jnp.reshape(x, [x.shape[0], -1, self.num_categories])
            orig_val = x[
                jnp.expand_dims(jnp.arange(x.shape[0]), axis=1), forward_trajectory["selected_idx"]
            ]
            ll_val = jnp.sum(orig_val * val_logprob, axis=-1)

        ll_y2x = jnp.sum(ll_y2x_traj, axis=-1)
        if self.num_categories > 2:
            ll_y2x = ll_y2x + jnp.sum(ll_val, axis=-1)

        return ll_y, ll_y2x, num_calls, is_valid_y


def build_sampler(config):
    return LBPforILP(config)
