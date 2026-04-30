import jax
import jax.numpy as jnp


def _vectorized_swap_step(new_x, P_x, temperature, penalty_coeffs, indices_a, indices_b, rng_key):
    penalty_coeff_a, penalty_coeff_b = penalty_coeffs[indices_a], penalty_coeffs[indices_b]
    P_a, P_b = P_x[indices_a]/penalty_coeff_b, P_x[indices_b]/penalty_coeff_a # P(x)= \Sum (Ax-b)

    log_acceptance_ratio = (1/temperature) * (penalty_coeff_b - penalty_coeff_a) * (P_b - P_a) # \beta * \Delata \lambda * (P_b-P_a)
    # shape: (num_swaps, batch_size)
    acceptance_ratio = jnp.exp(log_acceptance_ratio)
    is_accept = jax.random.uniform(rng_key, shape=log_acceptance_ratio.shape) < acceptance_ratio
    swapped_a = jnp.where(is_accept[..., None], new_x[indices_b], new_x[indices_a])
    swapped_b = jnp.where(is_accept[..., None], new_x[indices_a], new_x[indices_b])

    new_x = new_x.at[indices_a].set(swapped_a)
    new_x = new_x.at[indices_b].set(swapped_b)
    return new_x, jnp.minimum(1.0, acceptance_ratio)


def swap_samples_deo(new_x, logprob, obj_coeff, temperature, penalty_coeffs, rng_key, current_step):
    # define the indices of the even and odd penalty_coeffs
    rng_key, swap_subkey = jax.random.split(rng_key, 2)
    num_penalty_coeffs = penalty_coeffs.shape[0]
    indices_a = jnp.arange(0, num_penalty_coeffs - 1, 2) + current_step % 2
    indices_b = jnp.arange(1, num_penalty_coeffs, 2) + current_step % 2
    c_x = new_x @ obj_coeff
    
    P_x = -(logprob -c_x)  #(-1)(logprob-c^Tx) = (-1)-\lambda  * \Sum (Ax-b)

    new_x, acceptance_ratio = _vectorized_swap_step(
        new_x, P_x, temperature, penalty_coeffs, indices_a, indices_b, swap_subkey
    )
    return new_x, acceptance_ratio, indices_a, indices_b


def swap_samples_seo(new_x, logprob, obj_coeff, temperature, penalty_coeffs, rng_key, current_step):
    # define the indices of the even and odd penalty_coeffs
    rng_key, choice_subkey, swap_subkey = jax.random.split(rng_key, 3)

    do_odd_swap = jax.random.bernoulli(choice_subkey)

    num_penalty_coeffs = penalty_coeffs.shape[0]
    indices_a = jnp.arange(0, num_penalty_coeffs - 1, 2) + do_odd_swap
    indices_b = jnp.arange(1, num_penalty_coeffs, 2) + do_odd_swap

    new_x, acceptance_ratio = _vectorized_swap_step(
        new_x, logprob, temperature, penalty_coeffs, indices_a, indices_b, swap_subkey
    )
    return new_x, acceptance_ratio, indices_a, indices_b
