import jax
import jax.numpy as jnp


def _vectorized_swap_step(
    new_x, logprob, temperatures, indices_a, indices_b, rng_key
):
    logprob_a, logprob_b = logprob[indices_a], logprob[indices_b]
    temp_a, temp_b = temperatures[indices_a], temperatures[indices_b]
    
    log_acceptance_ratio = (1.0 / temp_a - 1.0 / temp_b) * (logprob_b - logprob_a) # log_acceptance_ratio shape: (num_swaps, batch_size)
    
    rng_key, subkey = jax.random.split(rng_key)
    is_accept = jax.random.uniform(subkey, shape=log_acceptance_ratio.shape) < jnp.exp(log_acceptance_ratio) # is_accept shape: (num_swaps, batch_size)

    swapped_a = jnp.where(is_accept[..., None], new_x[indices_b], new_x[indices_a])
    swapped_b = jnp.where(is_accept[..., None], new_x[indices_a], new_x[indices_b])
    
    new_x = new_x.at[indices_a].set(swapped_a)
    new_x = new_x.at[indices_b].set(swapped_b)
    return new_x, is_accept, rng_key 



def swap_samples_deo(new_x, logprob, temperatures, rng_key, current_step):
    # define the indices of the even and odd temperatures
    num_temperatures = temperatures.shape[0]
    even_indices_a = jnp.arange(0, num_temperatures - 1, 2)
    even_indices_b = jnp.arange(1, num_temperatures, 2)
    
    odd_indices_a = jnp.arange(1, num_temperatures - 1, 2)
    odd_indices_b = jnp.arange(2, num_temperatures, 2)
    
    rng_key, choice_subkey, swap_subkey = jax.random.split(rng_key, 3)
    do_even_swap = (current_step % 2 == 0)
    def perform_even_swap(operand_x):
        return _vectorized_swap_step( operand_x, logprob, temperatures, even_indices_a, even_indices_b, swap_subkey )
    def perform_odd_swap(operand_x):
        return _vectorized_swap_step( operand_x, logprob, temperatures, odd_indices_a, odd_indices_b, swap_subkey )
    
    res = jax.lax.cond(do_even_swap, perform_even_swap, perform_odd_swap, new_x)
    new_x, is_accept, new_rng_key = res

    return new_x, is_accept

def swap_samples_seo(new_x, logprob, temperatures, rng_key, current_step):
    # define the indices of the even and odd temperatures
    num_temperatures = temperatures.shape[0]
    even_indices_a = jnp.arange(0, num_temperatures - 1, 2)
    even_indices_b = jnp.arange(1, num_temperatures, 2)
    
    odd_indices_a = jnp.arange(1, num_temperatures - 1, 2)
    odd_indices_b = jnp.arange(2, num_temperatures, 2)
    
    rng_key, choice_subkey, swap_subkey = jax.random.split(rng_key, 3)

    do_even_swap = jax.random.bernoulli(choice_subkey)

    def perform_even_swap(operand_x):
        return _vectorized_swap_step( operand_x, logprob, temperatures, even_indices_a, even_indices_b, swap_subkey )
    def perform_odd_swap(operand_x):
        return _vectorized_swap_step( operand_x, logprob, temperatures, odd_indices_a, odd_indices_b, swap_subkey )

    res = jax.lax.cond(do_even_swap, perform_even_swap, perform_odd_swap, new_x)
    new_x, is_accept, new_rng_key = res

    return new_x, is_accept