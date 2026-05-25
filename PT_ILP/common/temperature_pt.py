import jax
import jax.numpy as jnp


def _vectorized_swap_step(new_x, logprob, temperatures, indices_a, indices_b, rng_key):
    logprob_a, logprob_b = logprob[indices_a], logprob[indices_b]
    temp_a, temp_b = temperatures[indices_a], temperatures[indices_b]

    log_acc = (1.0 / temp_a - 1.0 / temp_b) * (logprob_b - logprob_a)
    # shape: (num_swaps, batch_size)
    acc = jnp.exp(log_acc)
    is_accept = jax.random.uniform(rng_key, shape=log_acc.shape) < acc

    swapped_a = jnp.where(is_accept[..., None], new_x[indices_b], new_x[indices_a])
    swapped_b = jnp.where(is_accept[..., None], new_x[indices_a], new_x[indices_b])

    new_x = new_x.at[indices_a].set(swapped_a)
    new_x = new_x.at[indices_b].set(swapped_b)
    return new_x, jnp.minimum(1.0, acc)


def swap_samples_deo(new_x, logprob, temperatures, rng_key, current_step):
    _, swap_subkey = jax.random.split(rng_key, 2)
    ladder_size = temperatures.shape[0]
    indices_a = jnp.arange(0, ladder_size - 1, 2) + current_step % 2
    indices_b = jnp.arange(1, ladder_size, 2) + current_step % 2

    new_x, acc = _vectorized_swap_step(
        new_x, logprob, temperatures, indices_a, indices_b, swap_subkey
    )
    return new_x, acc, indices_a, indices_b


def swap_samples_seo(new_x, logprob, temperatures, rng_key, current_step):
    _, idx_subkey, swap_subkey = jax.random.split(rng_key, 3)

    swap_parity = jax.random.bernoulli(idx_subkey)

    ladder_size = temperatures.shape[0]
    indices_a = jnp.arange(0, ladder_size - 1, 2) + swap_parity
    indices_b = jnp.arange(1, ladder_size, 2) + swap_parity

    new_x, acc = _vectorized_swap_step(
        new_x, logprob, temperatures, indices_a, indices_b, swap_subkey
    )
    return new_x, acc, indices_a, indices_b


def swap_samples_reversible(new_x, logprob, temperatures, rng_key, current_step):
    ladder_size = temperatures.shape[0]
    swap_keys = jax.random.split(rng_key, ladder_size - 1)

    acc_list = []
    indices_a_list = []
    indices_b_list = []
    for k, i in enumerate(range(ladder_size - 1, 0, -1)):
        new_x, swap_rate = _vectorized_swap_step(
            new_x, logprob, temperatures, i, i - 1, swap_keys[k]
        )
        acc_list.append(swap_rate)
        indices_a_list.append(i)
        indices_b_list.append(i - 1)

    # deo와 동일한 반환 형태: (num_swaps, batch) 텐서 + (num_swaps,) 인덱스 배열
    acc = jnp.stack(acc_list, axis=0)
    indices_a = jnp.array(indices_a_list)
    indices_b = jnp.array(indices_b_list)
    return new_x, acc, indices_a, indices_b
