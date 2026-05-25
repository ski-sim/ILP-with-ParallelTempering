"""ILP model for combinatorial optimization."""

from PT_ILP.common.utils import get_datagen
import jax
import jax.numpy as jnp
import ml_collections


class ILP:
    """ILP model with linear objective and linear constraints."""

    def __init__(self, config: ml_collections.ConfigDict):
        self.datagen = get_datagen(config)
        self.config = config.model
        self.max_num_vars = self.config.max_num_vars
        self.config_experiment = config.experiment
        if self.config_experiment.t_schedule in ["pen_pt", "pen_pt_exp_decay"]:
            # per-chain penalty ladder, shape [batch_size]
            self.penalty_coeff = jnp.geomspace(
                self.config_experiment.l_min, self.config_experiment.l_max,
                num=self.config_experiment.batch_size,
            )
        else:
            # scalar (0-d array) shared across the batch
            self.penalty_coeff = jnp.asarray(self.config.get("penalty", 2.0))
        self.formulation = self.config.get("formulation", "max_linear")
        self.chunk_size = self.config.get("chunk_size", 1000)

        self.obj_sign = -1.0 if self.config.instance_name in ("sc", "mvc") else 1.0

    def make_init_params(self, rng):
        try:
            data_list = next(self.datagen)
        except StopIteration:
            return None
        return data_list

    def get_init_samples(self, rng, num_samples):
        return jax.random.bernoulli(
            key=rng, p=0.5, shape=(num_samples, self.max_num_vars)
        ).astype(jnp.int32)

    def get_neighbor_fn(self, _, x, neighbhor_idx):
        brange = jnp.arange(x.shape[0])
        cur_val = x[brange, neighbhor_idx]
        y = x.at[brange, neighbhor_idx].set(1 - cur_val)
        return y

    def objective(self, params, x):
        return self.obj_sign * jnp.dot(x, params["obj_coeffs"])

    def penalty(self, params, x):
        batch_size = x.shape[0]
        if self.formulation == "obj" or self.formulation == "lagrangian":
            return jnp.zeros(batch_size)

        A = params["constraint_matrix"]  # [M, N]
        ub = params["constraint_rhs"]  # [M]
        lb = params["constraint_lhs"]  # [M]
        M = A.shape[0]

        m_chunk = min(self.chunk_size, M)
        pad_m = (-M) % m_chunk
        if pad_m > 0:
            A = jnp.pad(A, ((0, pad_m), (0, 0)))
            ub = jnp.pad(ub, (0, pad_m), constant_values=jnp.inf)
            lb = jnp.pad(lb, (0, pad_m), constant_values=-jnp.inf)

        num_m_chunks = (M + pad_m) // m_chunk
        A_scan = A.reshape(num_m_chunks, m_chunk, -1)
        ub_scan = ub.reshape(num_m_chunks, m_chunk)
        lb_scan = lb.reshape(num_m_chunks, m_chunk)

        def scan_body(penalty_acc, inputs):
            A_chunk, ub_chunk, lb_chunk = inputs
            Ax_chunk = jnp.dot(A_chunk, x.T)  # [m_chunk, batch]
            v = jnp.maximum(
                0, jnp.maximum(Ax_chunk - ub_chunk[:, None], lb_chunk[:, None] - Ax_chunk)
            )
            if self.formulation == "max_linear_square":
                v = jnp.square(v)
            return penalty_acc + jnp.sum(v, axis=0), None

        penalty, _ = jax.lax.scan(scan_body, jnp.zeros(batch_size), (A_scan, ub_scan, lb_scan))
        return self.penalty_coeff * penalty

    def forward(self, params, x):
        x = x.astype(jnp.float32)
        obj = self.objective(params, x)
        penalty = self.penalty(params, x)
        isvalid = penalty <= 1e-3
        return (obj - penalty) / params["temperature"], isvalid

    def get_value_and_grad(self, params, x):
        # x.shape: [batch, num_vars]
        x = x.astype(jnp.float32)  # int tensor is not differentiable

        def fun(z):
            loglikelihood, isvalid = self.forward(params, z)
            return jnp.sum(loglikelihood), (loglikelihood, isvalid)

        (_, (loglikelihood, isvalid)), grad = jax.value_and_grad(fun, has_aux=True)(x)
        return grad, (loglikelihood, isvalid)

    def logratio_in_neighborhood(self, params, x):
        A = params["constraint_matrix"]  # [M, N]
        c = params["obj_coeffs"]  # [N]
        ub = params["constraint_rhs"]  # [M]
        lb = params["constraint_lhs"]  # [M]
        temp = params["temperature"]  # [batch] or [1]

        batch_size, N = x.shape
        M = A.shape[0]
        m_chunk = min(self.chunk_size, M)

        Ax = x @ A.T  # [batch, M]
        obj_x = self.obj_sign * (x @ c)  # [batch]

        v_curr = jnp.maximum(0, jnp.maximum(Ax - ub, lb - Ax))  # [batch, M]
        if self.formulation == "obj" or self.formulation == "lagrangian":
            penalty_x = jnp.zeros(batch_size)
            is_valid_x = jnp.sum(v_curr, axis=-1) <= 1e-3  # [batch]  # 1e-3 is a small threshold
        else:
            if self.formulation == "max_linear":
                penalty_x = self.penalty_coeff * jnp.sum(v_curr, axis=-1) # \Sum (Ax-b)
            elif self.formulation == "max_linear_square":
                penalty_x = self.penalty_coeff * jnp.sum(jnp.square(v_curr), axis=-1) # \Sum (Ax-b)^2
            is_valid_x = penalty_x <= 1e-3  # [batch]  # 1e-3 is a small threshold
        ll_x = (obj_x - penalty_x) / temp  # [batch]
        delta_x = 1 - 2 * x  # [batch, N]
        delta_obj = self.obj_sign * c[None, :] * delta_x  # [batch, N]

        if self.formulation == "obj" or self.formulation == "lagrangian":
            penalty_new = jnp.zeros((batch_size, N))
        else:
            # Scan over M (constraints)
            excess_upper = Ax - ub[None, :]  # [batch, M]
            excess_lower = lb[None, :] - Ax  # [batch, M]

            pad_m = (-M) % m_chunk
            if pad_m > 0:
                A_padded = jnp.pad(A, ((0, pad_m), (0, 0)))
                excess_upper = jnp.pad(excess_upper, ((0, 0), (0, pad_m)), constant_values=-jnp.inf)
                excess_lower = jnp.pad(excess_lower, ((0, 0), (0, pad_m)), constant_values=-jnp.inf)
            else:
                A_padded = A

            num_m_chunks = (M + pad_m) // m_chunk
            A_scan = A_padded.reshape(num_m_chunks, m_chunk, N)  # [num_m_chunks, m_chunk, N]
            eu_scan = excess_upper.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
            el_scan = excess_lower.reshape(batch_size, num_m_chunks, m_chunk).transpose(1, 0, 2)
            # [num_m_chunks, batch, m_chunk] each

            def scan_body(penalty_acc, inputs):
                A_chunk, eu_chunk, el_chunk = inputs
                # A_chunk: [m_chunk, N], eu/el_chunk: [batch, m_chunk]
                shift = A_chunk[None, :, :] * delta_x[:, None, :]  # [batch, m_chunk, N]
                v_new = jnp.maximum(
                    0, jnp.maximum(eu_chunk[:, :, None] + shift, el_chunk[:, :, None] - shift)
                )
                if self.formulation == "max_linear":
                    return penalty_acc + jnp.sum(v_new, axis=1), None
                else:  # formulation == "max_linear_square"
                    return penalty_acc + jnp.sum(jnp.square(v_new), axis=1), None

            penalty_new, _ = jax.lax.scan(
                scan_body, jnp.zeros((batch_size, N)), (A_scan, eu_scan, el_scan)
            )
            penalty_new = jnp.asarray(self.penalty_coeff)[..., None] * penalty_new

        ll_new = (obj_x[:, None] + delta_obj - penalty_new) / temp[:, None]
        logratios = ll_new - ll_x[:, None]
        return ll_x, logratios, 1, self.get_neighbor_fn, is_valid_x


def build_model(config):
    return ILP(config)
