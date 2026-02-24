from discs.models import comb_ebm
import jax
import jax.numpy as jnp
import ml_collections


class ILP(comb_ebm.BinaryNodeCombEBM):
    """Max Independent Set model."""

    def __init__(self, config: ml_collections.ConfigDict):
        super().__init__(config)
        self.config = config.model
        self.max_num_nodes = self.config.max_num_nodes
        self.penalty_coeff = self.config.get("penalty", 2.0)
        self.formulation = self.config.get("formulation", "max_linear")
        self.chunk_size = self.config.get("chunk_size", 2000)

    def make_init_params(self, rng):
        try:
            data_list = next(self.datagen)
        except:
            return None
        return data_list

    def penalty(self, params, x):
        # TODO: use scan
        batch_size = x.shape[0]
        num_vars = x.shape[-1]
        num_constraints = params["constraint_matrix"].shape[0]
        if num_vars > self.chunk_size:
            Ax = jnp.zeros((num_constraints, batch_size), dtype=jnp.float32)
            for i in range(0, num_vars, self.chunk_size):
                x_chunk = x[..., i : i + self.chunk_size]
                Ax_chunk = jnp.dot(
                    params["constraint_matrix"][:, i : i + self.chunk_size], x_chunk.T
                )
                Ax += Ax_chunk
        else:
            Ax = jnp.dot(params["constraint_matrix"], x.T)
        ub = params["constraint_rhs"][:, None]
        lb = params["constraint_lhs"][:, None]
        if self.formulation == "obj" or self.formulation == "lagrangian":
            return jnp.zeros(batch_size)  # Ax - lb + ub - Ax is constant w.r.t. x
        elif self.formulation == "max_linear":
            violation = jnp.maximum(0, jnp.maximum(Ax - ub, lb - Ax))
        elif self.formulation == "max_linear_square":
            violation = jnp.square(jnp.maximum(0, jnp.maximum(Ax - ub, lb - Ax)))
        penalty = self.penalty_coeff * jnp.sum(violation, axis=0)
        return penalty

    def objective(self, params, x):
        obj = jnp.dot(x, params["obj_coeffs"])
        if self.config.graph_type == "sc":
            obj = -obj
        return obj

    def logratio_in_neighborhood(self, params, x, m_chunk_size=1000):
        A = params["constraint_matrix"]  # [M, N]
        c = params["obj_coeffs"]  # [N]
        ub = params["constraint_rhs"]  # [M]
        lb = params["constraint_lhs"]  # [M]
        temp = params["temperature"]  # [batch] or [1]

        batch_size, N = x.shape
        M = A.shape[0]
        m_chunk = min(m_chunk_size, M)

        Ax = x @ A.T  # [batch, M]
        obj_x = x @ c  # [batch]
        if self.config.graph_type == "sc":
            obj_x = -obj_x

        v_curr = jnp.maximum(0, jnp.maximum(Ax - ub, lb - Ax))  # [batch, M]
        if self.formulation == "obj" or self.formulation == "lagrangian":
            penalty_x = jnp.zeros(batch_size)
            is_valid_x = jnp.sum(v_curr, axis=-1) <= 1e-3  # [batch]  # 1e-3 is a small threshold
        else:
            if self.formulation == "max_linear":
                penalty_x = self.penalty_coeff * jnp.sum(v_curr, axis=-1)
            elif self.formulation == "max_linear_square":
                penalty_x = self.penalty_coeff * jnp.sum(jnp.square(v_curr), axis=-1)
            is_valid_x = penalty_x <= 1e-3  # [batch]  # 1e-3 is a small threshold

        ll_x = (obj_x - penalty_x) / temp  # [batch]

        delta_x = 1 - 2 * x  # [batch, N]
        delta_obj = c[None, :] * delta_x  # [batch, N]
        if self.config.graph_type == "sc":
            delta_obj = -delta_obj

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
            penalty_new = self.penalty_coeff * penalty_new

        ll_new = (obj_x[:, None] + delta_obj - penalty_new) / temp[:, None]
        logratios = ll_new - ll_x[:, None]

        return ll_x, logratios, 1, self.get_neighbor_fn, is_valid_x


def build_model(config):
    return ILP(config)
