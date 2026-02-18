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
        self.proposal_type = self.config.get("proposal_type", "obj")
        self.chunk_size = self.config.get("chunk_size", 2000)

    def make_init_params(self, rng):
        try:
            data_list = next(self.datagen)
        except:
            return None
        return data_list

    def penalty(self, params, x):
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
        if self.formulation == "obj":
            return 0.0
        elif self.formulation == "obj_lagrangian":
            violation = Ax - ub + lb - Ax
        elif self.formulation == "max_linear":
            violation = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax)
        elif self.formulation == "max_linear_square":
            violation = jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax))
        penalty = self.penalty_coeff * jnp.sum(violation, axis=0)
        return penalty

    def penalty2(self, params, x):
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
        if self.proposal_type == "obj":
            return 0.0
        elif self.proposal_type == "uniform":
            return 0.0
        elif self.proposal_type == "obj_lagrangian":
            violation = Ax - ub + lb - Ax
        elif self.proposal_type == "max_linear":
            violation = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax)
        elif self.proposal_type == "max_linear_square":
            violation = jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax))
        elif self.proposal_type == "penalty_linear":
            violation = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax)
        elif self.proposal_type == "penalty_square":
            violation = jnp.square(jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax))

        penalty = self.penalty_coeff * jnp.sum(violation, axis=0)
        return penalty

    def objective(self, params, x):
        if self.config.graph_type == "ca":
            return jnp.dot(x, params["obj_coeffs"])
        elif self.config.graph_type == "sc":
            return -jnp.dot(x, params["obj_coeffs"])
        elif self.config.graph_type == "mis":
            return jnp.dot(x, params["obj_coeffs"])

    def logratio_in_neighborhood(self, params, x):
        # x shape: [batch, N]
        A = params["constraint_matrix"]  # [M, N]
        c = params["obj_coeffs"]  # [N]
        ub = params["constraint_rhs"]  # [M]
        lb = params["constraint_lhs"]  # [M]
        temp = params["temperature"]  # [1] or [batch] (pt)

        Ax = jnp.einsum("bn,nm->bm", x, A.T)  # current Ax [batch, M]
        obj_x = jnp.einsum("bn,n->b", x, c)  # current c^Tx [batch]
        if self.config.graph_type == "sc":
            obj_x = -obj_x
        v_curr = jnp.maximum(0, Ax - ub[None, :]) + jnp.maximum(0, lb[None, :] - Ax)  # [batch, M]
        penalty_x = self.penalty_coeff * jnp.sum(
            jnp.square(v_curr), axis=-1
        )  # current penalty [batch]
        ll_x = (obj_x - penalty_x) / temp  # current -energy [batch]

        # Deltas for flipping each bit j
        delta_x = 1 - 2 * x  # [batch, N]

        # Change in objective: c_j * delta_x_j
        delta_obj = c[None, :] * delta_x  # [batch, N]
        if self.config.graph_type == "sc":
            delta_obj = -delta_obj

        # Change in Penalty
        Ax_new = Ax[:, :, None] + A[None, :, :] * delta_x[:, None, :]  # [batch, M, N]
        v_new = jnp.maximum(0, Ax_new - ub[None, :, None]) + jnp.maximum(
            0, lb[None, :, None] - Ax_new
        )
        penalty_x_new = self.penalty_coeff * jnp.sum(jnp.square(v_new), axis=1)  # [batch, N]

        # Calculate Log-Ratios
        ll_x_new = (obj_x[:, None] + delta_obj - penalty_x_new) / temp[:, None]
        logratio = ll_x_new - ll_x[:, None]

        return ll_x, logratio, 1, self.get_neighbor_fn

    # def logratio_in_neighborhood(self, params, x, chunk_size=1000):
    #     # A: [M, N], x: [batch, N]
    #     A = params["constraint_matrix"]
    #     c = params["obj_coeffs"]
    #     ub = params["constraint_rhs"]
    #     lb = params["constraint_lhs"]
    #     temp = params["temperature"]

    #     batch_size, N = x.shape
    #     M = A.shape[0]

    #     # 1. Precompute current state (Same as before)
    #     Ax = jnp.einsum("bn,nm->bm", x, A.T)  # [batch, M]
    #     obj_x = jnp.einsum("bn,n->b", x, c)
    #     if self.config.graph_type == "sc":
    #         obj_x = -obj_x

    #     v_curr = jnp.maximum(0, Ax - ub) + jnp.maximum(0, lb - Ax)
    #     penalty_x = self.penalty_coeff * jnp.sum(jnp.square(v_curr), axis=-1)
    #     ll_x = (obj_x - penalty_x) / temp

    #     # 2. Prepare Data for Chunking
    #     # We pad N to be divisible by chunk_size to ensure static shapes for JIT
    #     remainder = N % chunk_size
    #     pad_len = (chunk_size - remainder) % chunk_size

    #     # Calculate all deltas at once [batch, N]
    #     delta_x_all = 1 - 2 * x
    #     delta_obj_all = c[None, :] * delta_x_all
    #     if self.config.graph_type == "sc":
    #         delta_obj_all = -delta_obj_all

    #     # Pad the arrays
    #     delta_x_padded = jnp.pad(delta_x_all, ((0, 0), (0, pad_len)))
    #     delta_obj_padded = jnp.pad(delta_obj_all, ((0, 0), (0, pad_len)))

    #     # Reshape to [num_chunks, batch, chunk_size] for scanning
    #     num_chunks = delta_x_padded.shape[1] // chunk_size

    #     delta_x_reshaped = delta_x_padded.reshape(batch_size, num_chunks, chunk_size).transpose(
    #         1, 0, 2
    #     )
    #     delta_obj_reshaped = delta_obj_padded.reshape(batch_size, num_chunks, chunk_size).transpose(
    #         1, 0, 2
    #     )

    #     # We also need to chunk A accordingly [M, N] -> [num_chunks, M, chunk_size]
    #     A_padded = jnp.pad(A, ((0, 0), (0, pad_len)))
    #     A_reshaped = A_padded.reshape(M, num_chunks, chunk_size).transpose(1, 0, 2)

    #     # 3. The Scan Function (Processes a block of 'chunk_size' neighbors)
    #     def scan_body(carry, inputs):
    #         # inputs: contains slices for this chunk
    #         dx_chunk, dobj_chunk, A_chunk = inputs
    #         # dx_chunk: [batch, chunk_size]
    #         # A_chunk:  [M, chunk_size]

    #         # Compute change in Ax for this chunk
    #         # [batch, 1, chunk] * [1, M, chunk] -> [batch, M, chunk]
    #         # (This is the memory-critical step. We strictly limit the 3rd dim to chunk_size)
    #         Ax_change = dx_chunk[:, None, :] * A_chunk[None, :, :]

    #         # Ax is [batch, M]. We need to broadcast it against the chunk dim
    #         Ax_new = Ax[:, :, None] + Ax_change

    #         # Re-compute penalty
    #         v_new = jnp.maximum(0, Ax_new - ub[:, None]) + jnp.maximum(0, lb[:, None] - Ax_new)
    #         penalty_chunk = self.penalty_coeff * jnp.sum(
    #             jnp.square(v_new), axis=1
    #         )  # Sum over M -> [batch, chunk]

    #         # Calculate Log Ratios
    #         ll_new = (obj_x[:, None] + dobj_chunk - penalty_chunk) / temp[:, None]
    #         logratio_chunk = ll_new - ll_x[:, None]

    #         return carry, logratio_chunk

    #     # 4. Run Scan
    #     _, logratios_padded = jax.lax.scan(
    #         scan_body, None, (delta_x_reshaped, delta_obj_reshaped, A_reshaped)
    #     )

    #     # logratios_padded is [num_chunks, batch, chunk_size]
    #     # Reshape back to [batch, N_padded]
    #     logratios = logratios_padded.transpose(1, 0, 2).reshape(batch_size, -1)

    #     # Remove padding
    #     logratios = logratios[:, :N]

    #     return ll_x, logratios, 1, self.get_neighbor_fn


def build_model(config):
    return ILP(config)
