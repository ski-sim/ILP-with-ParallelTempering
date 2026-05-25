import os
import re
import jax
import jax.numpy as jnp
import numpy as np
from pyscipopt import Model


def get_constraint_matrix(model):
    vars = model.getVars()
    cons = model.getConss()
    var_idx = {v.name: i for i, v in enumerate(vars)}

    A = np.zeros((len(cons), len(vars)))
    lhs = np.zeros(len(cons))
    rhs = np.zeros(len(cons))

    for i, cons in enumerate(cons):
        lhs[i] = model.getLhs(cons)
        rhs[i] = model.getRhs(cons)
        for var_name, coeff in model.getValsLinear(cons).items():
            A[i, var_idx[var_name]] = coeff

    return A, rhs, lhs


def get_obj_coefficients(expr, num_vars):
    coeffs_dict = {}
    for term, coeff in expr.terms.items():
        var_name = str(term).split("(")[1].split(")")[0]  # Term(x0) -> x0
        nums = re.findall(r"\d+", var_name)
        if nums:
            var_index = int(nums[0])
            coeffs_dict[var_index] = float(coeff)

    coeffs = []
    for i in range(num_vars):
        if i in coeffs_dict:
            coeffs.append(coeffs_dict[i])
        else:
            coeffs.append(0.0)

    return coeffs


def get_variable_types(model, num_vars):
    """Extract variable types from SCIP model.

    Returns:
        var_types: numpy array of shape (num_vars,)
            - 0: BINARY
            - 1: INTEGER
            - 2: IMPLINT (implicit integer)
            - 3: CONTINUOUS
    """
    var_types = np.full(num_vars, 3, dtype=np.int32)
    type_map = {"BINARY": 0, "INTEGER": 1, "CONTINUOUS": 3}
    for i, var in enumerate(model.getVars()):
        if i >= num_vars:
            break
        var_types[i] = type_map.get(var.vtype(), 3)
    return var_types


def get_variable_bounds(model, num_vars):
    """Extract per-variable lower/upper bounds from SCIP model.

    Returns:
        lbs, ubs: numpy arrays of shape (num_vars,). np.inf / -np.inf when
        SCIP reports an unbounded side. Padded entries get (-inf, +inf).
    """
    vars = model.getVars()
    num_vars = len(vars)
    lbs = np.full(num_vars, -np.inf, dtype=np.float64)
    ubs = np.full(num_vars, np.inf, dtype=np.float64)
    for i, var in enumerate(vars):
        if i >= num_vars:
            break
        lb = var.getLbOriginal()
        ub = var.getUbOriginal()
        # SCIP uses +/-1e20 as sentinel for infinity
        lbs[i] = -np.inf if lb <= -1e20 else float(lb)
        ubs[i] = np.inf if ub >= 1e20 else float(ub)
    return lbs, ubs


class ILPGen:
    """Generator for ILP instances."""

    def __init__(self, data_root, model_config):
        super().__init__()
        data_folder = os.path.join(
            data_root, f"{model_config.instance_name}_{model_config.max_num_vars}"
        )
        if not os.path.isdir(data_folder):
            print(f"Warning: data folder {data_folder} does not exist")

        file_list = []
        for fname in os.listdir(data_folder):
            if fname.endswith(".mps") or fname.endswith(".lp"):
                file_list.append(os.path.join(data_folder, fname))
        self.file_list = sorted(file_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        if model_config.num_instances > len(self.file_list):
            print(
                f"Warning: num_instances {model_config.num_instances} is greater than the number of instances {len(self.file_list)}"
            )
            model_config.num_instances = len(self.file_list)
        else:
            self.file_list = self.file_list[: model_config.num_instances]

        assert model_config.num_instances > 0
        assert model_config.max_num_vars > 0
        assert model_config.max_num_cons > 0
        
        self.num_instances = model_config.num_instances
        self.max_num_vars = model_config.max_num_vars
        self.max_num_cons = model_config.max_num_cons

        print("max num vars", self.max_num_vars)
        print("max num cons", self.max_num_cons)
        print("num instances", self.num_instances)

    def sample_gen(self, phase, repeat=False):
        assert phase == "test"
        while True:
            for fname in self.file_list:
                model = Model()
                model.readProblem(fname)
                yield model
            if not repeat:
                break

    def get_iterator(self, phase, batch_size, sharding=False):
        """Get sharded/distributed data loader."""
        num_proc = 1
        proc_idx = 0
        local_batch_size = batch_size
        if sharding:
            proc_idx = jax.process_index()
            num_proc = jax.process_count()
            assert batch_size % num_proc == 0
            local_batch_size = batch_size // num_proc
        generator = self.sample_gen(phase, repeat=False)
        buffer = []
        for idx, m in enumerate(generator):
            if idx % num_proc == proc_idx:
                try:
                    num_vars = m.getNVars()
                    obj_coeff = get_obj_coefficients(m.getObjective(), num_vars)
                    cons_m, rhs, lhs = get_constraint_matrix(m)
                    var_types = get_variable_types(m, self.max_num_vars)
                    var_lbs, var_ubs = get_variable_bounds(m, self.max_num_vars)
                    params = {
                        "mask": jnp.ones(self.max_num_vars),
                        "temperature": 1.0,
                        "obj_coeffs": jnp.array(obj_coeff),
                        "constraint_matrix": jnp.array(cons_m),
                        "constraint_rhs": jnp.array(rhs),
                        "constraint_lhs": jnp.array(lhs),
                        "var_types": jnp.array(var_types),
                        "var_lbs": jnp.array(var_lbs),
                        "var_ubs": jnp.array(var_ubs),
                    }
                except Exception as e:
                    print(f"Warning: Could not process MILP model {idx}: {e}")
                    raise
                buffer.append((idx, params))
                if len(buffer) == local_batch_size:
                    yield buffer
                    buffer = []

def get_instances(config):
    """Get ILP instance loader."""
    if config.model.instance_name in [
        'sc', 'ca', 'mis', 'mvc',
    ]:
        return ILPGen(config.model.data_root, config.model)
    raise ValueError('Unknown instance name %s' % config.model.instance_name)
