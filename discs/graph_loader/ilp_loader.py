"""Load MIS graphs."""

import os
import jax.numpy as jnp
import numpy as np
import pickle5 as pickle
from pyscipopt import Model
import jax
import re


def get_constraint_matrix(model):
    variables = model.getVars()
    constraints = model.getConss()
    var_idx = {v.name: i for i, v in enumerate(variables)}

    A = np.zeros((len(constraints), len(variables)))
    lhs = np.zeros(len(constraints))
    rhs = np.zeros(len(constraints))

    for i, cons in enumerate(constraints):
        lhs[i] = model.getLhs(cons)
        rhs[i] = model.getRhs(cons)
        for var_name, coef in model.getValsLinear(cons).items():
            A[i, var_idx[var_name]] = coef

    return A, rhs, lhs


def get_obj_coefficients(expr, num_vars):
    coeffs_dict = {}
    for term, coeff in expr.terms.items():
        var_name = str(term).split("(")[1].split(")")[0]  # Term(x0) -> x0
        numbers = re.findall(r"\d+", var_name)
        if numbers:
            var_index = int(numbers[0])
            coeffs_dict[var_index] = float(coeff)

    coefficients = []
    for i in range(num_vars):
        if i in coeffs_dict:
            coefficients.append(coeffs_dict[i])
        else:
            coefficients.append(0.0)

    return coefficients


def get_variable_types(model, max_num_nodes):
    """Extract variable types from SCIP model.

    Returns:
        var_types: numpy array of shape (max_num_nodes,)
            - 0: BINARY
            - 1: INTEGER
            - 2: IMPLINT (implicit integer)
            - 3: CONTINUOUS
    """
    variables = model.getVars()
    num_vars = len(variables)
    # SCIP variable type mapping
    # 'B' -> 0 (BINARY), 'I' -> 1 (INTEGER), 'C' -> 3 (CONTINUOUS)
    var_types = np.zeros(max_num_nodes, dtype=np.int32)

    for i, var in enumerate(variables):
        if i >= max_num_nodes:
            break
        vtype = var.vtype()
        if vtype == "BINARY":
            var_types[i] = 0  # BINARY
        elif vtype == "INTEGER":
            var_types[i] = 1  # INTEGER
        elif vtype == "CONTINUOUS":
            var_types[i] = 3  # CONTINUOUS
        else:
            # Default to continuous for unknown types
            var_types[i] = 3

    # Pad remaining variables as continuous (default)
    for i in range(num_vars, max_num_nodes):
        var_types[i] = 3

    return var_types


class ILPGraphGen:
    """Generator for ILP graphs."""

    def __init__(self, data_root, model_config):
        super().__init__()
        data_folder = os.path.join(
            data_root, "%s_%d" % (model_config.graph_type, model_config.max_num_nodes)
        )
        if not os.path.exists(data_folder):
            print(f"Warning: Data folder {data_folder} does not exist")

        file_list = []
        for fname in os.listdir(data_folder):
            if fname.endswith(".mps") or fname.endswith(".lp"):
                file_list.append(os.path.join(data_folder, fname))
        self.file_list = sorted(file_list, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        if (
            model_config.max_num_nodes > 0
            and model_config.max_num_constraints > 0
            and model_config.num_instances > 0
        ):
            self._max_num_nodes = model_config.max_num_nodes
            self._max_num_constraints = model_config.max_num_constraints
            self._num_instances = model_config.num_instances
        else:
            for fname in self.file_list:
                with open(fname, "rb") as f:
                    g = pickle.load(f)
                    self._max_num_nodes = max(self._max_num_nodes, len(g))
                    self._max_num_constraints = max(self._max_num_constraints, len(g.edges()))
                    self._num_instances += 1
        print("max num nodes", self.max_num_nodes)
        print("max num constraints", self.max_num_constraints)
        print("num instances", self.num_instances)

    def sample_gen(self, phase, repeat=False):
        assert phase == "test"
        while True:
            for fname in self.file_list:
                model = Model()
                model.readProblem(fname)
                # Use index as graph identifier
                obj = 0
                yield model, obj
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
        total_batches = self.num_instances // batch_size
        if self.num_instances % batch_size != 0:
            total_batches += 1
        generator = self.sample_gen(
            phase, repeat=False
        )  # graphs generator outpust the graph and obj
        num_batches = 0
        buffer = []
        for idx, (g, sol) in enumerate(generator):
            if idx % num_proc == proc_idx:
                try:
                    num_vars = g.getNVars()
                    obj_coefficients = get_obj_coefficients(g.getObjective(), num_vars)
                    constraint_matrix, rhs, lhs = get_constraint_matrix(g)
                    var_types = get_variable_types(g, self._max_num_nodes)
                    params = {
                        "mask": jnp.ones(self._max_num_nodes),
                        "temperature": 1.0,
                        # 'model': g,  # model을 params에 포함
                        "obj_coeffs": jnp.array(obj_coefficients),
                        "constraint_matrix": jnp.array(constraint_matrix),  # 제약식 행렬
                        "constraint_rhs": jnp.array(rhs),  # 제약식 우변
                        "constraint_lhs": jnp.array(lhs),  # 제약식 좌변
                        "var_types": jnp.array(
                            var_types
                        ),  # 변수 타입: 0=BINARY, 1=INTEGER, 3=CONTINUOUS
                    }
                except Exception as e:
                    print(f"Warning: Could not process MILP model {idx}: {e}")
                    raise e
                buffer.append((idx, params, sol))  # (index, params, solution) 형태로 반환
                if len(buffer) == local_batch_size:
                    num_batches += 1
                    yield buffer
                    buffer = []

    @property
    def num_instances(self):
        """Get the number of available instances."""
        return self._num_instances

    @property
    def max_num_nodes(self):
        """Get the maximum number of nodes (variables)."""
        return self._max_num_nodes

    @property
    def max_num_constraints(self):
        """Get the maximum number of edges (constraints)."""
        return self._max_num_constraints
