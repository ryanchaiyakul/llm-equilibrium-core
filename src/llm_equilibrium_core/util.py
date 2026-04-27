from pathlib import Path
from dataclasses import dataclass
import jax
import jax.numpy as jnp
import jax.flatten_util
import pandas as pd
import numpy as np
import equinox as eqx

import dismech_jax as djx
from sklearn.neighbors import NearestNeighbors

from .triplet_model import TripletModel


@dataclass
class TrainConfig:
    # Rod config
    length: float = 0.1
    radius: float = 1e-3
    density: float = 1e3
    youngs_mod: float = 1e6
    N: int = 5
    idx_b: jax.Array | None = None

    # Loss config
    S_factor: float = 0.1

    # Training config
    lr: float = 1e-3
    epochs: int = 1000
    print_every: int = 100
    verbose: bool = True


object_map = {
    "slinky": {"length": 0.2, "mass": 30e-3},
    "strip": {"length": 0.33, "mass": 7e-3},
    "brizier": {"length": 0.35, "mass": 10e-3},
    "tape": {"length": 0.3, "mass": 10e-3},
}


def load_csv(filepath: Path | str, is_2d: bool = False) -> np.ndarray:
    """Load output of CSV from llm_equilibrium tool."""
    df = pd.read_csv(filepath)

    # Extract markers
    coord_cols = [
        c
        for c in df.columns
        if "marker_" in c and any(a in c for a in ["_x", "_y", "_z"])
    ]

    def sort_key(name):
        parts = name.split("_")
        return (int(parts[1]), {"x": 0, "y": 1, "z": 2}[parts[2]])

    sorted_cols = sorted(coord_cols, key=sort_key)
    trajectories = np.array(df[sorted_cols].values).reshape(len(df), -1, 3)

    # 4. Rotate for gravity = Z-Down
    alignment_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
    trajectories = trajectories @ alignment_matrix.T

    # Center at (0,0,0)
    root_origin = trajectories[:, 0, :]
    trajectories = trajectories - root_origin[:, None, ...]

    if is_2d:
        trajectories = trajectories[..., [0, 2]]

    return trajectories


class Dataset(eqx.Module):
    qs: jax.Array
    S: jax.Array
    length: float
    mass: float

    @classmethod
    def from_npz(cls, filepath: Path | str):
        data = np.load(filepath)
        return cls(
            qs=jnp.asarray(data.get("qs", [])),
            S=jnp.asarray(data.get("S", [])),
            mass=data.get("mass", 0.0),
            length=data.get("length", 0.1),
        )

    @classmethod
    def from_csv(cls, filepath: Path | str, is_2d: bool = True, base_l2_reg=1e2):
        filepath = Path(filepath)
        obj_info = None
        for name, specs in object_map.items():
            if name in filepath.stem:
                obj_info = specs
                break

        if obj_info is None:
            raise ValueError(f"from_csv: {filepath.stem} is an unknown DLO.")

        trajectories = load_csv(filepath, is_2d)
        qs = trajectories.reshape(trajectories.shape[0], -1)
        S = get_S(trajectories, base_l2_reg=base_l2_reg)

        return cls(
            qs=jnp.asarray(qs),
            S=jnp.asarray(S),
            mass=obj_info["mass"],
            length=obj_info["length"],
        )

    def to_npz(self, filepath: Path | str):
        np.savez(filepath, qs=self.qs, S=self.S, mass=self.mass, length=self.length)


def get_S(
    trajectories: np.ndarray,
    fixed_idx: np.ndarray = np.array([0, -1]),
    k_neighbors: int = 15,
    base_l2_reg: float = 1e2,
):
    N, Nodes, _ = trajectories.shape
    fixed_idx = fixed_idx % Nodes
    free_idx = np.setdiff1d(np.arange(Nodes), fixed_idx)

    free = trajectories[:, free_idx].reshape(N, -1)
    fixed = trajectories[:, fixed_idx].reshape(N, -1)

    total_free_dofs = free.shape[1]
    total_fixed_dofs = fixed.shape[1]

    # Fit Nearest Neighbors
    nn = NearestNeighbors(n_neighbors=k_neighbors + 1)
    nn.fit(fixed)

    # Get distances/indices of the nearest neighbors
    distances, indices = nn.kneighbors(fixed)
    neighbor_idx = indices[:, 1:]
    neighbor_dist = distances[:, 1:]

    jacobians = np.zeros((N, total_free_dofs, total_fixed_dofs))

    for i in range(N):
        b_center, q_center = fixed[i], free[i]
        db = fixed[neighbor_idx[i]] - b_center
        dq = free[neighbor_idx[i]] - q_center

        # Calculate local variance to scale to handle collection variance
        local_variance = np.mean(neighbor_dist[i] ** 2)
        adaptive_reg = base_l2_reg * local_variance

        # Ridge regression
        A = (db.T @ db) + adaptive_reg * np.eye(total_fixed_dofs)
        B = db.T @ dq

        S_T = np.linalg.solve(A, B)
        jacobians[i] = S_T.T

    return jacobians


def get_FIM(model: TripletModel, data: Dataset, config: TrainConfig):
    # Boundary conditions
    idx_b = jnp.array([0, 1, -2, -1]) if config.idx_b is None else config.idx_b

    # Get rod
    geom = djx.Geometry(config.length, config.radius)
    mat = djx.Material(config.density, config.youngs_mod)

    # We are not using bc.get_q(...) so xb_m and xb_c are unneeded
    bc = djx.LinearBC(idx_b=idx_b, xb_m=jnp.array([]), xb_c=jnp.array([]))
    rod = djx.Rod2D.from_geometry(geom, mat, N=config.N, bc=bc)

    # 1. Get Batched Jacobians
    J_theta = jax.vmap(lambda q: jax.grad(rod.get_E, 2)(jnp.array([]), q, model))(
        data.qs
    )

    def flatten_fn(g):
        flat_vector, _ = jax.flatten_util.ravel_pytree(g)
        return flat_vector

    J_flat = jax.vmap(flatten_fn)(J_theta)

    # Setup dimensions and global FIM
    N = len(data.qs)
    P = J_flat.shape[1]

    # J^T J (The unscaled sum of information)
    FIM_sum = jnp.dot(J_flat.T, J_flat)

    # The Global Averaged FIM
    FIM = FIM_sum / N
    reg_eye = 1e-6 * jnp.eye(P)

    # --- Global Metrics ---
    t_opt_global = jnp.trace(FIM)
    _, d_opt_global = jnp.linalg.slogdet(FIM + reg_eye)
    e_opt_global = jnp.linalg.eigvalsh(FIM + reg_eye)[0]

    # Precompute inverse of FIM_sum for fast Leverage Score calculation
    inv_FIM_sum = jnp.linalg.inv(FIM_sum + reg_eye)

    # 2. Define the Per-Sample Evaluation Logic
    def evaluate_sample(j_i):
        """Evaluates the optimality of a single row/sample of the Jacobian"""

        # A. T-Optimality (Individual Magnitude)
        t_opt_i = jnp.dot(j_i, j_i)

        # B. Leverage Score (Uniqueness/Outlier Status)
        # Formula: j_i^T * (J^T J)^-1 * j_i
        lev_i = jnp.dot(j_i, jnp.dot(inv_FIM_sum, j_i))

        # C. Leave-One-Out (LOO) Matrix Updates
        # Remove this point's outer product from the total sum
        FIM_loo_sum = FIM_sum - jnp.outer(j_i, j_i)

        # Average over N-1 samples
        FIM_loo = FIM_loo_sum / (N - 1)
        reg_FIM_loo = FIM_loo + reg_eye

        # D. Calculate LOO D and E metrics
        _, d_loo_i = jnp.linalg.slogdet(reg_FIM_loo)
        e_loo_i = jnp.linalg.eigvalsh(reg_FIM_loo)[0]

        return t_opt_i, lev_i, d_loo_i, e_loo_i

    # 3. Vmap across the entire dataset
    t_scores, lev_scores, d_loo_scores, e_loo_scores = jax.vmap(evaluate_sample)(J_flat)

    # 4. Calculate Marginal Gain
    d_gain = d_opt_global - d_loo_scores
    e_gain = e_opt_global - e_loo_scores

    global_metrics = jnp.stack([t_opt_global, d_opt_global, e_opt_global])
    sample_metrics = jnp.stack([t_scores, lev_scores, d_gain, e_gain])
    return global_metrics, sample_metrics
