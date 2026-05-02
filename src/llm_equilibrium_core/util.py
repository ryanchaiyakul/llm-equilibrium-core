from __future__ import annotations
from pathlib import Path
from dataclasses import dataclass
from enum import IntEnum
import jax
import jax.numpy as jnp
import jax.flatten_util
import pandas as pd
import numpy as np
import equinox as eqx
import json

from sklearn.neighbors import NearestNeighbors


class ActiveMethod(IntEnum):
    RANDOM = 0  # Baseline
    EXPERIMENTAL_S = 1  # Oracle S
    OFFLINE_S = 2  # Offline S
    ONLINE_S = 3  # Online S
    OFFLINE_RES = 4  # Offline residual
    ONLINE_RES = 5  # Online residual
    OFFLINE_UNCERTAINTY = 6  # Ensemble uncertainty
    ONLINE_UNCERTAINTY = 7  # Furthest point sampling


class IniMethod(IntEnum):
    RANDOM = 0  # Baseline
    MAX_S = 1  # Oracle S
    CENTROID = 2  # Most average point


@dataclass
class TrainConfig:
    # Rod config
    length: float = 0.1
    radius: float = 1e-3
    density: float = 1e3
    youngs_mod: float = 1e6
    N: int = 5
    idx_b: jax.Array | None = None

    # Training config
    lr: float = 1e-1
    save_every: int = 1000
    ensemble_size: int = 5

    # Passive config
    epochs: int = 1000

    # Active config
    active: bool = True
    active_method: ActiveMethod = ActiveMethod.RANDOM
    N_epochs_per_addition: int = 500
    N_samples_per_addition: int = 1
    N_additions: int = 9

    # Initialization
    ini_method: IniMethod = IniMethod.CENTROID
    N0_samples: int = 1
    N0_filler: int = 0

    # Loss config
    S_factor: float = 0.0


class Dataset(eqx.Module):
    qs: jax.Array
    S: jax.Array
    length: float
    mass: float

    _OBJECT_SPECS = {
        "slinky": {"length": 0.2, "mass": 30e-3},
        "strip": {"length": 0.33, "mass": 7e-3},
        "brazier": {"length": 0.35, "mass": 10e-3},
        "tube": {"length": 0.35, "mass": 10e-3},  # alias
        "tape": {"length": 0.3, "mass": 10e-3},
    }

    @classmethod
    def _get_obj_info(cls, stem: str):
        for name, specs in cls._OBJECT_SPECS.items():
            if name in stem:
                return specs
        raise ValueError(f"Unknown DLO type: {stem}")

    @classmethod
    def from_csv(
        cls, filepath: Path | str, is_2d: bool = True, base_l2_reg=1e2
    ) -> Dataset:
        df = pd.read_csv(filepath)

        def get_csv_cols(columns):
            coord_cols = [
                c
                for c in columns
                if "marker_" in c and any(a in c for a in ["_x", "_y", "_z"])
            ]

            def sort_key(name):
                parts = name.split("_")
                return (int(parts[1]), {"x": 0, "y": 1, "z": 2}[parts[2]])

            return sorted(coord_cols, key=sort_key)

        return cls._process_raw_data(
            filepath=filepath,
            df=df,
            col_extractor_fn=get_csv_cols,
            is_2d=is_2d,
            base_l2_reg=base_l2_reg,
        )

    @classmethod
    def from_json(
        cls,
        filepath: Path | str,
        is_2d: bool = True,
        markers: int = 5,
        base_l2_reg=1e2,
    ) -> Dataset:
        with Path(filepath).open("r") as f:
            data = [json.loads(line) for line in f]
        df = pd.json_normalize(data)

        def get_json_cols(columns):
            coord_cols = [
                c
                for c in columns
                if "marker_" in c and any(a in c for a in [".x", ".y", ".z"])
            ]

            def sort_key(name: str):
                parts = name.split("_")[-1].split(".")
                return (int(parts[0]), {"x": 0, "y": 1, "z": 2}[parts[1]])

            return sorted(coord_cols, key=sort_key)[: markers * 3]

        return cls._process_raw_data(
            filepath=filepath,
            df=df,
            col_extractor_fn=get_json_cols,
            is_2d=is_2d,
            base_l2_reg=base_l2_reg,
        )

    @classmethod
    def _process_raw_data(
        cls,
        filepath: Path | str,
        df: pd.DataFrame,
        col_extractor_fn: callable,
        is_2d=True,
        base_l2_reg=1e2,
    ) -> Dataset:
        """Read dataframe, clean trajectories, compute S, and return new dataset."""
        filepath = Path(filepath)
        obj_info = cls._get_obj_info(filepath.stem)
        target_cols = col_extractor_fn(df.columns)
        clean_df = df[target_cols].dropna()
        trajectories = clean_df.values.reshape(len(clean_df), -1, 3)
        trajectories = cls.align_trajectories(trajectories, is_2d)
        qs = trajectories.reshape(trajectories.shape[0], -1)
        S = cls.get_S(trajectories, base_l2_reg=base_l2_reg)

        return cls(
            qs=jnp.asarray(qs),
            S=jnp.asarray(S),
            mass=obj_info["mass"],
            length=obj_info["length"],
        )

    @classmethod
    def from_npz(cls, filepath: Path | str):
        data = np.load(filepath)
        return cls(
            qs=jnp.asarray(data["qs"]),
            S=jnp.asarray(data["S"]),
            mass=float(data["mass"]),
            length=float(data["length"]),
        )

    def to_npz(self, filepath: Path | str):
        np.savez(filepath, qs=self.qs, S=self.S, mass=self.mass, length=self.length)

    @staticmethod
    def get_S(
        trajectories: np.ndarray,
        fixed_idx: np.ndarray = np.array([0, -1]),
        k_neighbors: int = 15,
        base_l2_reg: float = 1e2,
    ) -> np.ndarray:
        """Extrapolate sensitivity from nearest neighbors."""
        N, Nodes, _ = trajectories.shape
        fixed_idx = fixed_idx % Nodes
        free_idx = np.setdiff1d(np.arange(Nodes), fixed_idx)

        free = trajectories[:, free_idx].reshape(N, -1)
        fixed = trajectories[:, fixed_idx].reshape(N, -1)

        nn = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(fixed)
        distances, indices = nn.kneighbors(fixed)

        neighbor_idx = indices[:, 1:]
        neighbor_dist = distances[:, 1:]

        jacobians = np.zeros((N, free.shape[1], fixed.shape[1]))

        for i in range(N):
            db = fixed[neighbor_idx[i]] - fixed[i]
            dq = free[neighbor_idx[i]] - free[i]
            reg = base_l2_reg * np.mean(neighbor_dist[i] ** 2)
            A = (db.T @ db) + reg * np.eye(fixed.shape[1])
            B = db.T @ dq
            jacobians[i] = np.linalg.solve(A, B).T

        return jacobians

    @staticmethod
    def align_trajectories(trajectories: np.ndarray, is_2d: bool = False) -> np.ndarray:
        """Rotate and center trajectories at the origin."""
        # 4. Rotate for gravity = Z-Down
        alignment_matrix = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
        trajectories = trajectories @ alignment_matrix.T

        # Center at (0,0,0)
        root_origin = trajectories[:, 0, :]
        trajectories = trajectories - root_origin[:, None, ...]

        if is_2d:
            trajectories = trajectories[..., [0, 2]]
        return trajectories
