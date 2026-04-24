import pandas as pd
import numpy as np


def load_csv(filepath: str, is_2d: bool = False) -> np.ndarray:
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


def get_S(
    trajectories: np.ndarray,
    fixed_idx: np.ndarray = np.array([0, -1]),
    window_size: int = 11,
    l2_reg: float = 1e-5,
):
    """Extract a sensitivity operator from trajectories dataset.

    Args:
        trajectories (np.ndarray): (N, Nodes, DoFs)
        fixed_idx (np.ndarray, optional): Fixed boundary index. Defaults to np.array([0, -1]).
        window_size (int, optional): _description_. Defaults to 11.
        l2_reg (float, optional): _description_. Defaults to 1e-5.

    Returns:
        np.ndarray: (N, (Nodes - 2) * DoFs, DoFs)
    """
    N, Nodes, DoFs = trajectories.shape
    fixed_idx = fixed_idx % Nodes
    free_idx = np.setdiff1d(np.arange(Nodes), fixed_idx)

    free = trajectories[:, free_idx].reshape(N, -1)
    fixed = trajectories[:, fixed_idx].reshape(N, -1)

    total_free_dofs = free.shape[1]
    total_fixed_dofs = fixed.shape[1]

    dq = np.diff(free, axis=0)
    db = np.diff(fixed, axis=0)
    dq = np.vstack([dq, dq[-1:]])
    db = np.vstack([db, db[-1:]])

    pad_width = window_size // 2
    dq_padded = np.pad(dq, ((pad_width, pad_width), (0, 0)), mode="edge")
    db_padded = np.pad(db, ((pad_width, pad_width), (0, 0)), mode="edge")

    jacobians = np.zeros((N, total_free_dofs, total_fixed_dofs))

    for i in range(N):
        dq_win = dq_padded[i : i + window_size]
        db_win = db_padded[i : i + window_size]

        A = (db_win.T @ db_win) + l2_reg * np.eye(total_fixed_dofs)
        B = db_win.T @ dq_win

        S_T = np.linalg.solve(A, B)
        jacobians[i] = S_T.T

    return jacobians
