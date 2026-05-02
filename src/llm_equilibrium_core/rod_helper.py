import jax
import jax.numpy as jnp
import dismech_jax as djx

from .triplet_model import TripletModel
from .util import TrainConfig


def solve(
    model: TripletModel,
    xb: jax.Array,
    length: float,
) -> jax.Array:
    """Solve for full state for fixed DOFs.

    Args:
        model (TripletModel): triplet model instance.
        xb (jax.Array): fixed DOFs (N, 2) or (2,).
        length (float): Length of DLO.

    Returns:
        jax.Array: full DOFs (N, 10) or (10,).
    """
    # Default TrainConfig except length
    config: TrainConfig = TrainConfig()

    # Get rod
    geom = djx.Geometry(length, config.radius)
    mat = djx.Material(config.density, config.youngs_mod)
    base = djx.Rod2D.from_geometry(geom, mat, N=config.N)

    # Create bc for validation loss
    idx_b = jnp.array([0, 1, -2, -1]) if config.idx_b is None else config.idx_b
    xb_c = base.q0[idx_b]
    valid_xb_m = xb - xb_c
    if valid_xb_m.ndim == 1:
        bc = djx.LinearBC(idx_b=idx_b, xb_c=xb_c, xb_m=valid_xb_m)
    elif valid_xb_m.ndim == 2:
        bc = djx.BatchedLinearBC(idx_b=idx_b, xb_c=xb_c, xb_m=valid_xb_m)
    else:
        raise ValueError(f"solve: xb must have 1 or 2 dimensions, has: {xb.ndim}")
    rod = base.with_bc(bc)

    solve_fn = jax.vmap(
        lambda m: rod.solve(m, jnp.array([0.0, 1.0]), None, max_dlambda=1e-2, iters=5)
    )

    all_solutions = solve_fn(model)

    if valid_xb_m.ndim == 1:
        ensemble_preds = all_solutions[:, 1]
    else:
        ensemble_preds = all_solutions[:, :, 1]
    return jnp.median(ensemble_preds, axis=0)


def get_S(
    model: TripletModel,
    q: jax.Array,
    length: float,
) -> jax.Array:
    """Solve for full state for fixed DOFs.

    Args:
        model (TripletModel): triplet model instance.
        q (jax.Array): full DOFs (N, 10) or (10,).
        length (float): Length of DLO.

    Returns:
        jax.Array: full sensitivity jacobian (N, 6, 4) or (6, 4).
    """
    # Default TrainConfig except length
    config: TrainConfig = TrainConfig()

    # Get rod
    geom = djx.Geometry(length, config.radius)
    mat = djx.Material(config.density, config.youngs_mod)
    base = djx.Rod2D.from_geometry(geom, mat, N=config.N)

    idx_b = jnp.array([0, 1, -2, -1]) if config.idx_b is None else config.idx_b
    bc = djx.LinearBC(idx_b=idx_b, xb_c=jnp.array([]), xb_m=jnp.array([]))  # Don't need
    rod = base.with_bc(bc)

    idx_b = idx_b % rod.q0.size
    idx_f = jnp.setdiff1d(jnp.arange(rod.q0.size), idx_b)

    def get_E(x_f, x_b, current_model):
        q_new = jnp.zeros_like(rod.q0).at[idx_b].set(x_b).at[idx_f].set(x_f)
        return rod.get_E(jnp.array([]), q_new, current_model)

    def get_S(current_model, q):
        xf, xb = q[idx_f], q[idx_b]
        dxfdxfE = jax.hessian(get_E, 0)(xf, xb, current_model)
        dxfdxbE = jax.jacobian(jax.grad(get_E, 0), 1)(xf, xb, current_model)
        return -jnp.linalg.solve(dxfdxfE + 1e-8 * jnp.eye(xf.size), dxfdxbE)

    v_get_S = jax.vmap(jax.vmap(get_S, (None, 0)), (0, None))

    if q.ndim == 1:
        S_ensemble = jax.vmap(get_S, (0, None))(model, q)
    else:
        S_ensemble = v_get_S(model, q)

    return jnp.mean(S_ensemble, axis=0)
