from dataclasses import dataclass
from pathlib import Path
import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

import dismech_jax as djx
from llm_equilibrium_core import TripletModel, validate_model


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

    def to_npz(self, filepath: Path | str):
        jnp.savez(filepath, qs=self.qs, S=self.S, mass=self.mass, length=self.length)


@dataclass
class TrainConfig:
    # Rod config
    length: float = 0.1
    radius: float = 1e-3
    density: float = 1e3
    youngs_mod: float = 1e6
    N: int = 5
    idx_b: jax.Array | None = None
    K0: jax.Array | None = None

    # Loss config
    S_factor: float = 0.1

    # Training config
    lr: float = 1e-3
    epochs: int = 1000
    print_every: int = 100
    verbose: bool = True


def train(
    model_cls: type[TripletModel],
    train: Dataset,
    valid: Dataset,
    key: jax.Array = jax.random.PRNGKey(42),
    config: TrainConfig = TrainConfig(),
):
    validate_model(model_cls)

    # Boundary conditions
    idx_b = jnp.array([0, 1, -2, -1]) if config.idx_b is None else config.idx_b

    # Get rod
    geom = djx.Geometry(config.length, config.radius)
    mat = djx.Material(config.density, config.youngs_mod)

    # We are not using bc.get_q(...) so xb_m and xb_c are unneeded
    bc = djx.LinearBC(idx_b=idx_b, xb_m=jnp.array([]), xb_c=jnp.array([]))
    rod = djx.Rod2D.from_geometry(geom, mat, N=config.N, bc=bc)

    # Initialize base model
    K0 = rod.get_DER(geom, mat).K if config.K0 is None else config.K0
    model = model_cls(K0, key)

    # Resolve indicies
    idx_b = idx_b % rod.q0.size
    idx_f = jnp.setdiff1d(jnp.arange(rod.q0.size), idx_b)

    def get_E(x_f: jax.Array, x_b: jax.Array, current_model: TripletModel):
        """Recombines split state arrays into full state to compute energy."""
        q_new = jnp.zeros_like(rod.q0)
        q_new = q_new.at[idx_b].set(x_b)
        q_new = q_new.at[idx_f].set(x_f)
        return rod.get_E(jnp.array([]), q_new, current_model, None)

    def get_S(current_model: TripletModel, q: jax.Array):
        """Computes the Equilibrium Sensitivity Operator."""
        xf = q[idx_f]
        xb = q[idx_b]
        dxfdxfE = jax.hessian(get_E, 0)(xf, xb, current_model)
        dxfdxbE = jax.jacobian(jax.grad(get_E, 0), 1)(xf, xb, current_model)
        return -jnp.linalg.solve(dxfdxfE + 1e-8 * jnp.eye(xf.size), dxfdxbE)

    def compute_loss(current_model: TripletModel, dataset: Dataset) -> jax.Array:
        """Computes the combined equilibrium and sensitivity loss."""
        # If memory becomes an issue with large batches, change jax.vmap to jax.lax.map
        qs, S_truth = dataset.qs, dataset.S
        S_pred = jax.vmap(lambda q: get_S(current_model, q))(qs)
        F_pred = jax.vmap(lambda q: rod.get_F(jnp.array([]), q, current_model, None))(
            qs
        )
        L_eq = jnp.mean(jnp.square(F_pred))  # Want residual to be 0
        L_sens = jnp.sum(jnp.square(S_truth - S_pred))
        return L_eq + config.S_factor * L_sens

    def compute_eval_loss(current_model: TripletModel, dataset: Dataset) -> jax.Array:
        """Computes the combined equilibrium and sensitivity loss."""
        # If memory becomes an issue with large batches, change jax.vmap to jax.lax.map
        qs = dataset.qs
        F_pred = jax.vmap(lambda q: rod.get_F(jnp.array([]), q, current_model, None))(
            qs
        )
        return jnp.mean(jnp.square(F_pred))  # Want residual to be 0

    loss_and_grad = eqx.filter_value_and_grad(compute_loss)

    optimizer = optax.adam(config.lr)
    opt_state = optimizer.init(eqx.filter(model, eqx.is_array))

    def scan_body(carry, epoch):
        model_carry, opt_state_carry = carry

        train_loss, grads = loss_and_grad(model_carry, train)
        updates, new_opt_state = optimizer.update(grads, opt_state_carry, model_carry)
        new_model = eqx.apply_updates(model_carry, updates)
        valid_loss = compute_eval_loss(new_model, valid)

        # 4. Conditional Debug Printing
        def print_metrics(_):
            jax.debug.print(
                "Epoch {e:04d} | Train Loss: {t:.6f} | Valid Loss: {v:.6f}",
                e=epoch + 1,
                t=train_loss,
                v=valid_loss,
            )

        # jax.lax.cond allows us to safely execute the print operation inside a JIT-compiled scan
        do_print = jnp.logical_and(
            config.verbose, (epoch + 1) % config.print_every == 0
        )
        jax.lax.cond(do_print, print_metrics, lambda _: None, None)

        return (new_model, new_opt_state), (train_loss, valid_loss)

    @eqx.filter_jit
    def run_training(init_model, init_opt_state):
        epochs_array = jnp.arange(config.epochs)
        (final_model, _), (t_losses, v_losses) = jax.lax.scan(
            scan_body, (init_model, init_opt_state), epochs_array
        )
        return final_model, t_losses, v_losses

    print(f"Starting training for {config.epochs} epochs...")
    final_model, train_losses, valid_losses = run_training(model, opt_state)

    return final_model, train_losses, valid_losses
