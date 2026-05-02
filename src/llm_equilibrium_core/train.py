import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import optax

import dismech_jax as djx
from .triplet_model import TripletModel
from .util import Dataset, TrainConfig, ActiveMethod, IniMethod


def train(
    model_cls: type[TripletModel],
    train_ds: Dataset,
    valid_ds: Dataset,
    key: jax.Array = jax.random.PRNGKey(42),
    config: TrainConfig = TrainConfig(),
):
    # Config check
    TripletModel.validate(model_cls)
    if config.N0_filler > config.N0_samples:
        raise ValueError(
            f"train: config.N0_filler must be less than config.N0_samples ({config.N0_filler}>{config.N0_samples})"
        )

    # Pseudo random from key
    master_key, init_key, loop_key, numpy_key = jax.random.split(key, 4)
    np.random.seed(int(jax.random.randint(numpy_key, (), 0, 2**31 - 1)))

    # Get rod
    geom = djx.Geometry(config.length, config.radius)
    mat = djx.Material(config.density, config.youngs_mod)
    base = djx.Rod2D.from_geometry(geom, mat, N=config.N)

    # Create bc for validation loss
    idx_b = jnp.array([0, 1, -2, -1]) if config.idx_b is None else config.idx_b
    xb_c = base.q0[idx_b]
    valid_xb_m = valid_ds.qs[:, idx_b] - xb_c
    bc = djx.BatchedLinearBC(idx_b=idx_b, xb_c=xb_c, xb_m=valid_xb_m)
    rod = base.with_bc(bc)

    # For get_E
    idx_b = idx_b % rod.q0.size
    idx_f = jnp.setdiff1d(jnp.arange(rod.q0.size), idx_b)

    # ML setup
    init_keys = jax.random.split(init_key, config.ensemble_size)
    init_models = jax.vmap(lambda k: model_cls(k))(init_keys)
    optimizer = optax.adam(config.lr)
    init_opt_states = jax.vmap(optimizer.init)(eqx.filter(init_models, eqx.is_array))

    # Helper functions that encapsulate rod object
    def get_E(x_f, x_b, current_model):
        q_new = jnp.zeros_like(rod.q0).at[idx_b].set(x_b).at[idx_f].set(x_f)
        return rod.get_E(jnp.array([]), q_new, current_model)

    def get_S(current_model, q):
        xf, xb = q[idx_f], q[idx_b]
        dxfdxfE = jax.hessian(get_E, 0)(xf, xb, current_model)
        dxfdxbE = jax.jacobian(jax.grad(get_E, 0), 1)(xf, xb, current_model)
        return -jnp.linalg.solve(dxfdxfE + 1e-8 * jnp.eye(xf.size), dxfdxbE)

    def compute_pointwise_loss(current_model, qs):
        f_pred = jax.vmap(lambda q: rod.get_F(jnp.array([]), q, current_model))(qs)
        return jnp.mean(jnp.square(f_pred), axis=1)

    def compute_masked_loss(current_model, qs, mask):
        # Masked loss to preserve shape when changing # of training samples
        pointwise_sq = compute_pointwise_loss(current_model, qs)
        return jnp.sum(pointwise_sq * mask) / jnp.maximum(1.0, jnp.sum(mask))

    def compute_eval_loss(current_model, qs):
        qs_pred = rod.solve(
            current_model, jnp.array([0.0, 1.0]), None, max_dlambda=1e-2, iters=5
        )
        return jnp.mean(jnp.linalg.norm(qs - qs_pred[:, 1], axis=1))

    # Vectorized operators
    v_get_S = jax.vmap(jax.vmap(get_S, (None, 0)), (0, None))
    v_get_F = jax.vmap(
        jax.vmap(lambda m, q: rod.get_F(jnp.array([]), q, m), (None, 0)), (0, None)
    )

    pointwise_loss_fn = jax.vmap(compute_pointwise_loss, (0, None))
    loss_and_grad = jax.vmap(
        eqx.filter_value_and_grad(compute_masked_loss), (0, None, None)
    )
    eval_loss_fn = jax.vmap(compute_eval_loss, (0, None))
    solve_fn = jax.vmap(
        lambda m: rod.solve(m, jnp.array([0.0, 1.0]), None, max_dlambda=1e-2, iters=5)[
            :, 1
        ]
    )

    # Get new samples
    def acquire_new_samples(ensemble, mask, acq_key):
        qs_pool, S_pool = train_ds.qs, train_ds.S

        if config.active_method == ActiveMethod.RANDOM:
            scores = jax.random.uniform(acq_key, shape=(qs_pool.shape[0],))
        elif config.active_method == ActiveMethod.EXPERIMENTAL_S:
            scores = jnp.linalg.norm(S_pool, ord="fro", axis=(1, 2))
        elif config.active_method == ActiveMethod.OFFLINE_S:
            scores = jnp.linalg.norm(
                v_get_S(ensemble, qs_pool), ord="fro", axis=(2, 3)
            ).mean(axis=0)
        elif config.active_method == ActiveMethod.ONLINE_S:
            qs_pred = jnp.median(solve_fn(ensemble), axis=0)
            scores = jnp.linalg.norm(
                v_get_S(ensemble, qs_pred), ord="fro", axis=(2, 3)
            ).mean(axis=0)
        elif config.active_method == ActiveMethod.OFFLINE_RES:
            scores = pointwise_loss_fn(ensemble, qs_pool).mean(axis=0)
        elif config.active_method == ActiveMethod.ONLINE_RES:
            qs_pred = jnp.median(solve_fn(ensemble), axis=0)
            scores = pointwise_loss_fn(ensemble, qs_pred).mean(axis=0)
        elif config.active_method == ActiveMethod.OFFLINE_UNCERTAINTY:
            scores = jnp.var(v_get_F(ensemble, qs_pool), axis=0).mean(axis=-1)
        elif config.active_method == ActiveMethod.ONLINE_UNCERTAINTY:
            qs_pred = jnp.median(solve_fn(ensemble), axis=0)
            scores = jnp.var(v_get_F(ensemble, qs_pred), axis=0).mean(axis=-1)
        else:
            raise ValueError(f"Method {config.active_method} not implemented.")

        scores = jnp.where(mask, -jnp.inf, scores)
        _, top_k_idx = jax.lax.top_k(scores, config.N_samples_per_addition)
        return mask.at[top_k_idx].set(True)

    def train_n_epochs(models, opt_states, qs, mask, epochs):
        num_saves = epochs // config.save_every

        # Double scan to only calculate valid loss every config.save_every
        def outer_save_step(carry, save_idx):
            def inner_train_step(inner_carry, _):
                m_curr, opt_curr = inner_carry
                _, grads = loss_and_grad(m_curr, qs, mask)
                updates, new_opt = jax.vmap(optimizer.update)(grads, opt_curr, m_curr)
                new_m = jax.vmap(eqx.apply_updates)(m_curr, updates)
                return (new_m, new_opt), None

            (m_next, opt_next), _ = jax.lax.scan(
                inner_train_step, carry, jnp.arange(config.save_every)
            )

            v_loss = jnp.mean(eval_loss_fn(m_next, valid_ds.qs))
            return (m_next, opt_next), v_loss

        return jax.lax.scan(
            outer_save_step, (models, opt_states), jnp.arange(num_saves)
        )

    @eqx.filter_jit
    def active_learning_loop(models, opt_states, mask, key):
        def active_step(carry, step_idx):
            m_curr, opt_curr, mask_curr, rng_curr = carry

            (m_next, opt_next), valid_losses = train_n_epochs(
                m_curr, opt_curr, train_ds.qs, mask_curr, config.N_epochs_per_addition
            )

            rng_acq, rng_next = jax.random.split(rng_curr)
            mask_next = jax.lax.cond(
                step_idx < config.N_additions,
                lambda m: acquire_new_samples(m_next, m, rng_acq),
                lambda m: m,
                mask_curr,
            )
            return (m_next, opt_next, mask_next, rng_next), valid_losses

        final_carry, valid_history = jax.lax.scan(
            active_step,
            (models, opt_states, mask, key),
            jnp.arange(config.N_additions + 1),
        )
        return final_carry[0], valid_history

    @eqx.filter_jit
    def standard_training_loop(models, opt_states, active_qs):
        # We only pass in the true slice of data to save array size
        all_true_mask = jnp.ones(active_qs.shape[0], dtype=bool)
        (final_models, _), valid_history = train_n_epochs(
            models, opt_states, active_qs, all_true_mask, config.epochs
        )
        return final_models, valid_history

    # Main initialization
    N_train = train_ds.qs.shape[0]
    active_mask = np.zeros(N_train, dtype=bool)
    N_method = config.N0_samples - config.N0_filler

    # Get samples
    if N_method > 0:
        if config.ini_method == IniMethod.CENTROID:
            scores = np.linalg.norm(train_ds.qs - np.mean(train_ds.qs, axis=0), axis=1)
            active_mask[np.argsort(scores)[:N_method]] = True
        elif config.ini_method == IniMethod.RANDOM:
            active_mask[np.random.choice(N_train, N_method, replace=False)] = True
        elif config.ini_method == IniMethod.MAX_S:
            scores = np.linalg.norm(train_ds.S, ord="fro", axis=(1, 2))
            active_mask[np.argsort(scores)[-N_method:]] = True

    # Replace with random if filler (for ablation)
    if config.N0_filler > 0:
        unpicked_indices = np.where(~active_mask)[0]
        filler_idx = np.random.choice(unpicked_indices, config.N0_filler, replace=False)
        active_mask[filler_idx] = True

    if config.active:
        final_models, valid_losses = active_learning_loop(
            init_models, init_opt_states, jnp.array(active_mask), loop_key
        )
        return final_models, valid_losses.flatten()

    train_idx = np.where(active_mask)[0]
    final_models, valid_losses = standard_training_loop(
        init_models, init_opt_states, train_ds.qs[train_idx]
    )
    return final_models, valid_losses.flatten()
