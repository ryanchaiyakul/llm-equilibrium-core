import argparse

import jax
import jax.numpy as jnp
import numpy as np
import equinox as eqx
import tqdm

from llm_equilibrium_core import (
    TripletModel,
    Dataset,
    TrainConfig,
    IniMethod,
    train,
)

jax.config.update("jax_enable_x64", True)


class ICNN_DER(TripletModel):
    K: jax.Array
    line1: eqx.nn.Linear
    line2: eqx.nn.Linear

    def __init__(self, key: jax.Array):
        k_k, k1, k2 = jax.random.split(key, 3)
        self.K = jax.random.normal(k_k, (3,)) + 1.0

        in_size = 3
        width = 16
        out_size = 1

        self.line1 = eqx.nn.Linear(in_size, width, key=k1)
        self.line2 = eqx.nn.Linear(width, out_size, key=k2)

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        x = self.line1(del_strain)
        x = jax.nn.softplus(x)
        x = jax.nn.softplus(self.line2.weight) @ x + self.line2.bias
        return jnp.sum(jnp.abs(self.K) * del_strain**2) + jax.nn.softplus(x)[0]


class CoupledDER(TripletModel):
    l_elements: jax.Array

    def __init__(self, key: jax.Array):
        self.l_elements = jax.random.normal(key, (6,)) * 0.1 + 1.0

    def __call__(self, del_strain: jax.Array) -> jax.Array:
        # 1. Reconstruct the Lower Triangular Matrix L
        L = jnp.array(
            [
                [self.l_elements[0], 0, 0],
                [self.l_elements[1], self.l_elements[2], 0],
                [self.l_elements[3], self.l_elements[4], self.l_elements[5]],
            ]
        )
        K = L @ L.T
        energy = 0.5 * del_strain.T @ K @ del_strain
        return energy


class TapeDER(TripletModel):
    K_stretch: jax.Array
    D: jax.Array
    kappa_0: jax.Array
    nu: jax.Array
    kappa_crit_pos: jax.Array  # Snap point when bending WITH the curve
    kappa_crit_neg: jax.Array  # Snap point when bending AGAINST the curve
    eps: jax.Array

    def __init__(self, key: jax.Array):
        k_s, k_d, k_k0, k_nu, k_cp, k_cn = jax.random.split(key, 6)

        self.K_stretch = jax.random.uniform(k_s, (1,), minval=1000.0, maxval=5000.0)
        self.D = jax.random.uniform(k_d, (1,), minval=1.0, maxval=10.0)
        self.kappa_0 = jax.random.uniform(k_k0, (1,), minval=10.0, maxval=50.0)
        self.nu = jax.random.uniform(k_nu, (1,), minval=0.2, maxval=0.4)

        # Bending AGAINST the curve snaps easier (lower threshold)
        self.kappa_crit_neg = jax.random.uniform(k_cn, (1,), minval=0.05, maxval=0.2)
        # Bending WITH the curve requires more curvature to snap (higher threshold)
        self.kappa_crit_pos = jax.random.uniform(k_cp, (1,), minval=0.3, maxval=0.8)

        self.eps = jnp.array([0.02])

    def __call__(self, strains: jax.Array) -> jax.Array:
        stretch = strains[:2]
        bend_x = strains[2]

        # 1. Enforce physical constraints: stiffness and curvature must be > 0
        safe_K = jax.nn.softplus(self.K_stretch)
        safe_D = jax.nn.softplus(self.D)
        safe_kappa_0 = jax.nn.softplus(self.kappa_0)

        # 2. Stretch Energy
        e_stretch = 0.5 * jnp.sum(safe_K * stretch**2)

        # 3. Asymmetric Snap Logic
        # Select the critical threshold based on bend direction
        active_kappa_crit = jnp.where(
            bend_x > 0, self.kappa_crit_pos, self.kappa_crit_neg
        )

        # 4. Bending Energy
        flattening = jax.nn.sigmoid(
            (jnp.abs(bend_x) - jnp.abs(active_kappa_crit)) / self.eps
        )
        kappa_y = safe_kappa_0 * (1.0 - flattening)

        e_bend = (
            0.5
            * safe_D
            * (
                bend_x**2
                + (kappa_y - safe_kappa_0) ** 2
                + 2 * self.nu * bend_x * (kappa_y - safe_kappa_0)
            )
        )
        return jnp.squeeze(e_stretch + e_bend)


datasets = {
    "brazier": Dataset.from_npz("../data/DLO_brazier_2d_rect_2332pts_v1.npz"),
    "strip": Dataset.from_npz("../data/DLO_strip_2d_rect_2512pts_v5.npz"),
    "slinky": Dataset.from_npz("../data/DLO_slinky_2d_rect_1836pts_v1.npz"),
    "tape": Dataset.from_npz("../data/DLO_measuringtape_2d_rect_2357pts_v1.npz"),
}
model_cls = {
    "brazier": ICNN_DER,
    "strip": ICNN_DER,
    "slinky": CoupledDER,
    "tape": TapeDER,
}
if __name__ == "__main__":
    parser = argparse.ArgumentParser("Comparison")
    parser.add_argument("dataset", choices=["brazier", "strip", "slinky", "tape"])
    parser.add_argument("--seeds", type=int, default=10)
    parser.add_argument("--samples", type=int, default=10)
    args = parser.parse_args()
    dataset = datasets[args.dataset]

    output_loss = []
    for n in tqdm.trange(args.samples + 1, unit="samples"):
        valid_losses = []
        seed_pbar = tqdm.trange((args.seeds), unit="seed")
        for i in seed_pbar:
            config = TrainConfig(
                length=dataset.length,
                epochs=5000,
                lr=1e-1,
                S_factor=0.0,
                save_every=1000,
                active=False,
                N0_samples=args.samples,
                ini_method=IniMethod.MAX_S,
                N0_filler=n,
            )
            _, valid_loss = train(
                model_cls[args.dataset],
                dataset,
                dataset,
                config=config,
                key=jax.random.PRNGKey(42 + i),
            )
            valid_losses.append(valid_loss)
            seed_pbar.set_postfix(
                {
                    "Median Valid Loss": f"{np.nanmedian(np.asarray(valid_losses), axis=(0, 1)):.4f}"
                }
            )
        output_loss.append(valid_losses)
    np.savez(
        f"ablation_{args.dataset}_{args.seeds}.npz", valid_loss=np.asarray(output_loss)
    )
