from abc import abstractmethod
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Float, jaxtyped, TypeCheckError
from beartype import beartype


class TripletModel(eqx.Module):
    """NN base class."""

    def __init__(self, K0: jax.Array, key: jax.Array): ...

    @abstractmethod
    def __call__(self, del_strain: Float[jax.Array, "3"]) -> Float[jax.Array, ""]: ...

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce jaxtyped runtime check for all subclasses
        parent_annotations = TripletModel.__call__.__annotations__
        if "__call__" in cls.__dict__:
            child_call = cls.__dict__["__call__"]
            child_call.__annotations__ = parent_annotations.copy()
            cls.__call__ = jaxtyped(typechecker=beartype)(child_call)


def validate_model(cls: type) -> None:
    """Validates a provided class as a suitable TripletModel. Throws a ValueError if improper."""
    if not issubclass(cls, TripletModel):
        raise ValueError(f"validate_model: {cls} is not an subclass of {TripletModel}")
    try:
        obj = cls(jnp.ones(3), jax.random.PRNGKey(42))
    except TypeError:
        raise ValueError(f"validate_model: {cls} cannot be initialized with a PRNGKey")
    try:
        obj(jnp.empty(3))
    except TypeCheckError as e:
        raise ValueError(f"validate_model: obj.__call__ must return a scalar:\n {e}")
    except Exception as e:
        raise ValueError(
            f"validate_model: obj.__call__ encountered an unknown exception: \n {e}"
        )
