from abc import abstractmethod
import equinox as eqx
import jax
import jax.numpy as jnp

from jaxtyping import Float, jaxtyped, TypeCheckError
from beartype import beartype


class TripletModel(eqx.Module):
    """NN base class."""

    def __init__(self, key: jax.Array): ...

    @abstractmethod
    def __call__(self, del_strain: Float[jax.Array, "3"]) -> Float[jax.Array, ""]: ...  # noqa

    @classmethod
    def validate(cls, target_cls: type) -> None:
        """Validates a provided class as a suitable implementation of this model."""
        if not issubclass(target_cls, cls):
            raise ValueError(
                f"Validation failed: {target_cls} is not a subclass of {cls.__name__}"
            )
        try:
            obj = target_cls(jax.random.PRNGKey(42))
        except TypeError as e:
            raise ValueError(
                f"Validation failed: {target_cls.__name__} cannot be initialized with a PRNGKey.\n{e}"
            )
        try:
            res = obj(jnp.zeros(3))
            if res.shape != ():
                raise ValueError(
                    f"Validation failed: Expected scalar output, got shape {res.shape}"
                )
        except TypeCheckError as e:
            raise ValueError(
                f"Validation failed: {target_cls.__name__}.__call__ failed type check:\n{e}"
            )
        except Exception as e:
            raise ValueError(
                f"Validation failed: {target_cls.__name__}.__call__ raised an exception: \n{e}"
            )

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        # Enforce jaxtyped runtime check for all subclasses
        parent_annotations = TripletModel.__call__.__annotations__
        if "__call__" in cls.__dict__:
            child_call = cls.__dict__["__call__"]
            child_call.__annotations__ = parent_annotations.copy()
            cls.__call__ = jaxtyped(typechecker=beartype)(child_call)
