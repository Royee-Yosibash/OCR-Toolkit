import typing
from abc import ABC


class SerializableClass(ABC):
    """Abstract base class providing recursive serialization to dict."""

    @classmethod
    def from_dict(cls, raw_dict: dict):
        # TODO: Make sure init=False is also supported
        """Recursively create an instance from a dict.

        Inspects ``cls.__init__`` type hints to discover parameters whose
        types expose a ``from_dict`` classmethod (or generic containers
        like ``list[T]`` / ``dict[K, T]`` where ``T`` exposes one).
        Matching dict values are deserialized recursively before being
        passed to the constructor.

        Args:
            raw_dict: A dict as produced by ``to_dict``.

        Returns:
            An instance of ``cls``.
        """
        hints = typing.get_type_hints(cls.__init__)
        hints.pop("return", None)
        deserialized = {}
        for key, value in raw_dict.items():
            hint = hints.get(key)
            deserialized[key] = cls._deserialize_value(value, hint)
        return cls(**deserialized)

    @staticmethod
    def _resolve_serializable(hint):
        """Return the class for a type hint if it has a from_dict classmethod, or None.

        Args:
            hint: A type annotation to inspect.

        Returns:
            The class if it exposes a ``from_dict`` classmethod, else None.
        """
        try:
            if isinstance(hint, type) and callable(getattr(hint, "from_dict", None)):
                return hint
        except TypeError:
            pass
        return None

    @classmethod
    def _deserialize_value(cls, value, hint):
        """Recursively deserialize a single value using its type hint.

        Args:
            value: The raw value from the dict.
            hint: The type annotation for this value, or None.

        Returns:
            The deserialized value.
        """
        if hint is None:
            return value

        target = cls._resolve_serializable(hint)
        if target is not None:
            if isinstance(value, dict):
                return target.from_dict(value)
            return value

        origin = typing.get_origin(hint)
        args = typing.get_args(hint)

        if origin in (list, tuple, set) and args:
            elem_hint = args[0]
            if isinstance(value, (list, tuple, set)):
                deserialized = [cls._deserialize_value(item, elem_hint) for item in value]
                return origin(deserialized)

        if origin is dict and len(args) == 2:
            _, val_hint = args
            if isinstance(value, dict):
                return {k: cls._deserialize_value(v, val_hint) for k, v in value.items()}

        return value

    def to_dict(self) -> dict:
        """Recursively convert the instance to a dict.

        Builds a dict from ``self.__dict__`` and recursively converts any
        nested value that exposes a ``to_dict`` method.  Iterables (list,
        tuple, set) and dict values are traversed element-wise.

        Returns:
            A plain dict representation of the instance.
        """
        return {key: self._serialize_value(value) for key, value in self.__dict__.items()}

    @staticmethod
    def _serialize_value(value):
        """Recursively serialize a single value.

        Args:
            value: The value to serialize.

        Returns:
            The serialized form of the value.
        """
        if hasattr(value, "to_dict"):
            return value.to_dict()

        if isinstance(value, dict):
            return {k: SerializableClass._serialize_value(v) for k, v in value.items()}

        if isinstance(value, (list, tuple, set)):
            serialized = [SerializableClass._serialize_value(item) for item in value]
            return type(value)(serialized)

        return value