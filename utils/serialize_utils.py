import typing

TYPE_KEY = "_type"


def register_unique(registry: dict[str, type], cls: type) -> None:
    """Register ``cls`` under ``cls.__name__`` in ``registry`` or raise on conflict.

    Idempotent: re-registering the same class is a no-op.

    Args:
        registry: The mapping to mutate.
        cls: The class to register.
    Raises:
        ValueError: If a different class is already registered under
            ``cls.__name__``.
    """
    name = cls.__name__
    if name in registry and registry[name] is not cls:
        raise ValueError(f"'{name}' is already registered to {registry[name]!r}.")
    registry[name] = cls


class SerializableClass:
    """Abstract base class providing recursive serialization to dict.

    Subclasses are automatically registered in ``_registry`` via
    ``__init_subclass__`` and can be looked up by their class name. The
    ``to_dict`` method embeds a ``_type`` key so that ``from_dict`` can
    reconstruct the correct subclass without dynamic imports.
    """

    # TODO: Make sure init=False is also supported

    _registry: dict[str, type] = {}

    def __init_subclass__(cls, **kwargs):
        """Register every subclass by its class name."""
        super().__init_subclass__(**kwargs)
        register_unique(SerializableClass._registry, cls)

    @classmethod
    def _resolve_class(cls, type_name: str) -> type:
        """Look up a registered subclass by its class name.

        Args:
            type_name: The class name (e.g. ``BoundingBox``).

        Returns:
            The registered class.

        Raises:
            KeyError: If no subclass with that name has been registered.
        """
        if type_name not in cls._registry:
            raise KeyError(f"Unknown serializable type: '{type_name}'")
        return cls._registry[type_name]

    @classmethod
    def create(cls, raw_dict: dict):
        """Factory that instantiates the class specified by ``_type``, scoped to the calling class hierarchy.

        Resolves the class from the registry using the ``_type`` key and
        validates that it is the calling class or a subclass of it before
        delegating to ``from_dict``.

        Args:
            raw_dict: A dict containing a ``_type`` key identifying the
                target class, plus the fields needed to construct it.

        Returns:
            An instance of the class identified by ``_type``.

        Raises:
            KeyError: If ``_type`` is missing or not found in the registry.
            TypeError: If the resolved class is not a subclass of the
                calling class.
        """
        if TYPE_KEY not in raw_dict:
            raise KeyError(f"Cannot create: dict is missing the '{TYPE_KEY}' key.")
        target_cls = cls._resolve_class(raw_dict[TYPE_KEY])
        if not issubclass(target_cls, cls):
            raise TypeError(f"'{target_cls.__name__}' is not a subclass of '{cls.__name__}'.")
        return target_cls.from_dict(raw_dict)

    @classmethod
    def from_dict(cls, raw_dict: dict):
        """Recursively create an instance from a dict.

        If the dict contains a ``_type`` key, the corresponding registered
        class is used for instantiation. Otherwise the calling class is used.

        Args:
            raw_dict: A dict as produced by ``to_dict``.

        Returns:
            An instance of the appropriate class.
        """
        target_cls = cls._resolve_class(raw_dict[TYPE_KEY]) if TYPE_KEY in raw_dict else cls
        hints = typing.get_type_hints(target_cls.__init__)
        hints.pop("return", None)
        deserialized = {}
        for key, value in raw_dict.items():
            if key == TYPE_KEY:
                continue
            hint = hints.get(key)
            deserialized[key] = cls._deserialize_value(value, hint)
        return target_cls(**deserialized)

    @classmethod
    def _deserialize_value(cls, value, hint):
        """Recursively deserialize a single value using its type hint.

        If the value is a dict containing a ``_type`` key, the type is
        resolved from the registry regardless of the hint.

        Args:
            value: The raw value from the dict.
            hint: The type annotation for this value, or None.

        Returns:
            The deserialized value.
        """
        if isinstance(value, dict) and TYPE_KEY in value:
            return cls.from_dict(value)

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

    def to_dict(self) -> dict:
        """Recursively convert the instance to a dict.

        Embeds a ``_type`` key with the fully qualified class path so
        that ``from_dict`` can reconstruct the correct type.

        Returns:
            A plain dict representation of the instance.
        """
        data = {TYPE_KEY: type(self).__name__}
        for key, value in self.__dict__.items():
            data[key] = self._serialize_value(value)
        return data

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
