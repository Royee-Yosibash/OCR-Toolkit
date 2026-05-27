import unittest

from utils.serialize_utils import TYPE_KEY, SerializableClass


class Inner(SerializableClass):
    """Leaf fixture class with a single int field used as a nested serializable."""

    def __init__(self, value: int):
        self.value = value


class InnerChild(Inner):
    """Subclass of ``Inner`` used to exercise registry-based polymorphic deserialization."""

    def __init__(self, value: int, extra: str = ""):
        super().__init__(value)
        self.extra = extra


class Outer(SerializableClass):
    """Container fixture that nests ``Inner`` directly and via list/dict for round-trip tests."""

    def __init__(
        self,
        name: str,
        inner: Inner,
        items: list[Inner] | None = None,
        mapping: dict[str, Inner] | None = None,
    ):
        self.name = name
        self.inner = inner
        self.items = items or []
        self.mapping = mapping or {}


INNER_TYPE = "Inner"
OUTER_TYPE = "Outer"


class TestSerializableClass(unittest.TestCase):
    """Tests for ``SerializableClass`` ``to_dict`` / ``from_dict`` round-trips and nesting."""

    def test_simple_to_dict(self):
        obj = Inner(42)
        self.assertEqual(obj.to_dict(), {TYPE_KEY: INNER_TYPE, "value": 42})

    def test_nested_to_dict(self):
        obj = Outer(name="test", inner=Inner(7))
        result = obj.to_dict()
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["inner"], {TYPE_KEY: INNER_TYPE, "value": 7})

    def test_list_of_serializable(self):
        obj = Outer(name="x", inner=Inner(1), items=[Inner(2), Inner(3)])
        result = obj.to_dict()
        self.assertEqual(
            result["items"],
            [
                {TYPE_KEY: INNER_TYPE, "value": 2},
                {TYPE_KEY: INNER_TYPE, "value": 3},
            ],
        )

    def test_dict_of_serializable(self):
        obj = Outer(name="x", inner=Inner(0), mapping={"a": Inner(10)})
        result = obj.to_dict()
        self.assertEqual(result["mapping"], {"a": {TYPE_KEY: INNER_TYPE, "value": 10}})

    def test_plain_values_unchanged(self):
        obj = Outer(name="hello", inner=Inner(1), items=[1, "two", 3.0])
        result = obj.to_dict()
        self.assertEqual(result["items"], [1, "two", 3.0])

    def test_tuple_preserved(self):
        obj = Inner((1, 2))
        result = obj.to_dict()
        self.assertEqual(result["value"], (1, 2))

    def test_set_preserved(self):
        obj = Inner(frozenset())
        obj.value = {1, 2, 3}
        result = obj.to_dict()
        self.assertIsInstance(result["value"], set)
        self.assertEqual(result["value"], {1, 2, 3})

    def test_simple_from_dict(self):
        obj = Inner.from_dict({"value": 42})
        self.assertIsInstance(obj, Inner)
        self.assertEqual(obj.value, 42)

    def test_nested_from_dict(self):
        data = {"name": "test", "inner": {"value": 7}}
        obj = Outer.from_dict(data)
        self.assertIsInstance(obj, Outer)
        self.assertEqual(obj.name, "test")
        self.assertIsInstance(obj.inner, Inner)
        self.assertEqual(obj.inner.value, 7)

    def test_list_from_dict(self):
        data = {
            "name": "x",
            "inner": {"value": 1},
            "items": [{"value": 2}, {"value": 3}],
        }
        obj = Outer.from_dict(data)
        self.assertEqual(len(obj.items), 2)
        self.assertIsInstance(obj.items[0], Inner)
        self.assertEqual(obj.items[0].value, 2)
        self.assertEqual(obj.items[1].value, 3)

    def test_dict_from_dict(self):
        data = {
            "name": "x",
            "inner": {"value": 0},
            "mapping": {"a": {"value": 10}},
        }
        obj = Outer.from_dict(data)
        self.assertIn("a", obj.mapping)
        self.assertIsInstance(obj.mapping["a"], Inner)
        self.assertEqual(obj.mapping["a"].value, 10)

    def test_roundtrip(self):
        original = Outer(
            name="rt",
            inner=Inner(5),
            items=[Inner(6), Inner(7)],
            mapping={"k": Inner(8)},
        )
        restored = Outer.from_dict(original.to_dict())
        self.assertEqual(restored.to_dict(), original.to_dict())


class TestCreate(unittest.TestCase):
    """Tests for ``SerializableClass.create``: registry-driven class resolution from a dict."""

    def test_create_from_base_class(self):
        data = {TYPE_KEY: "Inner", "value": 42}
        obj = SerializableClass.create(data)
        self.assertIsInstance(obj, Inner)
        self.assertEqual(obj.value, 42)

    def test_create_child_from_parent(self):
        data = {TYPE_KEY: "InnerChild", "value": 10, "extra": "hi"}
        obj = Inner.create(data)
        self.assertIsInstance(obj, InnerChild)
        self.assertEqual(obj.value, 10)
        self.assertEqual(obj.extra, "hi")

    def test_create_same_class(self):
        data = {TYPE_KEY: "Inner", "value": 5}
        obj = Inner.create(data)
        self.assertIsInstance(obj, Inner)
        self.assertEqual(obj.value, 5)

    def test_create_rejects_unrelated_class(self):
        data = {TYPE_KEY: "Inner", "value": 1}
        with self.assertRaises(TypeError):
            Outer.create(data)

    def test_create_missing_type_key(self):
        with self.assertRaises(KeyError):
            Inner.create({"value": 1})

    def test_create_unknown_type(self):
        with self.assertRaises(KeyError):
            SerializableClass.create({TYPE_KEY: "NonExistent"})

    def test_create_roundtrip(self):
        original = InnerChild(value=99, extra="round")
        restored = Inner.create(original.to_dict())
        self.assertIsInstance(restored, InnerChild)
        self.assertEqual(restored.to_dict(), original.to_dict())


if __name__ == "__main__":
    unittest.main()
