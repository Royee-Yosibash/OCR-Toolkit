import unittest

from utils.serialize_utils import SerializableClass


class Inner(SerializableClass):

    def __init__(self, value: int):
        self.value = value


class Outer(SerializableClass):

    def __init__(
        self,
        name: str,
        inner: Inner,
        items: list[Inner] = None,
        mapping: dict[str, Inner] = None,
    ):
        self.name = name
        self.inner = inner
        self.items = items or []
        self.mapping = mapping or {}


class TestSerializableClass(unittest.TestCase):

    def test_simple_to_dict(self):
        obj = Inner(42)
        self.assertEqual(obj.to_dict(), {"value": 42})

    def test_nested_to_dict(self):
        obj = Outer(name="test", inner=Inner(7))
        result = obj.to_dict()
        self.assertEqual(result["name"], "test")
        self.assertEqual(result["inner"], {"value": 7})

    def test_list_of_serializable(self):
        obj = Outer(name="x", inner=Inner(1), items=[Inner(2), Inner(3)])
        result = obj.to_dict()
        self.assertEqual(result["items"], [{"value": 2}, {"value": 3}])

    def test_dict_of_serializable(self):
        obj = Outer(name="x", inner=Inner(0), mapping={"a": Inner(10)})
        result = obj.to_dict()
        self.assertEqual(result["mapping"], {"a": {"value": 10}})

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


if __name__ == "__main__":
    unittest.main()
