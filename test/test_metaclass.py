from detext.metaclass import SingletonMeta


class Dummy(metaclass=SingletonMeta):
    pass


def test_singleton():
    p = Dummy()
    q = Dummy()
    assert id(p) == id(q)
