
from processors.core import Names

def test_global_names() -> None:
    # assert INPUT_IDS is None
    # assert HEAD_POSITIONS is None
    pass

def test_scoped_names() -> None:
    assert Names.INPUT_IDS is not None
    assert Names.HEAD_POSITIONS is not None
