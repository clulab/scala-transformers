
from names import names

def test_global_names() -> None:
    # assert INPUT_IDS is None
    # assert HEAD_POSITIONS is None

def test_scoped_names() -> None:
    assert names.INPUT_IDS is not None
    assert names.HEAD_POSITIONS is not None
