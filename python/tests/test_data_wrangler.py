
from processors.tokenizers import CluTokenizer
from processors.core import DataWrangler
import pathlib

def test_read_label_set() -> None:
    label_set = DataWrangler.read_label_set(
        (pathlib.Path(__file__).parents[0] / "data" / "pos" /"train_sample.txt").resolve()
    )
    assert "#" in label_set
    assert "WDT" in label_set
    assert len(label_set) == 37

def test_read_dataframe() -> None:
    filename = (pathlib.Path(__file__).parents[0] / "data" / "pos" / "train_sample.txt").resolve()
    label_set = DataWrangler.read_label_set(filename)
    label_to_index = {label:index for index, label in enumerate(label_set)} 
    task_id = 0
    tokenizer = CluTokenizer.from_pretrained()
    data_frame = DataWrangler.read_dataframe(filename, label_to_index, task_id, tokenizer)
    for column in data_frame.columns:
        assert len(data_frame[column]) == 11


if __name__ == "__main__":
    # test_read_label_set()
    test_read_dataframe()
