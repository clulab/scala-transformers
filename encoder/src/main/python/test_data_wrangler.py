
from clu_tokenizer import CluTokenizer
from data_wrangler import DataWrangler

def test_read_label_set() -> None:
    label_set = DataWrangler.read_label_set("data/pos/train_small.txt")
    assert "#" in label_set
    assert "WRB" in label_set
    assert len(label_set) == 45

def test_read_dataframe() -> None:
    filename = "data/pos/train_small.txt"
    label_set = DataWrangler.read_label_set(filename)
    label_to_index = {label:index for index, label in enumerate(label_set)} 
    task_id = 0
    tokenizer = CluTokenizer.get_pretrained()
    data_frame = DataWrangler.read_dataframe(filename, label_to_index, task_id, tokenizer)
    for column in data_frame.columns:
        assert len(data_frame[column]) == 201


if __name__ == "__main__":
    # test_read_label_set()
    test_read_dataframe()
