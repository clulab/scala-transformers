from clu_tokenizer import CluTokenizer
from clu_trainer import CluTrainer
from task import ShortTaskDef, Task

def test_true() -> None:
    assert True == True

def test_false() -> None:
    assert False != True

def test_clu_trainer() -> None:
    tokenizer = CluTokenizer.get_pretrained()
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",       "conll-ner/", "train_small.txt",    "train_small.txt",  "train_small.txt"),
        ShortTaskDef("POS",             "pos/", "train_small.txt",    "train_small.txt",  "train_small.txt"),
        ShortTaskDef("Chunking",   "chunking/", "train_small.txt",    "test_small.txt",   "test_small.txt"),
        # ShortTaskDef("Deps Head",  "deps-wsj/", "train_small.heads",  "dev.heads",        "test.heads"),
        # ShortTaskDef("Deps Label", "deps-wsj/", "train_small.labels", "dev.labels",       "test.labels", dual_mode = True)
    ])
    CluTrainer(tokenizer).train(tasks)
    assert 42 == 42


if __name__ == "__main__":
    test_clu_trainer()
