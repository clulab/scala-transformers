from task import ShortTaskDef, Task
from tokenizer import Tokenizer
from trainer import OurTrainer

def test_true() -> None:
    assert True == True

def test_false() -> None:
    assert False != True

def test_trainer() -> None:
    tokenizer = Tokenizer.get_pretrained()
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",       "conll-ner/", "train_small.txt",    "train_small.txt",  "train_small.txt"),
        ShortTaskDef("POS",             "pos/", "train_small.txt",    "train_small.txt",  "train_small.txt"),
        ShortTaskDef("Chunking",   "chunking/", "train_small.txt",    "test_small.txt",   "test_small.txt"),
        # ShortTaskDef("Deps Head",  "deps-wsj/", "train_small.heads",  "dev.heads",        "test.heads"),
        # ShortTaskDef("Deps Label", "deps-wsj/", "train_small.labels", "dev.labels",       "test.labels", dual_mode = True)
    ])
    OurTrainer(tokenizer).train(tasks)
    assert 42 == 42