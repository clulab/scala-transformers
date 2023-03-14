from task import ShortTaskDef, Task
from tokenizer import Tokenizer
from trainer import OurTrainer

def test_trainer() -> None:
    print("Keith was here!")
    tokenizer = Tokenizer.get_pretrained()
    tasks = Task.mk_tasks("data/", tokenizer, [
        ShortTaskDef("NER",       "conll-ner/", "train_small.txt",    "dev.txt",          "test.txt"),
        ShortTaskDef("POS",             "pos/", "train_small.txt",    "dev.txt",          "test.txt"),
        ShortTaskDef("Chunking",   "chunking/", "train_small.txt",    "test_small.txt",   "test_small.txt")
    ])
    OurTrainer(tokenizer).train(tasks)
    assert 3 == 3
